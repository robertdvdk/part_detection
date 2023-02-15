import torch
from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, \
    AdaptiveAvgPool2d, Linear, Softmax2d, Parameter
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from typing import Tuple, Optional
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import numpy as np
from lib import landmark_coordinates


# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class Net(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_classes: int = 2000) -> None:
        super().__init__()
        self.conv1: Conv2d = init_model.conv1
        self.bn1: BatchNorm2d = init_model.bn1
        self.relu: ReLU = init_model.relu
        self.maxpool: MaxPool2d = init_model.maxpool
        self.layer1: Sequential = init_model.layer1
        self.layer2: Sequential = init_model.layer2
        self.layer3: Sequential = init_model.layer3
        self.layer4: Sequential = init_model.layer4

        # ResNet18
        # self.layer3[0].downsample[0].stride = (1, 1)
        # self.layer3[0].conv1.stride = (1, 1)
        # self.layer4[0].downsample[0].stride = (1, 1)
        # self.layer4[0].conv1.stride = (1, 1)

        # ResNet50
        # self.layer3[0].downsample[0].stride = (1, 1)
        # self.layer3[0].conv2.stride = (1, 1)
        # self.layer4[0].downsample[0].stride = (1, 1)
        # self.layer4[0].conv2.stride = (1, 1)

        self.finalpool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)

        # ResNet18
        # self.fc_class: Linear = torch.nn.Linear(512, num_classes, bias=False)

        # ResNet50
        # self.dropout = torch.nn.Dropout(0.5)
        self.fc_class: Linear = torch.nn.Linear(2048, num_classes, bias=False)

    def forward(self, x: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.finalpool(x).squeeze(-1).squeeze(-1)
        y: Tensor = self.fc_class(x)
        return x, x, y, x, x


# Proposed landmark-based model, also based on ResNet
class LandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4

        # ResNet18
        # self.layer3[0].downsample[0].stride = (1, 1)
        # self.layer3[0].conv1.stride = (1, 1)
        # self.layer4[0].downsample[0].stride = (1, 1)
        # self.layer4[0].conv1.stride = (1, 1)

        # ResNet50
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv2.stride = (1, 1)

        self.fc: Conv2d = torch.nn.Conv2d(2048, 500, 1, bias=False)
        self.summarize = torch.nn.Conv2d(500, 30, kernel_size=(8, 8),
                                         stride=(8, 8))
        self.pool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        self.fc_landmarks = torch.nn.Conv2d(2048, num_landmarks + 1, 1,
                                            bias=False)
        self.drop = torch.nn.Dropout(0.5)
        self.fc_class_landmarks = torch.nn.Linear(500, num_classes, bias=False)
        self.softmax: Softmax2d = torch.nn.Softmax2d()

        self.class_token = torch.nn.Parameter(
            torch.rand(1, 500 + self.num_landmarks))
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=500 + self.num_landmarks, num_heads=2, bias=False,
            batch_first=False, dropout=0.1)
        self.mha2 = torch.nn.MultiheadAttention(
            embed_dim=500 + self.num_landmarks, num_heads=2, bias=False,
            batch_first=False, dropout=0.1)
        self.vitblock = torch.nn.Sequential(
            torch.nn.Linear(500 + self.num_landmarks,
                            2 * (500 + self.num_landmarks)),
            torch.nn.GELU(),
            torch.nn.Linear(2 * (500 + self.num_landmarks),
                            500 + self.num_landmarks)
        )
        self.vitblock2 = torch.nn.Sequential(
            torch.nn.Linear(500 + self.num_landmarks,
                            2 * (500 + self.num_landmarks)),
            torch.nn.GELU(),
            torch.nn.Linear(2 * (500 + self.num_landmarks),
                            500 + self.num_landmarks)
        )
        self.fc_class_attention: Linear = torch.nn.Linear(
            500 + self.num_landmarks, num_classes, bias=False)
        self.layernorm = torch.nn.LayerNorm(510)
        self.attdrop = torch.nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # # Compute per landmark attention maps
        maps = self.fc_landmarks(x)
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        x = self.fc(x)
        feature_tensor: Tensor = x
        x = (maps[:, 0:-1, :, :].unsqueeze(-1).permute(0, 4, 2, 3,
                                                       1) * x.unsqueeze(
            -1)).mean(2).mean(2)

        # Use multihead attention to look at all feature vectors simultaneously and combine them
        identity = torch.eye(self.num_landmarks, requires_grad=True).repeat(
            x.size(dim=0), 1, 1).to(x.get_device())
        att_input = torch.permute(torch.cat((identity, x), dim=1), (2, 0, 1))
        att_input = torch.cat(
            (att_input, self.class_token.repeat(1, x.size(dim=0), 1)), dim=0)

        att, _ = self.mha(att_input, att_input, att_input, need_weights=False)
        att = self.attdrop(att)
        att += att_input
        att = self.layernorm(att)
        att2 = self.vitblock(att)
        att2 += att

        att2 = self.layernorm(att2)
        att, _ = self.mha2(att2, att2, att2, need_weights=False)
        att = self.attdrop(att)
        att += att2
        att = self.layernorm(att)
        att2 = self.vitblock2(att)
        att2 += att

        # Classification based on the landmarks
        x = self.drop(x)
        y = self.fc_class_landmarks(x.permute(0, 2, 1)).permute(0, 2, 1)
        # classification = y.mean(-1)

        # Classification based on the MHA output
        class_token = att2[0, :, :]
        # class_token = self.drop(class_token)
        # classification = self.fc_class_attention(self.drop(class_token))
        classification = self.fc_class_attention(class_token)

        return x, maps, y, feature_tensor, classification


class NewLandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000, height=256) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(1024 + 2048, 1024, 1, bias=True)

        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(1024 + 2048, num_classes, bias=False)
        self.upsample = torch.nn.Upsample(size=(height // 16, height // 16), mode='bilinear')

        self.mha = torch.nn.MultiheadAttention(1024 + 2048 + self.num_landmarks, 4)
        self.fc_class_attention: Linear = torch.nn.Linear(1024 + 2048 + self.num_landmarks, num_classes, bias=False)
        self.dropout = torch.nn.Dropout(0.7)

        # self.mha2 = torch.nn.MultiheadAttention(1024 + 2048 + self.num_landmarks, 4)

    def forward(self, x: Tensor) -> Tuple[  Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = self.upsample(x)
        x = torch.cat((x, l3), dim=1)
        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        feature_map_size = x.shape[-1]
        batch_size = x.shape[0]

        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, feature_map_size, feature_map_size)
        a_sq = a_sq.permute(1, 0, 2, 3)

        maps = b_sq - 2 * ab + a_sq

        # Softmax so that the attention maps for each pixel add up to 1
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        feature_tensor = x  # torch.cat((x, l3), dim=1)
        all_features = ((maps).unsqueeze(1) * x.unsqueeze(2)).mean(-1).mean(-1)

        # Classification based on the landmarks
        y = self.fc_class_landmarks(all_features.permute(0, 2, 1)).permute(0, 2, 1)

        x = all_features[:, :, 0:-1]
        identity = torch.eye(self.num_landmarks, requires_grad=True).repeat(x.size(dim=0), 1, 1).to(x.get_device())
        x = self.dropout(x)
        att_input = torch.permute(torch.cat((identity, x), dim=1), (2, 0, 1)).detach()
        att, _ = self.mha(att_input, att_input, att_input, need_weights=False)
        # att = att.relu()/2 + att_input/2

        # att1 = self.dropout(att)
        # att1, _ = self.mha2(att1, att1, att1, need_weights=False)
        # att1 = att1.relu()/2 + att/2
        #
        classification = self.fc_class_attention(self.dropout(att.mean(0).relu()))
        # att1 = self.dropout(att1)
        # classification = self.fc_class_attention(att1).mean(0)

        return all_features, maps, y, feature_tensor, classification