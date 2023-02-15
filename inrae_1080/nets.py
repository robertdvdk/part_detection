import torch
from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, AdaptiveAvgPool2d, Linear, Softmax2d, Parameter
from torchvision.models.resnet import ResNet
from typing import Tuple, Optional
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import numpy as np

# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class Net(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_classes: int=2000) -> None:
        super().__init__()
        self.conv1: Conv2d = init_model.conv1
        self.bn1: BatchNorm2d = init_model.bn1
        self.relu: ReLU = init_model.relu
        self.maxpool: MaxPool2d = init_model.maxpool
        self.layer1: Sequential = init_model.layer1
        self.layer2: Sequential = init_model.layer2
        self.layer3: Sequential = init_model.layer3
        self.layer4: Sequential = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv1.stride = (1, 1)
        self.finalpool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc: Linear = torch.nn.Linear(512, 300, bias=False)
        self.fc_class: Linear = torch.nn.Linear(512, num_classes, bias=False)


    def forward(self, x: Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.finalpool(x).squeeze(-1).squeeze(-1)
        # x = self.fc(x)
        y: Tensor = self.fc_class(x)
        return x, x, y, x, x

# Proposed landmark-based model, also based on ResNet
class LandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int=8, num_classes: int=2000) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.conv1: Conv2d = init_model.conv1
        self.bn1: BatchNorm2d = init_model.bn1
        self.relu: ReLU = init_model.relu
        self.maxpool: MaxPool2d = init_model.maxpool
        self.layer1: Sequential = init_model.layer1
        self.layer2: Sequential = init_model.layer2
        self.layer3: Sequential = init_model.layer3
        self.layer4: Sequential = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv2.stride = (1, 1)
        self.finalpool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_class = torch.nn.Linear(2048, 200, bias=False)
        self.fc_landmarks = torch.nn.Conv2d(2048, num_landmarks + 1, 1, bias=False)
        self.softmax = torch.nn.Softmax2d()


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Compute per landmark attention maps
        maps = self.fc_landmarks(x)
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        feature_tensor: Tensor = x
        x = (maps[:, 0:-1, :, :].unsqueeze(-1).permute(0, 4, 2, 3,1) * x.unsqueeze(-1)).mean(2).mean(2)

        # Classification based on the landmarks
        y = self.fc_class(x.permute(0, 2, 1)).permute(0, 2, 1)
        classification = y.mean(-1)

        return x, maps, y, feature_tensor, classification