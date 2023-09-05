import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
from typing import Tuple

# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class IndividualLandmarkNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000, landmark_dropout_rate: int = 0.5) -> None:
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

        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        # 1024 and 2048 are the dimensions of the 3rd and 4th ResNet layer
        # outputs, respectively
        self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(1024 + 2048, num_classes, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1,1024 + 2048,num_landmarks + 1)))

        self.dropout = torch.nn.Dropout(0.5)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.upsample_bilinear(x, size=(l3.shape[-2], l3.shape[-1]))
        x = torch.cat((x, l3), dim=1)
        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        batch_size = x.shape[0]

        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2], x.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)

        maps = b_sq - 2 * ab + a_sq
        maps = -maps

        # Softmax so that the attention maps for each pixel add up to 1
        maps = self.softmax(maps)

        # Use maps to get weighted average features per landmark
        feature_tensor = x  # torch.cat((x, l3), dim=1)
        all_features = ((maps).unsqueeze(1) * x.unsqueeze(2)).mean(-1).mean(-1)

        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0,2,1)).permute(0,2,1)
        y = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        classification = y[:, :, :-1].mean(-1)

        return all_features, maps, y, feature_tensor, classification
