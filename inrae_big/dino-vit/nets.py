import torch
from torch import Tensor
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
import torch.nn.functional as F
from typing import Tuple
import numpy as np

# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class IndividualLandmarkNet(torch.nn.Module):
    def __init__(self, partnet, classnet, num_landmarks: int = 8,
                 num_classes: int = 200) -> None:
        super().__init__()
        self.partnet = partnet
        self.classnet = classnet

        self.num_landmarks = num_landmarks

        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        self.fc_landmarks = torch.nn.Conv2d(384, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(384, num_classes, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1, 384, num_landmarks + 1)))

        self.dropout_full_landmarks = torch.nn.Dropout1d(0.3)

    def forward(self, x: Tensor) -> Tuple[  Tensor, Tensor, Tensor, Tensor, Tensor]:
        ims = x
        # Pretrained ViT part of the model
        batch_size = x.shape[0]
        with torch.no_grad():
            x = self.partnet(x)
            # for i in features[1:]:
            #     x = torch.cat((x, i), dim=2)
            x = x.permute(0, 2, 1)
            height = int(np.sqrt(x.shape[-1]))
            x = torch.reshape(x, (batch_size, -1, height, height))

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        # maps = self.fc_landmarks(x)

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
        feature_tensor = x
        all_features = ((maps).unsqueeze(1) * x.unsqueeze(2)).mean(-1).mean(-1)

        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0,2,1)).permute(0,2,1)
        y = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        classification = y[:, :, :-1].mean(-1)


        maps_upsampled = F.interpolate(maps[:, :-1, :, :], mode='bilinear',
                    size=(ims.shape[0], -1, ims.shape[-1], ims.shape[-1]))

        return all_features, maps, y, feature_tensor, classification