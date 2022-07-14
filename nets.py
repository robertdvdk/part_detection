import torch
from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Sequential, AdaptiveAvgPool2d, Linear, Softmax2d, Parameter
from torchvision.models.resnet import ResNet
from typing import Tuple

# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class Net(torch.nn.Module):
    def __init__(self, init_model: ResNet) -> None:
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
        self.finalpool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc: Linear = torch.nn.Linear(512, 300, bias=False)
        self.fc_class: Linear = torch.nn.Linear(300, 2000, bias=False)

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
        x = self.fc(x)
        y: Tensor = self.fc_class(x)
        return x, x, y, x

# Proposed landmark-based model, also based on ResNet
class LandmarkNet(torch.nn.Module):

    def __init__(self, init_model: ResNet, num_landmarks: int=8) -> None:
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
        self.fc: Conv2d = torch.nn.Conv2d(512, 300, 1, bias=False)
        self.pool: AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc_global = torch.nn.Conv2d(512,512,1)
        self.fc_landmarks: Conv2d = torch.nn.Conv2d(512, num_landmarks + 1, 1,
                                            bias=False)
        self.fc_class: Linear = torch.nn.Linear(300, 2000, bias=False)
        # self.landmark_mask = torch.nn.Parameter(torch.zeros(1,300,10-1))
        # torch.nn.init.normal_(self.landmark_mask,std=1)
        # self.landmark_proj = torch.nn.Parameter(torch.Tensor(300,300,num_landmarks))
        # torch.nn.init.kaiming_uniform_(self.landmark_proj)
        # self.landmark_proj = torch.nn.Parameter(torch.eye(300,300).unsqueeze(-1).repeat(1,1,10-1))
        # self.dropout = torch.nn.Dropout2d(0.5)
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.avg_dist_pos: Parameter = torch.nn.Parameter(torch.zeros([num_landmarks]),
                                               requires_grad=False)
        self.avg_dist_neg: Parameter = torch.nn.Parameter(torch.zeros([num_landmarks]),
                                               requires_grad=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Compute per landmark attention maps
        maps: Tensor = self.fc_landmarks(x)
        maps = self.softmax(maps)
        
        # Use maps to get weighted average features per landmark
        x = self.fc(x)
        feature_tensor: Tensor = x
        x = (maps[:, 0:-1, :, :].unsqueeze(-1).permute(0, 4, 2, 3,1) * x.unsqueeze(-1)).mean(2).mean(2)
        
        # x = x * (self.landmark_mask).sigmoid()
        # new_x = []
        # for i in range(x.shape[-1]):
        #    new_x.append(torch.matmul(x[:,:,i],self.landmark_proj[:,:,i]).unsqueeze(-1))
        # x = torch.cat(new_x,2)
        # x = x.sum(2)
        # x = torch.nn.functional.normalize(x)
        
        y: Tensor = self.fc_class(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x, maps, y, feature_tensor
