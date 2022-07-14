import torch


class Net(torch.nn.Module):
    def __init__(self, init_model):
        super().__init__()
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv1.stride = (1, 1)
        self.finalpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(512, 300, bias=False)
        self.fc_class = torch.nn.Linear(300, 2000, bias=False)

    def forward(self, x):
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
        y = self.fc_class(x)
        return x, x, y, x


class LandmarkNet(torch.nn.Module):
    def __init__(self, init_model, num_landmarks=8):
        super().__init__()
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.layer3[0].downsample[0].stride = (1, 1)
        self.layer3[0].conv1.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.layer4[0].conv1.stride = (1, 1)
        self.fc = torch.nn.Conv2d(512, 300, 1, bias=False)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.fc_global = torch.nn.Conv2d(512,512,1)
        self.fc_landmarks = torch.nn.Conv2d(512, num_landmarks + 1, 1,
                                            bias=False)
        self.fc_class = torch.nn.Linear(300, 2000, bias=False)
        # self.landmark_mask = torch.nn.Parameter(torch.zeros(1,300,10-1))
        # torch.nn.init.normal_(self.landmark_mask,std=1)
        # self.landmark_proj = torch.nn.Parameter(torch.Tensor(300,300,num_landmarks))
        # torch.nn.init.kaiming_uniform_(self.landmark_proj)
        # self.landmark_proj = torch.nn.Parameter(torch.eye(300,300).unsqueeze(-1).repeat(1,1,10-1))
        # self.dropout = torch.nn.Dropout2d(0.5)
        self.softmax = torch.nn.Softmax2d()
        self.avg_dist_pos = torch.nn.Parameter(torch.zeros([num_landmarks]),
                                               requires_grad=False)
        self.avg_dist_neg = torch.nn.Parameter(torch.zeros([num_landmarks]),
                                               requires_grad=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        maps = self.fc_landmarks(x)
        maps = self.softmax(maps)
        x = self.fc(x)
        feature_tensor = x
        x = (maps[:, 0:-1, :, :].unsqueeze(-1).permute(0, 4, 2, 3,
                                                       1) * x.unsqueeze(
            -1)).mean(2).mean(2)
        # x = x * (self.landmark_mask).sigmoid()
        # new_x = []
        # for i in range(x.shape[-1]):
        #    new_x.append(torch.matmul(x[:,:,i],self.landmark_proj[:,:,i]).unsqueeze(-1))
        # x = torch.cat(new_x,2)
        # x = x.sum(2)
        # x = torch.nn.functional.normalize(x)
        y = self.fc_class(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x, maps, y, feature_tensor
