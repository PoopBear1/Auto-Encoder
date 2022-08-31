import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PointNetAE(nn.Module):
    def __init__(self, num_points=2048):
        super(PointNetAE, self).__init__()
        self.num_points = num_points
        # self.encoder = nn.Sequential(
        # PointNetfeat(num_points, global_feat=True, trans = False),
        # nn.Linear(1024, 256),
        # nn.ReLU(),
        # nn.Linear(256, 100),
        # )

        self.encoder = PointEncoder(num_points)
        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        x = self.encoder(x)

        # encoded_embedding = x

        x = self.decoder(x)

        # return x, encoded_embedding

        return x


class PointEncoder(nn.Module):
    def __init__(self, num_points = 2500):
        super(PointEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 256)
        self.fc2 = torch.nn.Linear(256, 100)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x