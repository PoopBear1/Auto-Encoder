import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torchsummary import summary

class AutoEncoder(nn.Module):
    def __init__(self, num_points) -> None:
        super(AutoEncoder, self).__init__()
        # batch_size = point_cloud.get_shape()[0].value
        # num_point = point_cloud.get_shape()[1].value
        # point_dim = point_cloud.get_shape()[2].value

        # input: BxNx3 batch_size(32)*2500*3

        # output: BxNx3
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

        )

        self.downsample = nn.MaxPool1d(num_points)

        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(1024),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),

            nn.Linear(1024, num_points * 3),
        )

    def forward(self, x):
        batch_size, point_dim, n_points = x.shape[0], x.shape[1], x.shape[2]
        # print(batch_size, point_dim, n_points)
        # print("Original shape is : ", x.shape)

        encoder = self.encoder(x)
        # print("After encoding  the shape is : ", encoder.shape)

        out = self.downsample(encoder)
        # print("The shape of max pooling is ", out.shape)

        out = out.view(-1, 1024)

        global_feat = out
        # print("Before decoder, the Reshape size:", out.shape)

        decoder = self.decoder(out)
        decoder = torch.reshape(decoder, (batch_size, point_dim, n_points))
        # print("back decoding, the shape is : ", decoder.shape)

        return decoder, global_feat
        # return out



if torch.cuda.is_available():
    from chamfer_distance.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
    device = torch.device("cuda")
else:
    from chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
    device = torch.device("cpu")

# summary(AutoEncoder(2500).to(device), input_size=(3, 2500))

def train(train_loader,valid_loader,config):
    model = AutoEncoder(config["num_points"])
    loss_function = ChamferDistance()
    for epoch in config["num_epochs"]:

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            pass

        pass

    return model

def test(test_loader):

    return