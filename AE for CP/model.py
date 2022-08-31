import torch
import torch.nn as nn
from torchsummary import summary


class PointEncoder(nn.Module):
    def __init__(self, num_points=2500):
        super(PointEncoder, self).__init__()
        # batch_size = point_cloud.get_shape()[0].value
        # num_point = point_cloud.get_shape()[1].value
        # point_dim = point_cloud.get_shape()[2].value

        # input: Bx3xN batch_size(32)*3*2500

        # output: Bx3xN
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

    def forward(self, x):
        return self.encoder(x)


class PointDecoder(nn.Module):
    def __init__(self, num_points=2500) -> None:
        super(PointDecoder, self).__init__()
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
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, num_points) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = PointEncoder(num_points)

        self.downsample = nn.MaxPool1d(num_points)

        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        # print("Original shape is : ", x.shape)
        batch_size, point_dim, n_points = x.shape[0], x.shape[1], x.shape[2]
        # print(batch_size, point_dim, n_points)

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
        # return decoder


if __name__ == '__main__':
    device = "cuda"
    # summary(AutoEncoder(2500).to(device), input_size=(3, 2500), batch_size=32)
    model = AutoEncoder(2500).to(device)
    model.eval()
    inputs = torch.zeros((1, 3, 2500)).to(device)
    output, _ = model(inputs)
    print(output.shape)
    # outputs, features = model(inputs)
    # from chamfer_distance.chamfer_distance_gpu import ChamferDistance

    # chamfer loss, the tensor shape has to be (batch_size, num_points, num_dim).
    # loss_function = ChamferDistance()

    # outputs = outputs.transpose(1, 2)
    # inputs = inputs.transpose(1, 2)

    # dist1, dist2 = loss_function(outputs, inputs)
    # loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    # print(outputs.shape)
    # print(loss)
