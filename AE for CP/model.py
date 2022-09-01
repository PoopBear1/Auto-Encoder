import torch
import torch.nn as nn
from torchsummary import summary


class PointEncoder(nn.Module):
    def __init__(self):
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
        self.encoder = PointEncoder()

        self.downsample = nn.MaxPool1d(num_points)

        self.decoder = PointDecoder(num_points)

    def forward(self, x):

        batch_size, point_dim, n_points = x.shape[0], x.shape[1], x.shape[2]
        encoder = self.encoder_forward(x)

        out = self.downsample(encoder)

        out = out.view(-1, 1024)
        global_feat = out

        decoder = self.decoder_forward(out)
        decoder = torch.reshape(decoder, (batch_size, point_dim, n_points))

        return decoder, global_feat

    def encoder_forward(self, x):
        encoder = self.encoder(x)
        return encoder

    def decoder_forward(self, x):
        out = self.decoder(x)
        return out

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
