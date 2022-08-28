import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from torch.autograd import Variable
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

        # return decoder, global_feat
        return decoder





def train(train_loader, valid_loader, config):
    device = config["device"]
    loss_function = config["loss_function"]
    model = AutoEncoder(config["num_points"]).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["regular_constant"],
    )
    train_loss_value = []
    validate_loss_value = []
    train_accuracy_value = []
    validate_accuracy_value = []
    current_epoch = []
    low_loss = torch.tensor(float('inf')).cuda()

    print("####### Training Processing on {}#######".format(device))
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        train_accuracy = 0
        current_epoch.append(epoch + 1)
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # print(outputs[0].shape, "here\nhere", inputs[0].shape)
            dist1, dist2 = loss_function(outputs, inputs)
            loss = (torch.mean(dist1) + torch.mean(dist2))
            loss.backward()
            loss_value = loss.item()
            optimizer.step()
            train_loss += loss_value
            total += targets.size(0)

        train_loss /= len(train_loader.dataset)
        train_loss_value.append(loss_value)

        # Validation step
        model.eval()
        validation_loss = 0
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            dist1, dist2 = loss_function(outputs, inputs)
            validation_loss += (torch.mean(dist1) + torch.mean(dist2)).item()
        validation_loss /= len(valid_loader.dataset)
        print("\nIn epoch: ", epoch + 1)
        print("\nTraining set: Avg. loss: {:.6f}".format(train_loss))
        print("\nValidation set: Avg. loss: {:.6f}".format(validation_loss))
        validate_loss_value.append(validation_loss)

        if validation_loss < low_loss:
            low_loss = validation_loss

            torch.save(model.state_dict(), os.path.join(os.getcwd(), "ckpt.pth"))
            print("model save at checkpoint")

    plt.plot(current_epoch, train_loss_value, "b", label="Training Loss")
    plt.savefig(os.path.join(os.getcwd(), "loss_curve.jpg"))
    plt.figure()
    return model


def test(test_loader, model, config):
    model.eval()
    test_loss = 0
    device = config["device"]
    loss_function = config["loss_function"]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            dist1, dist2 = loss_function(output, data)
            loss = (torch.mean(dist1) + torch.mean(dist2))
            test_loss += loss.item()
        test_loss /= len(test_loader.dataset)

    print("\nTest set: Avg. loss: {:.6f}".format(test_loss))

# device = "cuda"
# summary(AutoEncoder(2500).to(device), input_size=(3, 2500), batch_size=32)