import numpy as np
import torch
import os
import torch.optim as optim
from model import AutoEncoder, PointEncoder, PointDecoder
# from modelV2 import PointNetAE
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
from Dataset import pc_normalize


def train(train_loader, valid_loader, config):
    device = config["device"]
    loss_function = config["loss_function"]
    # model = AutoEncoder(config["num_points"]).to(device)
    model = AutoEncoder(config["num_points"]).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["regular_constant"],
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # train_accuracy_value = []
    # validate_accuracy_value = []

    low_loss = torch.tensor(float('inf')).cuda()
    writer = SummaryWriter(os.path.join(os.getcwd(), "exp_data", config["class"], "runs"))
    latent_vector_all = torch.Tensor().to(device)
    filename_all = list()
    print("####### Training Processing on {}#######".format(device))
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        train_accuracy = 0
        current_epoch = epoch + 1
        for batch_idx, (inputs, file) in enumerate(train_loader):
            # print("in dataloader: ", inputs.shape)
            inputs = inputs.transpose(2, 1)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs, feature_vector = model(inputs)
            latent_vector_all = torch.cat((latent_vector_all, feature_vector), 0)

            filename_all.extend(file)
            # print(inputs.shape,outputs.shape)
            inputs = inputs.transpose(1, 2)
            outputs = outputs.transpose(1, 2)

            # print(outputs[0].shape, "here\nhere", inputs[0].shape)
            dist1, dist2 = loss_function(outputs, inputs)
            loss = (torch.mean(dist1) + torch.mean(dist2))
            loss.backward()
            loss_value = loss.item()
            optimizer.step()
            train_loss += loss_value

        train_loss /= len(train_loader.dataset)

        # Validation step
        model.eval()
        validation_loss = 0
        for batch_idx, (inputs, file) in enumerate(valid_loader):
            inputs = inputs.transpose(2, 1)
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            inputs = inputs.transpose(1, 2)
            outputs = outputs.transpose(1, 2)

            # print(outputs.shape, inputs.shape)
            dist1, dist2 = loss_function(outputs, inputs)
            validation_loss += (torch.mean(dist1) + torch.mean(dist2)).item()
        validation_loss /= len(valid_loader.dataset)

        # exit(-1)

        writer.add_scalar("loss/Train_loss_value", loss_value, current_epoch)
        writer.add_scalar("loss/validate_loss_value", validation_loss, current_epoch)

        print("\nIn epoch: ", epoch + 1)
        print("\nTraining set: Avg. loss: {:.6f}".format(train_loss))
        print("\nValidation set: Avg. loss: {:.6f}".format(validation_loss))

        if validation_loss < low_loss:
            low_loss = validation_loss

            torch.save(model.state_dict(), os.path.join(os.getcwd(), "exp_data", config["class"], "ckpt.pth"))
            print("model save at checkpoint")
        # writer.add_images("reconstruction", outputs, global_step=None, walltime=None, dataformats='NCHW')
        scheduler.step()

    # best_feature_vector(device, filename_all, latent_vector_all, train_loader, model)

    return model


def test(test_loader, model, config):
    model.eval()
    test_loss = 0
    device = config["device"]
    loss_function = config["loss_function"]

    with torch.no_grad():
        for data, file in test_loader:
            data = data.to(device)
            data = data.transpose(2, 1)
            output, _ = model(data)

            data = data.transpose(1, 2)
            output = output.transpose(1, 2)
            dist1, dist2 = loss_function(output, data)
            loss = (torch.mean(dist1) + torch.mean(dist2))
            test_loss += loss.item()
        test_loss /= len(test_loader.dataset)

    print("\nTest set: Avg. loss: {:.6f}".format(test_loss))


def best_feature_vector(config, train_dl, model):
    print("Generate the Best Latent Vectors")
    device = config["device"]
    path = os.path.join(os.getcwd(), "exp_data", config["class"])
    with torch.no_grad():
        all_latent_vector = torch.Tensor().to(device)
        decoder = PointDecoder(config["num_points"]).to(device)
        autoencoder_eval = model.eval()  # set the network in evaluation mode
        for itrid, data in enumerate(train_dl):
            # print(f"Evaluating Batch: {itrid}")
            # filenames = list(data[1])
            points = data[0]
            points = points.transpose(2, 1)
            points = points.to(device)

            reconstructed_points, latent_vector = autoencoder_eval(points)  # perform training
            all_latent_vector = torch.cat((all_latent_vector, latent_vector), 0)
            # best_filenames will be N by 1024

        #############################
        samples = all_latent_vector.shape[0]
        print("all latent variable", all_latent_vector.shape, samples)

        avg_feature = torch.sum(all_latent_vector, axis=0, keepdims=True) / samples
        print("avg latent variable", avg_feature.shape)


        reconstructed_point = autoencoder_eval.decoder_forward(avg_feature)

        reconstructed_point = torch.reshape(reconstructed_point, (1, 3, config["num_points"]))
        reconstructed_point = reconstructed_point.squeeze().transpose(0, 1)
        reconstructed_point = reconstructed_point.cpu().detach().numpy()
        np.savetxt("avg_feature", reconstructed_point)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reconstructed_point)
        o3d.io.write_point_cloud(os.path.join(path, "avg_Recon.ply"), pcd)

# return best_latent_vector
