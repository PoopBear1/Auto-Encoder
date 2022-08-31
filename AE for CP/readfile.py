from train import train, test
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from Dataset import PointCloudDataSet, pc_normalize
import open3d as o3d
# from modelV2 import PointNetAE
from model import AutoEncoder
# import show3d_balls


class CP_file():
    def __init__(self, root, npoints, class_choice=None, normalize=True):
        self.normalize = normalize
        self.data_path = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0")
        self.logs = os.path.join(self.data_path, "synsetoffset2category.txt")
        self.all_class_in_path = {}
        self.meta = {}
        self.npoints = npoints
        with open(self.logs, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.all_class_in_path[ls[0]] = ls[1]

        if class_choice:
            if class_choice in self.all_class_in_path:
                self.class_choice = class_choice
                self.meta[class_choice] = []
            else:
                self.class_choice = None
                print("No chosen class in Database")
                exit(-1)
        # else:
        # self.class_choice = list(self.all_class_in_path.keys())

        sub_path = os.path.join(self.data_path, str(self.all_class_in_path[self.class_choice]))
        point_dir = os.path.join(sub_path, "points")
        seg_dir = os.path.join(sub_path, "points_label")

        fns = sorted(os.listdir(point_dir))

        # min_point_amount = 100000
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            temp_data = os.path.join(point_dir, token + '.pts')
            temp_label = os.path.join(seg_dir, token + '.seg')
            # temp_render =

            self.meta[self.class_choice].append((temp_data, temp_label))

        #     count = len(open(temp_data, 'r').readlines())
        #     if count < min_point_amount:
        #         min_point_amount = count
        #
        # self.npoints = min_point_amount

    def split_to_set(self):
        data_set = []
        path_set = []
        for i in self.meta[self.class_choice]:
            data = i[0]
            label = i[1]

            data = np.loadtxt(data).astype(np.float64)
            label = np.loadtxt(label).astype(np.int64)

            choice = np.random.choice(len(label), self.npoints, replace=True)
            data = data[choice, :]

            data_set.append(data)
            path_set.append(i[0])

        train_len = int(len(self.meta[self.class_choice]) * 0.9)

        train_data, train_path = data_set[:train_len], path_set[:train_len]

        test_data, test_path = data_set[train_len:], path_set[train_len:]

        return train_data, train_path, test_data, test_path

    def __len__(self):
        return len(self.meta[self.class_choice])

    def getCat(self):
        return self.cat

    def getChoice(self):
        return self.class_choice

    def getTest(self):
        length = len(self.meta[self.class_choice])
        idx = np.random.choice(length)
        testfile, label = self.meta[self.class_choice][idx]
        testfile, label = np.loadtxt(testfile), np.loadtxt(label)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(testfile)
        o3d.io.write_point_cloud("Original.ply", pcd)

        testfile = np.asarray(pcd.points)

        choice = np.random.choice(testfile.shape[0], self.npoints, replace=True)
        point_set = testfile[choice, :]
        label = label[choice]

        point_set = pc_normalize(point_set)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set)
        o3d.io.write_point_cloud("sampledPC.ply", pcd)

        return point_set

    def __getitem__(self, index):
        point = self.meta[self.class_choice][index]
        label = self.meta[self.class_choice][index]
        return point, label


def load_data(file, config):
    path = os.path.join(os.getcwd(), "exp_data")

    if os.path.exists(path):
        path = os.path.join(path, config["class"])
        if os.path.exists(path):
            train_data = np.load(os.path.join(path, "train_data.npy"))
            train_path = np.load(os.path.join(path, "train_path.npy"))
            test_path = np.load(os.path.join(path, "test_path.npy"))
            test_data = np.load(os.path.join(path, "test_data.npy"))
        else:
            os.mkdir(path)
            train_data, train_path, test_data, test_path = file.split_to_set()

            np.save(os.path.join(path, "train_data.npy"), train_data)
            np.save(os.path.join(path, "train_path.npy"), train_path)

            np.save(os.path.join(path, "test_data.npy"), test_data)
            np.save(os.path.join(path, "test_path.npy"), test_path)
    else:
        os.mkdir(path)
        path = os.path.join(path, config["class"])
        os.mkdir(path)
        train_data, train_path, test_data, test_path = file.split_to_set()
        np.save(os.path.join(path, "train_data.npy"), train_data)
        np.save(os.path.join(path, "test_data.npy"), test_data)
        np.save(os.path.join(path, "train_path.npy"), train_path)
        np.save(os.path.join(path, "test_data.npy"), test_data)

    Total_set = PointCloudDataSet(train_data, train_path, config["num_points"])

    train_set, validation_set = random_split(Total_set, [int(len(Total_set) * 0.8),
                                                         int(len(Total_set) - int(len(Total_set) * 0.8))])
    test_set = PointCloudDataSet(test_data, test_path, config["num_points"])

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
    )


def trained_model_visualize(autoencoder, cloud_point, config):
    cloud_point = torch.tensor(np.expand_dims(cloud_point, 0)).to(device=config["device"], dtype=torch.float)
    cloud_point = cloud_point.transpose(1, 2)
    # print(cloud_point.shape)
    # autoencoder = autoencoder.eval()
    autoencoder.eval()
    reconstructed_point, _ = autoencoder(cloud_point)
    print("model output: ", reconstructed_point.shape)

    ### Wrong way to visualize
    # reconstructed_point = np.squeeze(reconstructed_point.reshape(-1, 3).cpu().detach().numpy())

    #### Correct way to do so:
    reconstructed_point = reconstructed_point.squeeze().transpose(0, 1)
    reconstructed_point = reconstructed_point.cpu().detach().numpy()
    print("model output: ", reconstructed_point.shape)

    np.savetxt("Recon_CP", reconstructed_point)
    print("the final output: ", reconstructed_point.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reconstructed_point)
    o3d.io.write_point_cloud("Recon.ply", pcd)
    # show3d_balls.showpoints(reconstructed_point, ballradius=8)


def main():
    if torch.cuda.is_available():
        from chamfer_distance.chamfer_distance_gpu import \
            ChamferDistance  # https://github.com/chrdiller/pyTorchChamferDistance

        device = torch.device("cuda")

    else:
        from chamfer_distance.chamfer_distance_cpu import \
            ChamferDistance  # https://github.com/chrdiller/pyTorchChamferDistance

        device = torch.device("cpu")

    loss_function = ChamferDistance()

    config = {
        "lr": 1e-3,
        "num_epochs": 110,
        "num_points": 2500,
        "batch_size": 32,
        "regular_constant": 1e-6,
        "device": device,
        "loss_function": loss_function,
        "class": "Chair",
    }

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    file = CP_file(data_dir, class_choice=config["class"], npoints=config["num_points"])
    # config["num_points"] = file.npoints

    cloud_point_testfile = file.getTest()

    train_loader, valid_loader, test_loader = load_data(file, config)
    model_path = os.path.join(os.path.join(os.getcwd(), "exp_data"), config["class"], "ckpt.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        autoencoder = AutoEncoder(config["num_points"]).to(device)
        autoencoder.load_state_dict(checkpoint)

    else:
        autoencoder = train(train_loader, valid_loader, config)
    # show numerical loss result
    test(test_loader, autoencoder, config)

    # train_loader, valid_loader, test_loader = load_data(file, config)
    # autoencoder = train(train_loader, valid_loader, config)

    # check model performance
    # trained_model_visualize(autoencoder, cloud_point_testfile, config)
    ############################

    # temp = PointCloudDataSet(np.expand_dims(cloud_point_testfile, 0), 1)
    # train_loader = DataLoader(temp, batch_size=1)
    #
    # valid_loader = DataLoader(temp, batch_size=1)
    # autoencoder = train(train_loader, valid_loader, config)
    trained_model_visualize(autoencoder, cloud_point_testfile, config)
    ############################


main()
