import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import *
# import show3d_balls


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class CP_file():
    def __init__(self, root, npoints=2400, class_choice=None, normalize=True):
        self.normalize = normalize
        self.npoints = npoints
        self.data_path = os.path.join(root, "shapenetcore_partanno_segmentation_benchmark_v0")
        self.logs = os.path.join(self.data_path, "synsetoffset2category.txt")
        self.cat = {}
        self.meta = {}

        with open(self.logs, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if class_choice:
            if class_choice in self.cat:
                self.class_choice = class_choice
                self.meta[class_choice] = []
            else:
                self.class_choice = None
                print("No chosen class in Database")
                exit(-1)
        # else:
        # self.class_choice = list(self.cat.keys())

        sub_path = os.path.join(self.data_path, str(self.cat[self.class_choice]))
        point_dir = os.path.join(sub_path, "points")
        seg_dir = os.path.join(sub_path, "points_label")

        fns = sorted(os.listdir(point_dir))

        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            self.meta[self.class_choice].append(
                (os.path.join(point_dir, token + '.pts'), os.path.join(seg_dir, token + '.seg')))

    def split_to_set(self):
        data_set = []
        label_set = []
        for i in self.meta[self.class_choice]:
            data = np.loadtxt(i[0]).astype(np.float32)
            label = np.loadtxt(i[1]).astype(np.int64)
            if self.normalize:
                point_set = pc_normalize(data)

            # resample
            choice = np.random.choice(len(label), self.npoints, replace=True)
            point_set = point_set[choice, :].reshape(-1, self.npoints)
            label = label[choice]
            data_set.append(point_set)
            label_set.append(label)

        data_set = np.array(data_set)
        label_set = np.array(label_set)

        train_len = int(len(self.meta[self.class_choice]) * 0.7)
        validation_len = int(len(self.meta[self.class_choice]) * 0.2)
        test_len = int(len(self.meta[self.class_choice]) * 0.1)

        # print(train_len,validation_len,test_len)
        train_data, train_label = data_set[:train_len], label_set[:train_len]

        validation_data, validation_label = data_set[train_len + 1:train_len + validation_len], label_set[
                                                                                                train_len + 1:train_len + validation_len]

        test_data, test_label = data_set[
                                train_len + validation_len + 1:train_len + validation_len + test_len], label_set[
                                                                                                       train_len + validation_len + 1:train_len + validation_len + test_len]

        return train_data, train_label, validation_data, validation_label, test_data, test_label

    def __len__(self):
        return len(self.meta[self.class_choice])

    def getCat(self):
        return self.cat

    def getChoice(self):
        return self.class_choice

    def getTest(self):
        length = len(self.meta[self.class_choice])
        idx = np.random.choice(length)
        testfile = np.loadtxt(self.meta[self.class_choice][idx][0])
        choice = np.random.choice(len(testfile[0]), self.npoints, replace=True)
        point_set = testfile[choice, :]

        return point_set

    def __getitem__(self, index):
        point = self.meta[self.class_choice][index]
        label = self.meta[self.class_choice][index]
        return point, label


class CloudPointDataSet(Dataset):
    def __init__(self, points, labels, transform=None, target_transform=None):
        self.points = points
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        points = self.points[index]
        labels = self.labels[index]

        if self.transform:
            points = self.transform(points)

        if self.target_transform:
            labels = self.target_transform(labels)

        return points, labels


def load_data(file, config):
    path = os.path.join(os.getcwd(), "exp_data")

    if os.path.exists(path):
        train_data = np.load(os.path.join(path, "train_data.npy"))
        train_label = np.load(os.path.join(path, "train_label.npy"))

        validation_data = np.load(os.path.join(path, "validation_data.npy"))
        validation_label = np.load(os.path.join(path, "validation_label.npy"))

        test_data = np.load(os.path.join(path, "test_data.npy"))
        test_label = np.load(os.path.join(path, "test_label.npy"))
    else:
        os.mkdir("exp_data")
        train_data, train_label, validation_data, validation_label, test_data, test_label = file.split_to_set()

        np.save(os.path.join(path, "train_data.npy"), train_data)
        np.save(os.path.join(path, "train_label.npy"), train_label)

        np.save(os.path.join(path, "validation_data.npy"), validation_data)
        np.save(os.path.join(path, "validation_label.npy"), validation_label)

        np.save(os.path.join(path, "test_data.npy"), test_data)
        np.save(os.path.join(path, "test_label.npy"), test_label)

    train_set = CloudPointDataSet(train_data, train_label)
    validation_set = CloudPointDataSet(validation_data, validation_label)
    test_set = CloudPointDataSet(test_data, test_label)

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
    )


def trained_model_visualize(model, cloud_point, config):
    cloud_point = torch.tensor(np.expand_dims(cloud_point, 0)).to(device=config["device"], dtype=torch.float)
    # print(cloud_point.shape)
    # print(cloud_point)
    model.eval()
    reconstructed_point = model(cloud_point).cpu().detach().numpy()
    print("model output: ", reconstructed_point.shape)
    reconstructed_point = np.squeeze(reconstructed_point).reshape(-1, 3)
    print("the final output: ", reconstructed_point.shape)
    np.savetxt("Recon_CP", reconstructed_point)
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
        "num_epochs": 90,
        "num_points": 2500,
        "batch_size": 32,
        "regular_constant": 4e-8,
        "device": device,
        "loss_function": loss_function,
    }

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    Choice = "Chair"
    num_points = config["num_points"]
    file = CP_file(data_dir, npoints=num_points, class_choice=Choice)
    cloud_point_testfile = file.getTest()
    print("this is the original CP ", cloud_point_testfile.shape)
    np.savetxt("Ori_sampledCP", cloud_point_testfile)
    # show3d_balls.showpoints(cloud_point_testfile, ballradius=8)

    if os.path.exists("ckpt.pth"):
        checkpoint = torch.load("ckpt.pth", map_location=device)
        model = AutoEncoder(num_points).to(device)
        model.load_state_dict(checkpoint)
    else:
        train_loader, valid_loader, test_loader = load_data(file, config)
        model = train(train_loader, valid_loader, config)
        # show numerical loss result
        test(test_loader, model, config)

    # check model performance
    trained_model_visualize(model, cloud_point_testfile.reshape(3, -1), config)


main()
