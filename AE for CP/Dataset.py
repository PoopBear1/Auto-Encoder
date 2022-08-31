import torch
from torch.utils.data import Dataset
import numpy as np
import os


def pc_normalize(pc):
    # print(pc.shape)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PointCloudDataSet(Dataset):
    def __init__(self, data, path, transform=True):
        self.path = path
        self.dataset = data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform:
            data = pc_normalize(data)

        points = torch.from_numpy(data).float()

        return points, self.path[index]
