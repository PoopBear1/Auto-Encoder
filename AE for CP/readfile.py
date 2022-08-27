from cProfile import label
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class CP_file():
    def __init__(self,root, npoints = 2500, class_choice = None, normalize=True):
        self.normalize = normalize
        self.npoints = npoints
        self.data_path = os.path.join(root,"shapenetcore_partanno_segmentation_benchmark_v0")
        self.logs = os.path.join(self.data_path,"synsetoffset2category.txt")
        self.cat = {}
        self.meta = {}

        with open(self.logs,'r') as f:
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


        sub_path =  os.path.join(self.data_path,str(self.cat[self.class_choice]))
        point_dir = os.path.join(sub_path,"points")
        seg_dir = os.path.join(sub_path,"points_label")
        
        fns = sorted(os.listdir(point_dir))
        
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0]) 
            self.meta[self.class_choice].append((os.path.join(point_dir, token + '.pts'), os.path.join(seg_dir, token + '.seg')))
        
        
    def split_to_set(self):
        data_set = []
        label_set = []
        for i in self.meta[self.class_choice]:
            data = np.loadtxt(i[0]).astype(np.float32)
            label = np.loadtxt(i[1]).astype(np.int64)
            if self.normalize:
                point_set = pc_normalize(data)

            choice = np.random.choice(len(label), self.npoints, replace=True)
            #resample
            point_set = point_set[choice, :]
            label = label[choice]    
            data_set.append(point_set)
            label_set.append(label)
            
        data_set = np.array(data_set)
        label_set = np.array(label_set)
        
        train_len = int(len(self.meta[self.class_choice]) * 0.7 ) 
        validation_len = int(len(self.meta[self.class_choice])* 0.2) 
        test_len = int(len(self.meta[self.class_choice])* 0.1) 

        train_data, train_label = point_set[:train_len], label_set[:train_len]

        validation_data, validation_label = point_set[train_len+1:validation_len], label_set[train_len+1:validation_len]

        test_data, test_label = point_set[validation_len+1:test_len], label_set[validation_len+1:test_len]

        return train_data, train_label,validation_data, validation_label,test_data, test_label

    def __len__(self):
        return len(self.meta[self.class_choice])

    def getCat(self):
        return self.cat

    def getChoice(self):
        return self.class_choice


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
        point = self.points[index][0]
        label = self.labels[index][1]
        if self.transform:
            point = self.transform(point)

        if self.target_transform:
            label = self.target_transform(label)

        return point, label



def load_data(config):
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir,"Data")
    Choice = "Chair"
    file = CP_file(data_dir,class_choice=Choice)
    train_data, train_label,validation_data, validation_label,test_data, test_label = file.split_to_set()

    train_set = CloudPointDataSet(train_data,train_label)
    validation_set = CloudPointDataSet(validation_data,validation_label)
    test_set = CloudPointDataSet(test_data,test_label)

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(validation_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
    )

config = {
        "lr": 1e-3,
        "num_epochs": 1,
        "batch_size": 64,
        "regular_constant": 4e-8,
    }


