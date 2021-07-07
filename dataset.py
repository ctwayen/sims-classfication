import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import volo
import torch
import torch.nn as nn
import torch.optim as opt
torch.set_printoptions(linewidth=120)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

class classification(nn.Module):
    def __init__(self, paths, labels, aug=False):
        self.paths = paths
        self.labels = labels
        self.aug = aug
    
    def __getitem__(self, idx):
            path = self.paths[idx]
            img = cv2.imread(path)[:, :, 0]
            img = cv2.resize(img, (448, 448))
            x = torch.from_numpy(np.array(img)).view((1, 448, 448))
            x = x.float()
            y = np.argmax(self.labels[idx])
            y = torch.tensor(y)
            return x, y
        
    def __len__(self):
        return len(self.paths)
    
def get_dataset():
    study_label = pd.read_csv('../archive/train_study_level.csv')
    paths = []
    ids = study_label.id
    labels = study_label.iloc[:, 1:].to_numpy()
    for i in range(len(ids)):
        paths.append('../archive/study/{x}.png'.format(x = ids[i]))
    paths = np.array(paths)
    dataset = classification(paths, labels)
    print(len(dataset))
    train_set, val_set = torch.utils.data.random_split(dataset, [5000, 1054])
    return train_set, val_set
    
    