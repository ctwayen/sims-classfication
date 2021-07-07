from dataset import get_dataset
from train import Trainer
import volo
import torch

train_set, val_set = get_dataset()
kwargs = { 
    'img_size': 448,
    'in_chans': 1,
    'num_classes': 4
}
model = volo.volo_d1(**kwargs)
opts = {
    'lr': 1e-2,
    'epochs': 100,
    'batch_size': 10
}
train = Trainer(model, train_set, val_set, opts)
train.train()