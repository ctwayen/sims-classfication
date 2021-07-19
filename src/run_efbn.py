import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as opt
torch.set_printoptions(linewidth=120)
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import torch.nn.functional as F
from focal import FocalLoss
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T
from torch.optim.lr_scheduler import ExponentialLR

print(torch.__version__)
torch.manual_seed(0)

study_label = pd.read_csv('../archive/train_study_level.csv')
paths = []
ids = study_label.id
labels = study_label.iloc[:, 1:].to_numpy()
for i in range(len(ids)):
    paths.append('../archive/study/{x}.png'.format(x = ids[i]))
paths = np.array(paths)
labels = np.argmax(labels, axis=1)

np.random.seed(seed=2)
train_idx = np.random.choice(np.arange(6054), size=5000, replace=False)
train_path = np.array(paths)[train_idx]
test_path = np.array(paths)[[x for x in np.arange(6054) if x not in train_idx]]

train_label = np.array(labels)[train_idx]
test_label = np.array(labels)[[x for x in np.arange(6054) if x not in train_idx]]

class efbn(nn.Module):
    def __init__(self, out_size):
        super(efbn, self).__init__()
        self.efbn = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs = self.efbn._fc.in_features
        self.efbn._fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )
        self.efbn._conv_stem = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=(2, 2), bias=False),
            nn.ZeroPad2d(padding=(0, 1, 0, 1))
        )

    def forward(self, x):
        x = self.efbn(x)
        return x
    
model = efbn(4)
trans = T.RandomApply(torch.nn.ModuleList([
                T.RandomAffine(
                    degrees = (10, 30),
                    translate = (0.2, 0.2),
                ),
                T.RandomRotation(degrees=(0, 50)),
                T.RandomHorizontalFlip(p=0.5)  
            ]),p = 0.8)
class classification(torch.utils.data.Dataset):
    def __init__(self, paths, labels, size=(512,512), train=False, aug=True):
        self.paths = paths
        self.labels = labels
        self.example = []
        self.size =size
        self.train = train
        self.aug = aug
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)[:, :, 0]
        img = cv2.equalizeHist(img)
        #img = (img - np.mean(img))/np.std(img)
        img = img/np.mean(img)
        img = cv2.resize(img, self.size)

        x = torch.from_numpy(np.array(img)).view((1, self.size[0], self.size[1]))
        x = x.float()
        y = self.labels[idx]
        y = torch.tensor(y)
        if self.train:
            if self.aug:
                x = trans(x)
            return x, y
        else:
            return x, y
        
    def __len__(self):
        return len(self.paths)
    
def roc_auc_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:

    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    return roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')

def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

class Trainer():
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
    
        self.epochs = opts['epochs']
        if opts['opt'] == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), opts['lr'], weight_decay=1e-5, amsgrad=True)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), opts['lr'], momentum=0.9)
        if opts['loss'] == 'focal':
            self.criterion = FocalLoss(**{"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'})
            self.mix = False
        elif opts['loss'] == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss() 
            self.mix = False
        else:
            self.criterion1 = torch.nn.CrossEntropyLoss() 
            self.criterion2 = FocalLoss(**{"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'})
            self.mix = True
            
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=False)
        self.tb = SummaryWriter(log_dir='./runs/efbn3')
        self.best_loss = 1e10
        self.tr_loss = []
        self.sche = ExponentialLR(self.optimizer, gamma=0.1)
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train() #put model in training mode
            self.tr_loss = []
            for i, (data,labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                data, labels = data.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()  
                outputs = self.model(data)
                if self.mix == False:
                    loss = self.criterion(outputs, labels) 
                    loss.backward()                        
                    self.optimizer.step()                  
                    self.tr_loss.append(loss.item())   
                else:
                    loss = self.criterion1(outputs, labels) + self.criterion2(outputs, labels)
                    loss.backward()                        
                    self.optimizer.step()                  
                    self.tr_loss.append(loss.item()) 
            self.tb.add_scalar("Train Loss", np.mean(self.tr_loss), epoch)
            self.test(epoch) # run through the validation set
            self.sche.step()
        self.tb.close()
            
    def test(self,epoch):    
        self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
        self.test_loss = []
        self.test_accuracy = []
        self.predicted = []
        self.true = []
        for i, (data, labels) in enumerate(self.test_loader):

            data, labels = data.to(self.device),labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(data)
            if self.mix == False:
                loss = self.criterion(outputs, labels)
            else:
                loss = self.criterion1(outputs, labels) + self.criterion2(outputs, labels)
            outputs = torch.nn.functional.softmax(outputs, 1)
            _, predicted = torch.max(outputs.data, 1)
            self.test_loss.append(loss.item())
            self.test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            self.predicted.append(predicted)
            self.true.append(labels)
        
        test_auc = roc_auc_compute_fn(one_hot(torch.cat(self.predicted, dim=0), 4), one_hot(torch.cat(self.true, dim=0), 4))

        print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}, test auc: {}'.format( 
              epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy), test_auc))
        self.tb.add_scalar("Val Acc", np.mean(self.test_accuracy), epoch)
        self.tb.add_scalar("Val Loss", np.mean(self.test_loss), epoch)
        self.tb.add_scalar("Val auc", test_auc, epoch)
        if np.mean(self.test_loss) < self.best_loss:
            self.best_loss = np.mean(self.test_loss)
            torch.save(self.model.state_dict(), './model_weights/efbnbest.pt')
            
train_set = classification(train_path, train_label, size=(300, 300), train=True, aug=False)
val_set = classification(test_path, train_label, size=(300, 300), train=False, aug=False)

opts = {
    'lr': 1e-4,
    'epochs': 30,
    'batch_size': 16,
    'opt': 'adam',
    'loss': 'ce'
}
train = Trainer(model, train_set, val_set, opts)
train.train()