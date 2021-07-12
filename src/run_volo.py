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

study_label = pd.read_csv('../archive/train_study_level.csv')
paths = []
ids = study_label.id
labels = study_label.iloc[:, 1:].to_numpy()
for i in range(len(ids)):
    paths.append('../archive/study/{x}.png'.format(x = ids[i]))
paths = np.array(paths)

class adding_bn(nn.Module):
    def __init__(self, n):
        super(adding_bn, self).__init__()
        self.bn = nn.BatchNorm2d(n)
        kwargs = { 
            'img_size': 224,
            'in_chans': 1,
            'num_classes': 4,
            'return_dense': False,
            'mix_token': False,
            'drop_rate' : 0.3, 
            'attn_drop_rate' : 0.3, 
            'drop_path_rate' : 0.3,
        }
        self.model = volo.volo_d1(**kwargs)
        
    def forward(self, x):
        x = self.bn(x)
        return self.model(x)

model = adding_bn(1)
class classification(nn.Module):
    def __init__(self, paths, labels, aug=False):
        self.paths = paths
        self.labels = labels
        self.aug = aug
        self.example = []
    
    def __getitem__(self, idx):
            path = self.paths[idx]
            img = cv2.imread(path)[:, :, 0]
            
            img = cv2.resize(img, (224, 224))
            img = (img - np.mean(img))/np.std(img)
            x = torch.from_numpy(np.array(img)).view((1, 224, 224))
            x = x.float()
            y = np.argmax(self.labels[idx])
            y = torch.tensor(y)
            return x, y
        
    def __len__(self):
        return len(self.paths)
    
    def get(self):
        return self.example
    
dataset = classification(paths, labels)
print(len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [5000, 1054])

class Trainer():
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        
        self.epochs = opts['epochs']
        self.optimizer = torch.optim.Adam(model.parameters(), opts['lr'], weight_decay=1e-5, amsgrad=True) # optimizer method for gradient descent
        self.criterion = torch.nn.CrossEntropyLoss()                      # loss function
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                        batch_size=opts['batch_size'],
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                       batch_size=opts['batch_size'],
                                                       shuffle=False)
        self.tb = SummaryWriter()
        self.best_loss = 1e10
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train() #put model in training mode
            self.tr_loss = []
            for i, (data,labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                data, labels = data.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()  
                outputs = self.model(data)   
                loss = self.criterion(outputs, labels) 
                loss.backward()                        
                self.optimizer.step()                  
                self.tr_loss.append(loss.item())     
            self.tb.add_scalar("Train Loss", np.mean(self.tr_loss), epoch)
            self.test(epoch) # run through the validation set
        self.tb.close()
            
    def test(self,epoch):
            
            self.model.eval()    # puts model in eval mode - not necessary for this demo but good to know
            self.test_loss = []
            self.test_accuracy = []
            
            for i, (data, labels) in enumerate(self.test_loader):
                
                data, labels = data.to(self.device),labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(data)
                
                _, predicted = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)
                self.test_loss.append(loss.item())
                
                self.test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            
            print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy)))
            self.tb.add_scalar("Val Acc", np.mean(self.test_accuracy), epoch)
            self.tb.add_scalar("Val Loss", np.mean(self.test_loss), epoch)
            if np.mean(self.test_loss) < self.best_loss:
                self.best_loss = np.mean(self.test_loss)
                torch.save(self.model.state_dict(), './model_weights/best1.pt')
opts = {
    'lr': 1e-5,
    'epochs': 100,
    'batch_size': 32
}
train = Trainer(model, train_set, val_set, opts)
train.train()