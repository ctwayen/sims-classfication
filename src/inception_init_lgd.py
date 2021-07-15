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
print(torch.__version__)

training = os.listdir('../train/')
study_label = pd.read_csv('../archive/train_study_level.csv')
image_label = pd.read_csv('../archive/train_image_level.csv')
paths = []
labels = []
dct = pd.read_csv('train.csv', index_col=0)
dct['image'] = dct.image.apply(lambda x: x[:-4])
dct = dct.set_index('study').to_dict()['image']

for index, row in study_label.iterrows():
    name = dct[row['id'].replace('_study', '')] + '.png'
    if name in training:
        paths.append('../train/' + name)
        if row['Negative for Pneumonia'] == 1:
            labels.append(0)
        elif row['Typical Appearance'] == 1:
            labels.append(1)
        elif row['Indeterminate Appearance'] == 1:
            labels.append(2)
        elif row['Atypical Appearance'] == 1:
            labels.append(3)
    else:
        print(name)
        
class Net(nn.Module):
    def __init__(self, out_size, model):
        super(Net, self).__init__()
        if model == 'dense':
            self.model = torchvision.models.densenet121(pretrained=True, **{'drop_rate' : 0.3})
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                nn.Sigmoid()
            )
            
        elif model == 'res':
            self.model = torchvision.models.wide_resnet101_2(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                nn.Sigmoid()
            )
        elif model == 'inception':
            self.model = torchvision.models.inception_v3(pretrained=True, **{"aux_logits": False})
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, out_size)
            )
    def forward(self, x):
        x = self.model(x)
        return x
    

class classification(nn.Module):
    def __init__(self, paths, labels, size=(512,512), aug=False):
        self.paths = paths
        self.labels = labels
        self.aug = aug
        self.example = []
        self.size =size
    
    def __getitem__(self, idx):
            path = self.paths[idx]
            img = cv2.imread(path)
            R, G, B = cv2.split(img)
            output1_R = cv2.equalizeHist(R)
            output1_G = cv2.equalizeHist(G)
            output1_B = cv2.equalizeHist(B)

            img = cv2.merge((output1_R, output1_G, output1_B))
            img = (img - np.mean(img))/np.std(img)
            img = cv2.resize(img, self.size)
            x = torch.from_numpy(np.array(img)).view((3, self.size[0], self.size[1]))
            x = x.float()
            y = self.labels[idx]
            y = torch.tensor(y)
            return x, y
        
    def __len__(self):
        return len(self.paths)
    
    def get(self):
        return self.example
    
class Trainer():
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        
        self.epochs = opts['epochs']
        if opts['opt'] == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), opts['lr'], weight_decay=1e-5, amsgrad=True)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), opts['lr'], momentum=0.9)
        if opts['loss'] == 'focal':
            self.criterion = FocalLoss(**{"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'})
        else:
            self.criterion = torch.nn.CrossEntropyLoss()  
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=opts['batch_size'],
                                                    shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                    batch_size=opts['batch_size'],
                                                    shuffle=False)
        self.tb = SummaryWriter(log_dir='./runs/init_sgd_focal_2/')
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
                
                loss = self.criterion(outputs, labels)
                outputs = torch.nn.functional.softmax(outputs, 1)
                _, predicted = torch.max(outputs.data, 1)
                self.test_loss.append(loss.item())
                self.test_accuracy.append((predicted == labels).sum().item() / predicted.size(0))
            
            print('epoch: {}, train loss: {}, test loss: {}, test accuracy: {}'.format( 
                  epoch+1, np.mean(self.tr_loss), np.mean(self.test_loss), np.mean(self.test_accuracy)))
            self.tb.add_scalar("Val Acc", np.mean(self.test_accuracy), epoch)
            self.tb.add_scalar("Val Loss", np.mean(self.test_loss), epoch)
            if np.mean(self.test_loss) < self.best_loss:
                self.best_loss = np.mean(self.test_loss)
                torch.save(self.model.state_dict(), './model_weights/inception_after_init_lgd_focal_2.pt')

dataset = classification(paths, labels, size=(512, 512))
print(len(dataset))
train_set, val_set = torch.utils.data.random_split(dataset, [5200, 854])
model = Net(4, 'inception')
model.load_state_dict(torch.load('./model_weights/inception_init.pt'))
opts = {
    'lr': 3e-5,
    'epochs': 100,
    'batch_size': 32,
    'opt': 'lgd',
    'loss': 'focal'
}
train = Trainer(model, train_set, val_set, opts)
train.train()