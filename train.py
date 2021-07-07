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

class Trainer():
    def __init__(self,model,train_set,test_set,opts):
        self.model = model  # neural net
        # device agnostic code snippet
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
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
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train() #put model in training mode
            self.tr_loss = []
            for i, (data,labels) in tqdm(enumerate(self.train_loader),
                                                   total = len(self.train_loader)):
                data, labels = data.to(self.device),labels.to(self.device)
                self.optimizer.zero_grad()  
                outputs = self.model(data)   
                loss = self.criterion(outputs[0], labels) 
                loss.backward()                        
                self.optimizer.step()                  
                self.tr_loss.append(loss.item())     
            self.tb.add_scalar("Train Loss", np.mean(self.tr_loss), epoch)
            self.test(epoch) # run through the validation set
            torch.save(self.model, './model_weights/d1/{x}.pt'.format(x=str(epoch)))
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