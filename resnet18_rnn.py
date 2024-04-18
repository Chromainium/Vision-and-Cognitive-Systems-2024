# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:40:05 2024

@author: chels
"""

from torchvision import models
import torch.nn as nn
import torch

class Resnet18Rnn(nn.Module):
    def __init__(self, num_classes, hidden_size, pretrained=True):
        super(Resnet18Rnn, self).__init__()
        
        # Load pre-trained ResNet-18 model
        encoderBaseModel = models.resnet18(pretrained=pretrained)
        input_size = encoderBaseModel.fc.in_features
        encoderBaseModel.fc = Identity()
        
        # # Remove the fully connected layer at the end
        # modules = list(encoderBaseModel.children())[:-1]
        # self.encoder = nn.Sequential(*modules)
        
        # # Freeze the parameters of the encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        self.baseModel = encoderBaseModel
        
        # Define the RNN decoder
        self.hidden_size = hidden_size
        #self.rnn = nn.RNN(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
        #self.rnn = nn.RNN(input_size=1000, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.rnn = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=1)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(hidden_size, num_classes*5)
        #self.fc2 = nn.Softmax(dim=2)
        #self.fc2 = softargmax()
    
    #############################################
    # def forward(self, x):
    #     # Pass input through the ResNet encoder
    #     with torch.no_grad():
    #         features = self.encoder(x)
    #     features = features.view(features.size(0), -1)
        
    #     # Initialize hidden state for RNN
    #     batch_size = x.size(0)
    #     hidden = self.init_hidden(batch_size)
        
    #     # Pass features through RNN decoder
    #     out, _ = self.rnn(features.unsqueeze(1), hidden)
        
    #     # Pass RNN output through linear layer for classification
    #     out = self.fc(out.squeeze(1))
        
    #     return out
    
    def forward(self, x):
        #b_z is batch size
        #ts is timesteps, timesteps is always 5
        b_z, ts, c, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:,ii])) #y is a Tensor with shape (2, 1000)
        #output, hn = self.rnn(y.unsqueeze(1)) #y.unsqueeze is a Tensor with shape (2, 1, 1000)
        output, (hn, cn) = self.rnn(y.unsqueeze(1)) #y.unsqueeze is a Tensor with shape (2, 1, 1000)
        # self.rnn(y.unsqueeze(1)) is a tuple (Tensor1, Tensor1)
        # output = Tensor1 (2,1,256) BUT (hn, cn) = Tensor2 is not a tuple (1,2,205)
        # line 58, in forward output, (hn, cn) = self.rnn(y.unsqueeze(1))
        # ValueError: not enough values to unpack (expected 2, got 1)
        for ii in range(1, ts):
            y = self.baseModel((x[:,ii]))
            #out, hn = self.rnn(y.unsqueeze(1), hn)
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
            #print("WHAT IS OUT? 1")
            #print(out.shape) #shape (5,1,256)
            #print(out)
            
        out = self.dropout(out[:,-1])
        #print("WHAT IS OUT? 2 (after dropout)")
        #print(out.shape) #shape ([5, 256])
        #print(out)
        
        ## NEW ADDITION
        #out = torch.reshape(out, (5,5,41))
        #out = torch.argmax(out, dim=1) #produces a tensor of shape torch.Size([5, 41])
        ##
        
        out = self.fc1(out)
        #print("WHAT IS OUT? 3 (after fc)")
        #print(out.shape, b_z) #shape torch.Size([5, 205])
        #print(out)
        
        out = torch.reshape(out, (b_z,5,41))
        #out = torch.argmax(out, dim=2) #produces a tensor of shape torch.Size([5, 5])
        out = softargmax(out) #should return tensor (5,5) i.e. (5,5,41) with max along 41
        
        return out
    
    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

def softargmax(x, beta=1e10):
    #x = torch.tensor(x)
    x_range = torch.arange(x.shape[-1], dtype=x.dtype)
    return torch.sum(torch.nn.functional.softmax(x*beta, dim=-1) * x_range, dim=-1)