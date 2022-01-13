from torch.optim import optimizer
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from ConvLayers import ConvLayers
from torchvision import models

class Spinal(nn.Module):

    def __init__(self, num_classes=78, layer_width=128, half_width=128):
        super(Spinal, self).__init__()

        self.half_width = half_width
        self.layer_width = layer_width
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_width, self.layer_width),
            nn.BatchNorm1d(self.layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(self.layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(self.layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_width+self.layer_width, self.layer_width),
            nn.BatchNorm1d(self.layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.layer_width*4, num_classes),)
        
    
    def forward(self, x):
        
        x1 = self.fc_spinal_layer1(x[:, 0:self.half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,self.half_width:2*self.half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:self.half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,self.half_width:2*self.half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)
        return x