"""
Implementation for numpy-version LeNet, including LeNet_Small, LeNet_Medium, LeNet_Large.


By Boyuan Feng
"""

import numpy as np
import sys
import os
sys.path.append('../operators')
import torch
import torch.nn as nn

class LeNet_Small(nn.Module):
    def __init__(self):
        super(LeNet_Small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=4800, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=40)
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        return x

class LeNet_Medium(nn.Module):
    def __init__(self):
        super(LeNet_Medium, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=10240, out_features=128)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=40)
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        return x

class LeNet_Large(nn.Module):
    def __init__(self):
        super(LeNet_Large, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=20480, out_features=256)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=256, out_features=40)
    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        return x




