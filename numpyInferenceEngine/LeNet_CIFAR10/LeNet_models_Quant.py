"""
Implementation for numpy-version Quantized LeNet, 
    including LeNet_Small_Quant, LeNet_Medium_Quant, LeNet_Large_Quant.


By Boyuan Feng
"""

import numpy as np
import sys
import os
sys.path.append('../operators')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class LeNet_Small_Quant(nn.Module):
    def __init__(self):
        super(LeNet_Small_Quant, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=480, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        x = self.dequant(x)
        return x
    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 3, 32, 32))).float()
        x = self.quant(dummy_image)
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        conv2 = self.conv2(pool1)
        act2 = self.act2(conv2)
        pool2 = self.pool2(act2)
        conv3 = self.conv3(pool2)
        act3 = self.act3(conv3)
        view_output = act3.reshape(act3.size(0), -1)
        linear1 = self.linear1(view_output)
        act4 = self.act4(linear1)
        linear2 = self.linear2(act4)
        output = self.dequant(linear2)
        feature_quantize_parameters = {'x_q_scale': x.q_scale(), 'x_q_zero_point': x.q_zero_point(),
        "conv1_q_scale": conv1.q_scale(), 'conv1_q_zero_point': conv1.q_zero_point(),
        'act1_q_scale': act1.q_scale(), "act1_q_zero_point":act1.q_zero_point(), 
        'pool1_q_scale': pool1.q_scale(), 'pool1_q_zero_point': pool1.q_zero_point(), 
        'conv2_q_scale': conv2.q_scale(), "conv2_q_zero_point":conv2.q_zero_point(), 
        "act2_q_scale":act2.q_scale(), "act2_q_zero_point":act2.q_zero_point(), 
        'pool2_q_scale':pool2.q_scale(), "pool2_q_zero_point":pool2.q_zero_point(),
        "conv3_q_scale":conv3.q_scale(), "conv3_q_zero_point":conv3.q_zero_point(), 
        "act3_q_scale": act3.q_scale(), "act3_q_zero_point": act3.q_zero_point(), 
        "linear1_q_scale": linear1.q_scale(), "linear1_q_zero_point": linear1.q_zero_point(), 
        "act4_q_scale":act4.q_scale(), "act4_q_zero_point":act4.q_zero_point(), 
        "linear2_q_scale": linear2.q_scale(), "linear2_q_zero_point": linear2.q_zero_point()
        }
        return feature_quantize_parameters
    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()

class LeNet_Medium_Quant(nn.Module):
    def __init__(self):
        super(LeNet_Medium_Quant, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=1024, out_features=128)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        x = self.dequant(x)
        return x
    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 3, 32, 32))).float()
        x = self.quant(dummy_image)
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        conv2 = self.conv2(pool1)
        act2 = self.act2(conv2)
        pool2 = self.pool2(act2)
        conv3 = self.conv3(pool2)
        act3 = self.act3(conv3)
        view_output = act3.reshape(act3.size(0), -1)
        linear1 = self.linear1(view_output)
        act4 = self.act4(linear1)
        linear2 = self.linear2(act4)
        output = self.dequant(linear2)
        feature_quantize_parameters = {'x_q_scale': x.q_scale(), 'x_q_zero_point': x.q_zero_point(),
        "conv1_q_scale": conv1.q_scale(), 'conv1_q_zero_point': conv1.q_zero_point(),
        'act1_q_scale': act1.q_scale(), "act1_q_zero_point":act1.q_zero_point(), 
        'pool1_q_scale': pool1.q_scale(), 'pool1_q_zero_point': pool1.q_zero_point(), 
        'conv2_q_scale': conv2.q_scale(), "conv2_q_zero_point":conv2.q_zero_point(), 
        "act2_q_scale":act2.q_scale(), "act2_q_zero_point":act2.q_zero_point(), 
        'pool2_q_scale':pool2.q_scale(), "pool2_q_zero_point":pool2.q_zero_point(),
        "conv3_q_scale":conv3.q_scale(), "conv3_q_zero_point":conv3.q_zero_point(), 
        "act3_q_scale": act3.q_scale(), "act3_q_zero_point": act3.q_zero_point(), 
        "linear1_q_scale": linear1.q_scale(), "linear1_q_zero_point": linear1.q_zero_point(), 
        "act4_q_scale":act4.q_scale(), "act4_q_zero_point":act4.q_zero_point(), 
        "linear2_q_scale": linear2.q_scale(), "linear2_q_zero_point": linear2.q_zero_point()
        }
        return feature_quantize_parameters
    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()

class LeNet_Large_Quant(nn.Module):
    def __init__(self):
        super(LeNet_Large_Quant, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=1024, out_features=512)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
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
        x = self.dequant(x)
        return x
    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 1, 28, 28))).float()
        x = self.quant(dummy_image)
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        conv2 = self.conv2(pool1)
        act2 = self.act2(conv2)
        pool2 = self.pool2(act2)
        conv3 = self.conv3(pool2)
        act3 = self.act3(conv3)
        view_output = act3.view(act3.size(0), -1)
        linear1 = self.linear1(view_output)
        act4 = self.act4(linear1)
        linear2 = self.linear2(act4)
        output = self.dequant(linear2)
        feature_quantize_parameters = {'x_q_scale': x.q_scale(), 'x_q_zero_point': x.q_zero_point(),
        "conv1_q_scale": conv1.q_scale(), 'conv1_q_zero_point': conv1.q_zero_point(),
        'act1_q_scale': act1.q_scale(), "act1_q_zero_point":act1.q_zero_point(), 
        'pool1_q_scale': pool1.q_scale(), 'pool1_q_zero_point': pool1.q_zero_point(), 
        'conv2_q_scale': conv2.q_scale(), "conv2_q_zero_point":conv2.q_zero_point(), 
        "act2_q_scale":act2.q_scale(), "act2_q_zero_point":act2.q_zero_point(), 
        'pool2_q_scale':pool2.q_scale(), "pool2_q_zero_point":pool2.q_zero_point(),
        "conv3_q_scale":conv3.q_scale(), "conv3_q_zero_point":conv3.q_zero_point(), 
        "act3_q_scale": act3.q_scale(), "act3_q_zero_point": act3.q_zero_point(), 
        "linear1_q_scale": linear1.q_scale(), "linear1_q_zero_point": linear1.q_zero_point(), 
        "act4_q_scale":act4.q_scale(), "act4_q_zero_point":act4.q_zero_point(), 
        "linear2_q_scale": linear2.q_scale(), "linear2_q_zero_point": linear2.q_zero_point()
        }
        return feature_quantize_parameters
    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()




