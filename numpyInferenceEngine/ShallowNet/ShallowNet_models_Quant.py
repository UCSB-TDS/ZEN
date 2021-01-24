"""
Implementation for numpy-version Quantized ShallowNet 

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

class ShallowNet_Quant(torch.nn.Module):
    def __init__(self):
        super(ShallowNet_Quant, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.dequant(x)
        return x
    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 28*28))).float()
        x = self.quant(dummy_image)
        l1_output = self.l1(x)
        act_output = self.act(l1_output)
        l2_output = self.l2(act_output)
        # print("l2: ", x)
        output = self.dequant(l2_output)
        return x.q_scale(), x.q_zero_point(), l1_output.q_scale(), l1_output.q_zero_point(), act_output.q_scale(), act_output.q_zero_point(), l2_output.q_scale(), l2_output.q_zero_point()
    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()