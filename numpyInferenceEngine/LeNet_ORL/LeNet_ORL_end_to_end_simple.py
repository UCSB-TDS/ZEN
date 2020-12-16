"""
Implementation for numpy-version LeNet.
Trained on MNIST and achieve 98.36% accuracy with floating point operations.
The numpy version and the PyTorch version provides the same accuracy.

By Boyuan Feng
"""

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../operators')
from floatOperators import conv, avg_pool
from winograd import wino_conv
import torch
import torch.nn as nn
from LeNet_models import *
import argparse
import random

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms

torch.set_printoptions(precision=8)
np.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Flags for Floating-point LeNet.')
parser.add_argument('--model', type=str, required=True,
                   help='Version of LeNet in [LeNet_Small, LeNet_Medium, LeNet_Large]')

args = parser.parse_args()

# load the mnist dataset

X_train = np.load('orl_face_dataset/train_X.npy').astype('float64')
Y_train = np.load('orl_face_dataset/train_Y.npy').astype('float64')
X_test = np.load('orl_face_dataset/test_X.npy').astype('float64')
Y_test = np.load('orl_face_dataset/test_Y.npy').astype('float64')


if args.model == 'LeNet_Small':
    model = LeNet_Small().cuda()
    weight_file = "LeNet_Small_weights.pkl"
    FC1_in_channel_num = 4800
    FC2_out_channel_num = 40
elif args.model == 'LeNet_Medium':
    model = LeNet_Medium().cuda()
    weight_file = "LeNet_Medium_weights.pkl"
    FC1_in_channel_num = 10240
    FC2_out_channel_num = 40
elif args.model == 'LeNet_Large':
    model = LeNet_Large().cuda()
    weight_file = "LeNet_Large_weights.pkl"
    FC1_in_channel_num = 20480
    FC2_out_channel_num = 40
else:
    assert 0 == 1, "Error in LeNet_end_to_end.py: args.model not in [LeNet_Small, LeNet_Medium, LeNet_Large]"

def train():
    BS = 1024
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001) # adam optimizer
    losses, accuracies = [], []

    for i in (t := trange(200 * (10000//BS))):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp]).float().cuda()
        Y = torch.tensor(Y_train[samp]).long().cuda()
        optim.zero_grad()
        out = model(X)
        # compute accuracy
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        t.set_description("loss %.2f accuracy %.2f" % (loss.item(), accuracy.item()))
    torch.save(model.state_dict(), weight_file)

if os.path.exists(weight_file):
    model.load_state_dict(torch.load(weight_file))
    model.eval()
else:
    train()

# copy weights from pytorch
conv1_weight = model.conv1.weight.cpu().detach().numpy().astype(np.single)
conv2_weight = model.conv2.weight.cpu().detach().numpy().astype(np.single)
conv3_weight = model.conv3.weight.cpu().detach().numpy().astype(np.single)
l1_weight = model.linear1.weight.cpu().detach().numpy().astype(np.single).T
l2_weight = model.linear2.weight.cpu().detach().numpy().astype(np.single).T

# numpy forward pass
def forward(x):
    x = x.astype(np.single)
    x = conv(x, conv1_weight)
    x = np.maximum(x, 0)
    x = avg_pool(x, 2)
    x = conv(x, conv2_weight)
    x = np.maximum(x, 0)
    x = avg_pool(x, 2)
    x = conv(x, conv3_weight)
    x = np.maximum(x, 0)
    x = x.reshape((-1,FC1_in_channel_num))
    x = x.dot(l1_weight)
    x = np.maximum(x, 0)
    x = x.dot(l2_weight)
    return x

#evaluation

def cosine_similarity(x, y):
    return x.dot(y)/(x.dot(x)**0.5 * y.dot(y)**0.5)

y_pred = []
y_gt = []

for i in range(1000):
    if i % 10 == 0:
        print(i)
    idx0 = random.randint(0,39)
    x0 = X_test[idx0].reshape((1, 1, 56, 46))
    while True:
        idx1 = random.randint(0,39)
        x1 = X_test[idx1].reshape((1, 1, 56, 46))
        label = (Y_test[idx0] == Y_test[idx1])
        if label == i%2:
            break
    # print("Y_test[idx0]: ", Y_test[idx0], ", Y_test[idx1]: ", Y_test[idx1])
    
    output1 = forward(x0)
    output2 = forward(x1)
    cos = cosine_similarity(output1.reshape((FC2_out_channel_num,)), output2.reshape((FC2_out_channel_num,)))
    # print("output1: ", output1, ", output2: ", output2)
    # print("cos: ", cos)
    if cos > 0.5:
        y_pred.append(1) # The same person
    else:
        y_pred.append(0) # Different person
    y_gt.append(int(label))

print(y_pred)
print(y_gt)
print((np.array(y_pred) == np.array(y_gt)))
print((np.array(y_pred) == np.array(y_gt)).mean())


# Y_test_preds = torch.argmax(model(torch.tensor(X_test).cuda().float()), dim=1).cpu().numpy()
# print((Y_test_preds == Y_test).mean())

