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

torch.set_printoptions(precision=8)
np.set_printoptions(precision=10)

parser = argparse.ArgumentParser(description='Flags for Floating-point LeNet.')
parser.add_argument('--model', type=str, required=True,
                   help='Version of LeNet in [LeNet_Small, LeNet_Medium, LeNet_Large]')

args = parser.parse_args()

# load the mnist dataset

def fetch(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if not os.path.isfile(fp):
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    else:
        with open(fp, "rb") as f:
            dat = f.read()    
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

if args.model == 'LeNet_Small':
    model = LeNet_Small()
    weight_file = "LeNet_Small_weights.pkl"
    FC1_in_channel_num = 120
elif args.model == 'LeNet_Medium':
    model = LeNet_Medium()
    weight_file = "LeNet_Medium_weights.pkl"
    FC1_in_channel_num = 512
elif args.model == 'LeNet_Large':
    model = LeNet_Large()
    weight_file = "LeNet_Large_weights.pkl"
    FC1_in_channel_num = 1024
else:
    assert 0 == 1, "Error in LeNet_end_to_end.py: args.model not in [LeNet_Small, LeNet_Medium, LeNet_Large]"

def train():
    BS = 128
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001) # adam optimizer
    losses, accuracies = [], []
    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((BS, 1, 28, 28))).float()
        Y = torch.tensor(Y_train[samp]).long()
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

#evaluation
Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 1, 28, 28))).float()), dim=1).numpy()
print((Y_test_preds == Y_test).mean())

# copy weights from pytorch
conv1_weight = model.conv1.weight.detach().numpy().astype(np.single)
conv2_weight = model.conv2.weight.detach().numpy().astype(np.single)
conv3_weight = model.conv3.weight.detach().numpy().astype(np.single)
l1_weight = model.linear1.weight.detach().numpy().astype(np.single).T
l2_weight = model.linear2.weight.detach().numpy().astype(np.single).T

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

# eval
Y_test_preds_out = forward(X_test.reshape((-1, 1, 28, 28)))
Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
print((Y_test == Y_test_preds).mean())

