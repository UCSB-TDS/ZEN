import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from ShallowNet_models_Quant import *
import sys
sys.path.append('../operators')
from intOperators import FullyConnected, ConvolutionOperator
from utility import *

np.set_printoptions(threshold=sys.maxsize)

debug = False

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

model = ShallowNet_Quant()

# Accuracy varies when training the model for several times.
# Thus we stick to a fixed and pretrained weights.
model.load_state_dict(torch.load('weights.pkl'))

# Evaluate Floating-Point Model. Should be 93.14%
Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
print("Floating-point PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# Calibrate
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(128))
    X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
    Y = torch.tensor(Y_train[samp]).long()
    out = model(X)

print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

# Evaluate Quantized Model. Accuracy varies across trails. Have seen [0.9258, 0.931, 0.9314, 0.9317, 0.9321]
Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
print("Quantized PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

weights = model.state_dict()
l1_weight = weights['l1._packed_params._packed_params'][0]
l2_weight = weights['l2._packed_params._packed_params'][0]

input_qscale, input_zero_point, l1_qscale, l1_zero_point, act_qscale, act_zero_point, l2_qscale, l2_zero_point = model.dump_feat_param()

assert l1_qscale == act_qscale, "Warning: l1_qscale != act_qscale. Voiate assumption in numpy inference."
assert l1_zero_point == act_zero_point, "Warning: l1_zero_point != act_zero_point. Voiate assumption in numpy inference."

DUMP_FLAG = True

def forward(x):
    # First quant on input x.
    x_quant_int_repr, x_quant_scale, x_quant_zero_point = model.quant_input(x)
    if DUMP_FLAG == True:
        dump_txt(x_quant_int_repr, x_quant_zero_point, x_quant_scale, 'pretrained_model/X')

    # 1st layer 
    # weight
    q1, z1, s1 = extract_Uint_Weight(l1_weight)
    # input feature. 
    # The input feature is indeed per_tensor_affine, instead of per_channel_affine.
    q2 = x_quant_int_repr # suppose that x is integer
    z2 = x_quant_zero_point
    s2 = x_quant_scale
    # output feature. q3 needs to be computed. z3 and s3 is fixed.
    z3 = 128
    s3 = l1_qscale
    q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3) # Here, q3
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'pretrained_model/l1_weight')
        dump_txt(q3, z3, 0, 'pretrained_model/l1_output')
    if debug:
        print("1st layer output: ", q3, ", q3.dtype: ", q3.dtype)
    # print("multiplier1 : {}".format(s1 * s2 / s3))
    # Activation Function
    act = np.maximum(q3, z3)

    if debug:
        print("act layer output: ", x, ", x.dtype: ", x.dtype)

    # 2nd layer.
    # weight
    q1, z1, s1 = extract_Uint_Weight(l2_weight)

    # input feature. 
    # The input feature is indeed per_tensor_affine, instead of per_channel_affine.
    # Still use Per_channel_affine to use the same FullyConnected API.
    q2, z2, s2 = act, z3, s3
    # output feature. q3 needs to be computed. z3 and s3 is fixed.
    z3 = 128
    s3 = l2_qscale
    q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'pretrained_model/l2_weight')
        dump_txt(q3, z3, 0, 'pretrained_model/l2_output')
    # print("multiplier2 : {}".format(s1 * s2 / s3))

    return q3

# eval
Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
Y_test_preds = np.argmax(Y_test_preds_out, axis=1)

if debug:
    print("Y_test_preds: ", Y_test_preds)

print("Quantized Numpy Model Accuracy: ", (Y_test == Y_test_preds).mean())
if DUMP_FLAG == True:
    np.savetxt("pretrained_model/Y.txt", Y_test, fmt='%u', delimiter=',')

