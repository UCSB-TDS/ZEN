import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from QuantizedInference import FullyConnected


np.set_printoptions(suppress=True)

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class BobNet(torch.nn.Module):
    def __init__(self):
        super(BobNet, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x


class BobNet_Quant(torch.nn.Module):
    def __init__(self):
        super(BobNet_Quant, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(128, 10, bias=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        # print("input: ", x)
        x = self.l1(x)
        if debug:
            try:
                print("1st layer output int repr: ", x.int_repr())
            except:
                pass
        x = self.act(x)
        if debug:
            try:
                print("act layer output int repr: ", x.int_repr())
            except:
                pass
        x = self.l2(x)
        if debug:
            try:
                print("2nd layer output int repr: ", x.int_repr())
            except:
                pass
        # print("l2: ", x)
        x = self.dequant(x)
        return x
    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 28*28))).float()
        x = self.quant(dummy_image)
        #print("input: ", x)
        l1_output = self.l1(x)
        #print("l1_output: ", l1_output.q_scale(), ', l1_output: ')
        # try:
        #     print("1st layer output: ", l1_output, ", int repr: ", l1_output.int_repr())
        # except:
        #     pass
        act_output = self.act(l1_output)
        l2_output = self.l2(act_output)
        # print("l2: ", x)
        output = self.dequant(l2_output)
        return x.q_scale(), x.q_zero_point(), l1_output.q_scale(), l1_output.q_zero_point(), act_output.q_scale(), act_output.q_zero_point(), l2_output.q_scale(), l2_output.q_zero_point()
    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()




model = BobNet_Quant()


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

# print('Model Weights: ', model.state_dict())

'''
weights.keys()
odict_keys(['l1.scale', 'l1.zero_point', 'l1._packed_params.dtype', 
'l1._packed_params._packed_params', 'l2.scale', 'l2.zero_point', 
'l2._packed_params.dtype', 'l2._packed_params._packed_params', 'quant.scale', 
'quant.zero_point'])
'''

weights = model.state_dict()
l1_weight = weights['l1._packed_params._packed_params'][0]
l2_weight = weights['l2._packed_params._packed_params'][0]
# print(l1_weight)
"""
Output should be something like:
tensor([[-0.0080,  0.0073, -0.0029,  ...,  0.0194, -0.0322, -0.0016],
        [ 0.0286,  0.0114,  0.0197,  ..., -0.0349, -0.0124,  0.0108],
        [-0.0207,  0.0084,  0.0354,  ..., -0.0189, -0.0228,  0.0300],
        ...,
        [-0.0351, -0.0221,  0.0212,  ..., -0.0191,  0.0218, -0.0309],
        [ 0.0225,  0.0338,  0.0176,  ...,  0.0173, -0.0124,  0.0026],
        [-0.0343,  0.0113,  0.0284,  ...,  0.0051,  0.0306, -0.0263]],
       size=(128, 784), dtype=torch.qint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0004, 0.0003, 0.0003,
        0.0004, 0.0003, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004,
        0.0003, 0.0003, 0.0004, 0.0003, 0.0003, 0.0004, 0.0004, 0.0003, 0.0003,
        0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003,
        0.0003, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003,
        0.0003, 0.0004, 0.0003, 0.0003, 0.0004, 0.0003, 0.0004, 0.0003, 0.0004,
        0.0004, 0.0003, 0.0004, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003, 0.0004, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0004, 0.0003,
        0.0004, 0.0004], dtype=torch.float64),
       zero_point=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]),
       axis=0)
"""

# print(l1_weight.int_repr())
"""
This is the int representation:
tensor([[ -25,   23,   -9,  ...,   61, -101,   -5],
        [  90,   36,   62,  ..., -110,  -39,   34],
        [ -69,   28,  118,  ...,  -63,  -76,  100],
        ...,
        [-116,  -73,   70,  ...,  -63,   72, -102],
        [  60,   90,   47,  ...,   46,  -33,    7],
        [ -94,   31,   78,  ...,   14,   84,  -72]], dtype=torch.int8)
"""





# l1_output.q_scale(), l1_output.q_zero_point(), act_output.q_scale(), act_output.q_zero_point(), l2_output.q_scale(), l2_output.q_zero_point() = model.dump_feat_param()

input_qscale, input_zero_point, l1_qscale, l1_zero_point, act_qscale, act_zero_point, l2_qscale, l2_zero_point = model.dump_feat_param()

assert l1_qscale == act_qscale, "Warning: l1_qscale != act_qscale. Voiate assumption in numpy inference."
assert l1_zero_point == act_zero_point, "Warning: l1_zero_point != act_zero_point. Voiate assumption in numpy inference."


def extract_Uint_Weight(weight):
    q = weight.int_repr().numpy().astype(np.int32)
    z = weight.q_per_channel_zero_points().numpy().astype(np.int32)
    s = weight.q_per_channel_scales().numpy()
    z -= q.min()
    q -= q.min()
    z = z.astype(np.uint32)
    q = q.astype(np.uint32)
    return q, z, s

def dump_txt(q, z, s, prefix):
    
    np.savetxt(prefix+"_q.txt", q.flatten(), fmt='%u', delimiter=',')
    print(z, s)
    f1 = open(prefix+"_z.txt", 'w+')
    if(str(z)[0] == '['):
        f1.write(str(z)[1:-1])
    else:
        f1.write(str(z))
    f1.close()
    f2 = open(prefix+"_s.txt", 'w+')
    if(str(s)[0]=='['):
        f2.write(str(s)[1:-1])
    else:
        f2.write(str(s))
    f2.close()

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
    z3 = l1_zero_point
    s3 = l1_qscale
    q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3) # Here, q3
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'pretrained_model/l1_weight')
        dump_txt(q3, z3, 0, 'pretrained_model/l1_output')
    if debug:
        print("1st layer output: ", q3, ", q3.dtype: ", q3.dtype)
    print("multiplier1 : {}".format(s1 * s2 / s3))
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
    q2 = act # Reuse q3 from output of 1st layer
    z2 = l1_zero_point
    s2 = l1_qscale
    # output feature. q3 needs to be computed. z3 and s3 is fixed.
    z3 = l2_zero_point
    s3 = l2_qscale
    q3 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'pretrained_model/l2_weight')
        dump_txt(q3, z3, 0, 'pretrained_model/l2_output')
    print("multiplier2 : {}".format(s1 * s2 / s3))

    return q3

# eval
Y_test_preds_out = forward(X_test.reshape((-1, 28*28)))
Y_test_preds = np.argmax(Y_test_preds_out, axis=1)

if debug:
    print("Y_test_preds: ", Y_test_preds)

print("Quantized Numpy Model Accuracy: ", (Y_test == Y_test_preds).mean())
if DUMP_FLAG == True:
    np.savetxt("pretrained_model/Y.txt", Y_test, fmt='%u', delimiter=',')

