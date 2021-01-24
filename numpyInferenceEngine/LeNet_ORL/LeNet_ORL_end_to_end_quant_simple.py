import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from LeNet_models_Quant import *
import argparse
import random

import sys
sys.path.append('../operators')
from intOperators import FullyConnected, ConvolutionOperator, AvgPoolOperator
from utility import *

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Flags for Floating-point LeNet.')
parser.add_argument('--model', type=str, required=True,
                   help='Version of LeNet in [LeNet_Small, LeNet_Medium, LeNet_Large]')

args = parser.parse_args()

debug = False

X_train = np.load('orl_face_dataset/train_X.npy').astype('float64')
Y_train = np.load('orl_face_dataset/train_Y.npy').astype('float64')
X_test = np.load('orl_face_dataset/test_X.npy').astype('float64')
Y_test = np.load('orl_face_dataset/test_Y.npy').astype('float64')


if args.model == 'LeNet_Small':
    model = LeNet_Small_Quant()
    weight_file = "LeNet_Small_weights.pkl"
    FC1_in_channel_num = 4800
    FC2_out_channel_num = 40
elif args.model == 'LeNet_Medium':
    model = LeNet_Medium_Quant()
    weight_file = "LeNet_Medium_weights.pkl"
    FC1_in_channel_num = 10240
    FC2_out_channel_num = 40
elif args.model == 'LeNet_Large':
    model = LeNet_Large_Quant()
    weight_file = "LeNet_Large_weights.pkl"
    FC1_in_channel_num = 20480
    FC2_out_channel_num = 40
else:
    assert 0 == 1, "Error in LeNet_end_to_end.py: args.model not in [LeNet_Small, LeNet_Medium, LeNet_Large]"

model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# Calibrate
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(128))
    X = torch.tensor(X_train[samp]).float()
    Y = torch.tensor(Y_train[samp]).long()
    out = model(X)

print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

weights = model.state_dict()

feature_quantize_parameters = model.dump_feat_param()

DUMP_FLAG = True

def forward(x):
    # First quant on input x.
    x_quant_int_repr, x_quant_scale, x_quant_zero_point = model.quant_input(x)
    if DUMP_FLAG == True:
        dump_txt(x_quant_int_repr, x_quant_zero_point, x_quant_scale, 'LeNet_ORL_pretrained/X')

    # 1st layer 
    q1, z1, s1 = extract_Uint_Weight(weights['conv1.weight'])
    
    q2, z2, s2 = x_quant_int_repr, x_quant_zero_point, x_quant_scale
    z3, s3 = feature_quantize_parameters['conv1_q_zero_point'], feature_quantize_parameters['conv1_q_scale']
    z3 = 128
    conv1 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    print("conv1.min(): ", conv1.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_ORL_pretrained/'+args.model+'_conv1_weight')
        dump_txt(conv1, z3, s3, 'LeNet_ORL_pretrained/'+args.model+'_conv1_output')
    act1 = np.maximum(conv1, z3)
    # pool1, s3, z3 = SumPoolOperator(act1, 2), s3/4, z3*4
    pool1 = AvgPoolOperator(act1, 2)

    # 2nd layer
    q1, z1, s1 = extract_Uint_Weight(weights['conv2.weight'])
    q2, z2, s2 = pool1, z3, s3
    z3, s3 = feature_quantize_parameters['conv2_q_zero_point'], feature_quantize_parameters['conv2_q_scale']
    z3 = 128
    conv2 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    print("conv2.min(): ", conv2.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_ORL_pretrained/'+args.model+'_conv2_weight')
        dump_txt(conv2, z3, s3, 'LeNet_ORL_pretrained/'+args.model+'_conv2_output')
    act2 = np.maximum(conv2, z3)
    # pool2, s3, z3 = SumPoolOperator(act2, 2), s3/4, z3*4
    pool2 = AvgPoolOperator(act2, 2)

    # 3rd layer
    q1, z1, s1 = extract_Uint_Weight(weights['conv3.weight'])
    q2, z2, s2 = pool2, z3, s3
    z3, s3 = feature_quantize_parameters['conv3_q_zero_point'], feature_quantize_parameters['conv3_q_scale']
    z3 = 128    
    conv3 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    print("conv3.min(): ", conv3.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_ORL_pretrained/'+args.model+'_conv3_weight')
        dump_txt(conv3, z3, s3, 'LeNet_ORL_pretrained/'+args.model+'_conv3_output')
    act3 = np.maximum(conv3, z3)

    view_output = act3.reshape((-1,FC1_in_channel_num))

    # 4th layer
    q1, z1, s1 = extract_Uint_Weight(weights['linear1._packed_params._packed_params'][0])
    q2, z2, s2 = view_output, z3, s3
    z3, s3 = feature_quantize_parameters['linear1_q_zero_point'], feature_quantize_parameters['linear1_q_scale']
    z3 = 128 
    linear1 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    print("linear1.min(): ", linear1.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_ORL_pretrained/'+args.model+'_linear1_weight')
        dump_txt(linear1, z3, s3, 'LeNet_ORL_pretrained/'+args.model+'_linear1_output')
    act4 = np.maximum(linear1, z3)

    # 5th layer
    q1, z1, s1 = extract_Uint_Weight(weights['linear2._packed_params._packed_params'][0])
    q2, z2, s2 = act4, z3, s3
    z3, s3 = feature_quantize_parameters['linear2_q_zero_point'], feature_quantize_parameters['linear2_q_scale']
    z3 = 128
    linear2 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    print("linear2.min(): ", linear2.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_ORL_pretrained/'+args.model+'_linear2_weight')
        dump_txt(linear2, z3, s3, 'LeNet_ORL_pretrained/'+args.model+'_linear2_output')

    return linear2

#evaluation

def cosine_similarity(x, y):
    return x.dot(y)/(x.dot(x)**0.5 * y.dot(y)**0.5)

y_pred = []
y_gt = []

for i in range(100):
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
    z3 = feature_quantize_parameters['linear2_q_zero_point']

    output1 = forward(x0) - z3
    output2 = forward(x1) - z3
    cos = cosine_similarity(output1.reshape((FC2_out_channel_num,)), output2.reshape((FC2_out_channel_num,)))
    # print("output1: ", output1, ", output2: ", output2, ', z3: ', z3)
    # print("cos: ", cos)
    if cos > 0.5:
        y_pred.append(1) # The same person
    else:
        y_pred.append(0) # Different person
    y_gt.append(int(label))

print("Quantized Model Accuracy: ", (np.array(y_pred) == np.array(y_gt)).mean())


