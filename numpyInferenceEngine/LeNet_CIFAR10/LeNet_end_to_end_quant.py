import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import sys
sys.path.append('../operators')
from intOperators import FullyConnected, ConvolutionOperator, SumPoolOperator
from LeNet_models_Quant import *
import argparse

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(description='Flags for Floating-point LeNet.')
parser.add_argument('--model', type=str, required=True,
                   help='Version of LeNet in [LeNet_Small, LeNet_Medium, LeNet_Large]')

args = parser.parse_args()

debug = False

X_train = np.load('data/CIFAR_10/train_X.npy').astype('float64')
Y_train = np.load('data/CIFAR_10/train_Y.npy').astype('float64')
X_test = np.load('data/CIFAR_10/test_X.npy').astype('float64')
Y_test = np.load('data/CIFAR_10/test_Y.npy').astype('float64')

if args.model == 'LeNet_Small':
    model = LeNet_Small_Quant()
    weight_file = "LeNet_Small_weights.pkl"
    FC1_in_channel_num = 480
elif args.model == 'LeNet_Medium':
    model = LeNet_Medium_Quant()
    weight_file = "LeNet_Medium_weights.pkl"
    FC1_in_channel_num = 1024
elif args.model == 'LeNet_Large':
    model = LeNet_Large_Quant()
    weight_file = "LeNet_Large_weights.pkl"
    FC1_in_channel_num = 1024
else:
    assert 0 == 1, "Error in LeNet_end_to_end.py: args.model not in [LeNet_Small, LeNet_Medium, LeNet_Large]"

model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

# Evaluate Floating-Point Model.
Y_test_preds = torch.argmax(model(torch.tensor(X_test).float()), dim=1).numpy()
print("Floating-point PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

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

# Evaluate Quantized Model. Accuracy varies across trails.
Y_test_preds = torch.argmax(model(torch.tensor(X_test).float()), dim=1).numpy()
print("Quantized PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

weights = model.state_dict()

feature_quantize_parameters = model.dump_feat_param()

def extract_Weight(weight):
    q = weight.int_repr().numpy().astype(np.int32)
    z = weight.q_per_channel_zero_points().numpy().astype(np.int32)
    s = weight.q_per_channel_scales().numpy()
    return q, z, s


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
        dump_txt(x_quant_int_repr, x_quant_zero_point, x_quant_scale, 'LeNet_CIFAR_pretrained/X')    
    # 1st layer 
    q1, z1, s1 = extract_Uint_Weight(weights['conv1.weight'])

    q2, z2, s2 = x_quant_int_repr, x_quant_zero_point, x_quant_scale
    z3, s3 = feature_quantize_parameters['conv1_q_zero_point'], feature_quantize_parameters['conv1_q_scale']
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv1_weight')
    z3 = 128
    conv1 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    #print("conv1 {}".format(conv1))
    print("conv1.min(): ", conv1.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(conv1, z3, s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv1_output')
    act1 = np.maximum(conv1, z3)
    pool1, s3, z3 = SumPoolOperator(act1, 2), s3/4, z3*4

    # 2nd layer
    q1, z1, s1 = extract_Uint_Weight(weights['conv2.weight'])

    q2, z2, s2 = pool1, z3, s3
    z3, s3 = feature_quantize_parameters['conv2_q_zero_point'], feature_quantize_parameters['conv2_q_scale']
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv2_weight')
    z3 = 128
    conv2 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    #print("conv2 > 0? {}".format(conv2 > 0))
    #print("conv2 > 0? {}".format(conv2))
    print("conv2.min(): ", conv2.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(conv2, z3, s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv2_output')
    act2 = np.maximum(conv2, z3)
    pool2, s3, z3 = SumPoolOperator(act2, 2), s3/4, z3*4

    # 3rd layer
    q1, z1, s1 = extract_Uint_Weight(weights['conv3.weight'])
    
    q2, z2, s2 = pool2, z3, s3
    z3, s3 = feature_quantize_parameters['conv3_q_zero_point'], feature_quantize_parameters['conv3_q_scale']
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv3_weight')
    z3 = 128
    conv3 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    #print("conv3 > 0? {}".format(conv3))
    print("conv3.min(): ", conv3.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(conv3, z3, s3, 'LeNet_CIFAR_pretrained/'+args.model+'_conv3_output')
    act3 = np.maximum(conv3, z3)

    view_output = act3.reshape((-1,FC1_in_channel_num))

    # 4th layer
    q1, z1, s1 = extract_Uint_Weight(weights['linear1._packed_params._packed_params'][0])
    
    q2, z2, s2 = view_output, z3, s3
    z3, s3 = feature_quantize_parameters['linear1_q_zero_point'], feature_quantize_parameters['linear1_q_scale']
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_CIFAR_pretrained/'+args.model+'_linear1_weight')
    z3 = 128
    linear1 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    #print("linear1 > 0? {}".format(linear1))
    print("linear1.min(): ", linear1.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(linear1, z3, s3, 'LeNet_CIFAR_pretrained/'+args.model+'_linear1_output')
    act4 = np.maximum(linear1, z3)

    # 5th layer
    q1, z1, s1 = extract_Uint_Weight(weights['linear2._packed_params._packed_params'][0])
    
    q2, z2, s2 = act4, z3, s3
    z3, s3 = feature_quantize_parameters['linear2_q_zero_point'], feature_quantize_parameters['linear2_q_scale']
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'LeNet_CIFAR_pretrained/'+args.model+'_linear2_weight')
    z3 = 128
    linear2 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    #print("linear2 > 0? {}".format(linear2) )
    print("linear2.min(): ", linear2.min(), ", z3: ", z3)
    if DUMP_FLAG == True:
        dump_txt(linear2, z3, s3, 'LeNet_CIFAR_pretrained/'+args.model+'_linear2_output')

    return linear2

# eval
Y_test_preds_out = forward([X_test[0]])
Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
print("Quantized Numpy Model Accuracy: ", (Y_test == Y_test_preds).mean())




