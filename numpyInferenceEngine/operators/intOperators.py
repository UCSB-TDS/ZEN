"""
Implementation of quantized NN operators.

There are two versions. The 'cvpr' indicates the operators following the cvpr [1] paper. The 'zk' indicates the operators following our paper.

[1] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, HartwigAdam, and Dmitry Kalenichenko. 
    Quantization and training of neural networks for efficient integer-arithmetic-only inference. In CVPR, pages 2704–2713, 2018.

Written by Boyuan Feng
"""

import numpy as np

debug = False

version = 'zk' # could be 'zk' or 'cvpr'

def FullyConnected_cvpr(q_W, z_W, s_W, q_X, z_X, s_X, z_Y, s_Y):
    # @Param:
    #   q1, z1, s1: integer value, zero value, and scale factor for weight
    #   q2, z2, s2: integer value, zero value, and scale factor for input feature
    #   q3, z3, s3: integer value, zero value, and scale factor for output feature
    q_W = q_W.astype(np.int32)
    # q_X = q_X.astype(np.int32)
    M = s_W * s_X / s_Y
    M = (M * 2**22).astype(np.int32)
    q_Y = (M * np.matmul(q_X - z_X, (q_W - z_W).T))
    q_Y = q_Y // 2**22
    q_Y = z_Y + q_Y
    return q_Y#.astype(np.int8)

def FullyConnected_zk(q_W, z_W, s_W, q_X, z_X, s_X, z_Y, s_Y):
    # @Param:
    #   q_W, z_W, s_W: integer value, zero value, and scale factor for weight
    #   q_X, z_X, s_X: integer value, zero value, and scale factor for input feature
    #   q_Y, z_Y, s_Y: integer value, zero value, and scale factor for output feature
    k = 22

    q_W = q_W.astype(np.int64)
    q_X = q_X.astype(np.int64)
    M = s_W * s_X / s_Y
    M_prime = (M * 2**k).astype(np.int64)
    G1 = np.matmul(q_X, q_W.T)
    G2 = np.matmul(z_X*np.ones(q_X.shape, dtype=np.int64), q_W.T)
    G3 = np.matmul(q_X, z_W*np.ones(q_W.T.shape, dtype=np.int64))
    G4 = np.matmul(z_X*np.ones(q_X.shape, dtype=np.int64), z_W*np.ones(q_W.shape, dtype=np.int64).T)
    # M_prime2 = (2**k/M_prime).astype(np.int64)
    # print("q_X: ", q_X, ', q_W: ', q_W, ', z_X: ', z_X, ', z_W: ', z_W)
    # print('M_prime: ', M_prime, ', M_prime.shape: ', M_prime.shape, ", M_prime2: ", M_prime2)
    # print("G1.shape: ", G1.shape, ", G2.shape: ", G2.shape, ", G3.shape: ", G3.shape, ", M_prime2.shape: ", M_prime2.shape, ', z_Y: ', z_Y)
    # print("G1: ", G1, ", G2: ", G2, ", G3: ", G3, ", G4: ", G4)
    q_Y = M_prime*(G1+G4+(z_Y*2**k/M_prime).astype(np.int64)[None, :]-G2-G3)
    q_Y = q_Y // 2**k
    # print(q_Y.min(), q_Y.max())
    return q_Y.astype(np.uint8)

def FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3):
    # @Param:
    #   q1, z1, s1: integer value, zero value, and scale factor for weight
    #   q2, z2, s2: integer value, zero value, and scale factor for input feature
    #   q3, z3, s3: integer value, zero value, and scale factor for output feature
    if version == 'zk':
        return FullyConnected_zk(q1, z1, s1, q2, z2, s2, z3, s3)
    else:
        return FullyConnected_cvpr(q1, z1, s1, q2, z2, s2, z3, s3)

def ConvolutionOperator_cvpr(q_W, z_W, s_W, q_X, z_X, s_X, z_Y, s_Y):
    """
    Integer convolution operator.

    @input q1, z1, s1: integer value, zero value, and scale factor for weight
    @input q2, z2, s2: integer value, zero value, and scale factor for input feature
    @input: z3, s3: zero value, and scale factor for output feature
    @return q3: integer value for output feature
    """
    assert len(q_W.shape) == 4
    assert q_W.shape[2] == q_W.shape[3]
    K, C, kernel_size, _ = q_W.shape
    N, C, H, W = q_X.shape
    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.int32)
    q_W = q_W.astype(np.int32) - z_W
    for n in range(N):
        for x in range(H - kernel_size + 1):
            for y in range(W - kernel_size + 1):
                for k in range(K):
                    data = (q_X[n, :, x:x+kernel_size, y:y+kernel_size] - z_X).reshape(1, -1)
                    weight = q_W[k].reshape(1, -1)
                    M = s_W[k] * s_X / s_Y
                    M = (M * 2**22).astype(np.int32)
                    q_Y = (M *np.matmul(data, weight.T))
                    q_Y = q_Y // 2**22
                    output[n][k][x][y] = z_Y + q_Y
    return output

def ConvolutionOperator_zk(q_W, z_W, s_W, q_X, z_X, s_X, z_Y, s_Y):
    """
    Integer convolution operator.

    @input q1, z1, s1: integer value, zero value, and scale factor for weight
    @input q2, z2, s2: integer value, zero value, and scale factor for input feature
    @input: z3, s3: zero value, and scale factor for output feature
    @return q3: integer value for output feature
    """
    assert len(q_W.shape) == 4
    assert q_W.shape[2] == q_W.shape[3]

    exponent_k = 22

    K, C, kernel_size, _ = q_W.shape
    N, C, H, W = q_X.shape
    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.int32)
    for n in range(N):
        for x in range(H - kernel_size + 1):
            for y in range(W - kernel_size + 1):
                for k in range(K):
                    q_X_tmp = q_X[n, :, x:x+kernel_size, y:y+kernel_size].reshape(1,-1)
                    q_W_tmp = q_W[k].reshape(1, -1)
                    # print("s_W: ", s_W)
                    M = s_W[k] * s_X / s_Y
                    # print("M: ", M)
                    M_prime = int(M * 2**exponent_k)
                    G1 = np.matmul(q_X_tmp, q_W_tmp.T)
                    G2 = np.matmul(z_X*np.ones(q_X_tmp.shape, dtype=np.int64), q_W_tmp.T)
                    G3 = np.matmul(q_X_tmp, z_W*np.ones(q_W_tmp.T.shape, dtype=np.int64))
                    G4 = np.matmul(z_X*np.ones(q_X_tmp.shape, dtype=np.int64), z_W*np.ones(q_W_tmp.shape, dtype=np.int64).T)
                    # M_prime2 = int(2**exponent_k/M_prime)
                    # print("q_X: ", q_X, ', q_W: ', q_W, ', z_X: ', z_X, ', z_W: ', z_W)
                    # print('M_prime: ', M_prime, ', M_prime.shape: ', M_prime.shape, ", M_prime2: ", M_prime2)
                    # print("G1.shape: ", G1.shape, ", G2.shape: ", G2.shape, ", G3.shape: ", G3.shape, ", M_prime2.shape: ", M_prime2.shape, ', z_Y: ', z_Y)
                    # print("G1: ", G1, ", G2: ", G2, ", G3: ", G3, ", G4: ", G4)

                    q_Y = M_prime*(G1+G4+int(z_Y*2**exponent_k/M_prime)-G2-G3)
                    q_Y = q_Y // 2**exponent_k
                    output[n][k][x][y] = q_Y
    return output


def ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3):
    """
    Integer convolution operator.

    @input q1, z1, s1: integer value, zero value, and scale factor for weight
    @input q2, z2, s2: integer value, zero value, and scale factor for input feature
    @input: z3, s3: zero value, and scale factor for output feature
    @return q3: integer value for output feature
    """
    if version == 'zk':
        return ConvolutionOperator_zk(q1, z1, s1, q2, z2, s2, z3, s3)
    else:
        return ConvolutionOperator_cvpr(q1, z1, s1, q2, z2, s2, z3, s3)

def AvgPoolOperator(data, kernel_size):
    """
    Numpy version of the average pooling operation.
    
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @return output: a numpy array of shape [N, C, H//kernel_size, W//kernel_size]. 
    """
    N, C, H, W = data.shape
    output = np.ones((N, C, H//kernel_size, W//kernel_size)).astype(np.int32)
    for n in range(N):
        for c in range(C):
            for x in range(H//kernel_size):
                for y in range(W//kernel_size):
                    output[n, c, x, y] = np.mean(data[n, c, kernel_size*x:kernel_size*(x+1), kernel_size*y:kernel_size*(y+1)]).astype(np.int32)
    return output

def SumPoolOperator(data, kernel_size):
    """
    Numpy version of the sum pooling operation.
    
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @return output: a numpy array of shape [N, C, H//kernel_size, W//kernel_size]. 
    """
    N, C, H, W = data.shape
    output = np.ones((N, C, H//kernel_size, W//kernel_size)).astype(np.int32)
    for n in range(N):
        for c in range(C):
            for x in range(H//kernel_size):
                for y in range(W//kernel_size):
                    output[n, c, x, y] = np.sum(data[n, c, kernel_size*x:kernel_size*(x+1), kernel_size*y:kernel_size*(y+1)]).astype(np.int32)
    return output




