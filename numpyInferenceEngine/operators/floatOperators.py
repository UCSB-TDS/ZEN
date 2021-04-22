"""
Several implementations for the convolution operation.
Include: NaiveConv.

To be implemented: WinogradConv

By Boyuan Feng
"""

import numpy as np

def NaiveConv(data, weight):
    """
    Numpy version of the convolution operation.
    
    @input data: an numpy array of shape [N,N]
    @input weight: an numpy array of shape [kernel_size, kernel_size]
    @return 
    """
    assert data.shape[0] == data.shape[1]
    assert weight.shape[0] == weight.shape[1]
    N = data.shape[0]
    kernel_size = weight.shape[0]
    output = np.ones((N - kernel_size + 1, N - kernel_size + 1))
    for x in range(N - kernel_size + 1):
        for y in range(N - kernel_size + 1):
            output[x][y] = 0
            for i in range(kernel_size):
                for j in range(kernel_size):
                    output[x][y] += data[x+i][y+j] * weight[i][j]

    return output

if False:
    data = np.ones((28,28))
    weight = np.ones((3,3))
    output = NaiveConv(data, weight)


def GeneralConv(data, weight, bias=None):
    """
    Numpy version of the convolution operation.
    
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @input weight: an numpy array of shape [K, C, kernel_size, kernel_size]. K is the number of kernels, C is the number of channels, kernel_size is the kernel size.
    @return output: a numpy array of shape [N, K, H-kernel_size+1, W - kernel_size + 1]. 
    """
    assert weight.shape[2] == weight.shape[3]
    N, C, H, W = data.shape
    K, C, kernel_size, _ = weight.shape
    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.single)
    for n in range(N):
        for x in range(H - kernel_size + 1):
            for y in range(W - kernel_size + 1):
                for k in range(K):
                    output[n][k][x][y] = np.sum(data[n, :, x:x+kernel_size, y:y+kernel_size] * weight[k])
    if bias is not None:
        for k in range(K):
            output[:, k, :, :] += bias[k]

    return output



def conv(data, weight, bias=None):
    """
    Wrapper function for conv operator.
    May use it to switch between GeneralConv and WinogradConv.

    These two implementations have the same results but different number of computations.
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @input weight: an numpy array of shape [K, C, kernel_size, kernel_size]. K is the number of kernels, C is the number of channels, kernel_size is the kernel size.
    @return output: a numpy array of shape [N, K, H-kernel_size+1, W - kernel_size + 1]. 
    """
    return GeneralConv(data, weight, bias)


def avg_pool(data, kernel_size):
    """
    Numpy version of the average pooling operation.
    
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @return output: a numpy array of shape [N, C, H//kernel_size, W//kernel_size]. 
    """
    N, C, H, W = data.shape
    output = np.ones((N, C, H//kernel_size, W//kernel_size))
    for n in range(N):
        for c in range(C):
            for x in range(H//kernel_size):
                for y in range(W//kernel_size):
                    output[n, c, x, y] = np.mean(data[n, c, kernel_size*x:kernel_size*(x+1), kernel_size*y:kernel_size*(y+1)])
    return output


if __name__ == "__main__":
    data = np.ones((5, 128, 28,28))
    weight = np.ones((128, 128, 3,3))
    output = conv(data, weight)

    print("output: ", output)
