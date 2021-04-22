"""
Winograd Convolution
To be implemented: WinogradConv

By Boyuan Feng
"""
from floatOperators import conv


import numpy as np

def winograd_basic(data, weight):
    kernel_size, _ = weight.shape
    assert kernel_size == 5, "Error in WinogradConv: kernel size is not 5"

    H, W = data.shape
    assert H == 8, "Error in WinogradConv: input size is not 8x8"

    AT = np.array([
        [1,1,1,1,1,8,8,0],
        [0,1,-1,2,-2,4,-4,0],
        [0,1,1,4,4,2,2,0],
        [0, 1, -1, 8, -8, 1, -1, 1]
    ])

    G = np.array([
        [1, 0, 0, 0, 0],
        [-2/9, -2/9, -2/9, -2/9, -2/9],
        [-2/9, 2/9, -2/9, 2/9, -2/9],
        [1/90,1/45, 2/45, 4/45, 8/45],
        [1/90,-1/45, 2/45, -4/45, 8/45],
        [4/45, 2/45, 1/45, 1/90, 1/180],
        [4/45, -2/45, 1/45, -1/90, 1/180],
        [0, 0, 0, 0, 1]
    ])

    BT = np.array([
        [1, 0, -21/4, 0, 21/4, 0, -1, 0],
        [0, 1, 1, -17/4, -17/4, 1, 1, 0],
        [0, -1, 1, 17/4, -17/4, -1, 1, 0],
        [0, 1/2, 1/4, -5/2, -5/4, 2, 1, 0],
        [0, -1/2, 1/4, 5/2, -5/4, -2, 1, 0],
        [0, 2, 4, -5/2, -5, 1/2, 1, 0],
        [0, -2, 4, 5/2, -5, -1/2, 1, 0],
        [0, -1, 0, 21/4, 0, -21/4, 0, 1]
    ])

    G_part = np.matmul(np.matmul(G, weight), G.T)
    B_part = np.matmul(np.matmul(BT, data), BT.T)
    inner_part = G_part * B_part
    Y = np.matmul(np.matmul(AT, inner_part), AT.T)
    return Y

def winograd_multi_channel(data, weight):
    C, H, W = data.shape
    C, kernel_size, _ = weight.shape

    assert kernel_size == 5, "Error in winograd_multi_channel: kernel size is not 5"
    assert H == 8, "Error in winograd_multi_channel: input size is not 8x8"

    Y = np.zeros((4,4))
    for c in range(C):
        Y += winograd_basic(data[c], weight[c])
    
    return Y

def winograd_tile(data, weight):
    N, C, H, W = data.shape
    K, C, kernel_size, _ = weight.shape
    assert kernel_size == 5, "Error in winograd_tile: kernel size is not 5"
    assert H == 8, "Error in winograd_tile: input size is not 8x8"

    Y = np.zeros((N, K, 4, 4))
    for n in range(N):
        for k in range(K):
            Y[n][k] = winograd_multi_channel(data[n], weight[k])
    
    return Y

def winograd_conv(data, weight):
    N, C, H, W = data.shape
    K, C, kernel_size, _ = weight.shape
    assert kernel_size == 5, "Error in winograd_conv: kernel size is not 5"

    H4 = (H-4)//4
    W4 = (W-4)//4

    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.single)

    # First compute 8x8 tiles with winograd
    for i in range(H4):
        x = i*4
        for j in range(W4):
            y = j*4
            subdata = data[:, :, x: (x+8), y:(y+8)]
            sub_output = winograd_tile(subdata, weight)
            output[:,:,i*4:(i+1)*4, j*4:(j+1)*4] = winograd_tile(subdata, weight)
    
    # Then compute remaining part not fit into 8x8
    for n in range(N):
        for k in range(K):
            for x in range(H4 * 4, H - kernel_size + 1):
                for y in range(W4*4, W - kernel_size + 1):
                    output[n][k][x][y] = np.sum(data[n, :, x:x+kernel_size, y:y+kernel_size] * weight[k])

    return output


def wino_conv(data, weight, bias=None):
    """
    Wrapper function for conv operator.
    May use it to switch between GeneralConv and WinogradConv.

    These two implementations have the same results but different number of computations.
    @input data: an numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the width and height of the image/feature maps.
    @input weight: an numpy array of shape [K, C, kernel_size, kernel_size]. K is the number of kernels, C is the number of channels, kernel_size is the kernel size.
    @return output: a numpy array of shape [N, K, H-kernel_size+1, W - kernel_size + 1]. 
    """
    return winograd_conv(data, weight)



def test1():
    data = np.random.rand(8,8)
    weight = np.random.rand(5,5)
    standard_conv_output = conv(data[None, None, :, :], weight[None, None, :, :])
    winograd_conv_output = winograd_basic(data, weight)
    print("test1:")
    print("standard_conv_output: ", standard_conv_output)
    print("winograd_conv_output: ", winograd_conv_output)
    print("\n\n")


def test2():
    data = np.random.rand(6, 8,8)
    weight = np.random.rand(6, 5,5)
    standard_conv_output = conv(data[None, :, :, :], weight[None, :, :, :])
    winograd_conv_output = winograd_multi_channel(data, weight)
    print("test2:")
    print("standard_conv_output: ", standard_conv_output)
    print("winograd_conv_output: ", winograd_conv_output)
    print("\n\n")

def test3():
    data = np.random.rand(2, 6, 8,8)
    weight = np.random.rand(3, 6, 5,5)
    standard_conv_output = conv(data, weight)
    winograd_conv_output = winograd_tile(data, weight)
    print("test3:")
    print("standard_conv_output: ", standard_conv_output)
    print("winograd_conv_output: ", winograd_conv_output)
    print("\n\n")

def test4():
    data = np.random.rand(2, 6, 24,24)
    weight = np.random.rand(3, 6, 5,5)
    standard_conv_output = conv(data, weight)
    winograd_conv_output = winograd_conv(data, weight)
    print("test4:")
    print("standard_conv_output: ", standard_conv_output)
    print("winograd_conv_output: ", winograd_conv_output)
    print("\n\n")

if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()








