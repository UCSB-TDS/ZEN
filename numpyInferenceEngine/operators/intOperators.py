import numpy as np

debug = False

def FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3):
    # @Param:
    #   q1, z1, s1: integer value, zero value, and scale factor for weight
    #   q2, z2, s2: integer value, zero value, and scale factor for input feature
    #   q2, z2, s2: integer value, zero value, and scale factor for output feature
    q1 = q1.astype(np.int32)
    z1 = z1.astype(np.int32)
    M = s1 * s2 / s3
    M = (M * 2**22).astype(np.int32)
    q3 = (M * np.matmul(q2 - z2, (q1 - z1[:, None]).T))
    q3 = q3 // 2**22
    q3 = z3 + q3
    return q3.astype(np.int32)


def ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3):
    """
    Integer convolution operator.

    @input q1, z1, s1: integer value, zero value, and scale factor for weight
    @input q2, z2, s2: integer value, zero value, and scale factor for input feature
    @input: z3, s3: zero value, and scale factor for output feature
    @return q3: integer value for output feature
    """
    assert len(q1.shape) == 4
    assert q1.shape[2] == q1.shape[3]
    K, C, kernel_size, _ = q1.shape
    N, C, H, W = q2.shape
    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.int32)
    q1 = q1.astype(np.int32) - z1[:, None, None, None]
    for n in range(N):
        for x in range(H - kernel_size + 1):
            for y in range(W - kernel_size + 1):
                for k in range(K):
                    data = (q2[n, :, x:x+kernel_size, y:y+kernel_size] - z2).reshape(1, -1)
                    weight = q1[k].reshape(1, -1)
                    M = s1[k] * s2 / s3
                    M = (M * 2**22).astype(np.int32)
                
                    #print("data {}".format(data))
                    #print("weight {}".format(weight))
                    q3 = (M *np.matmul(data, weight.T))
                    #print("q3 {}".format(q3))
                    q3 = q3 // 2**22
                    #print("q3 {}".format(q3 + z3))
                    output[n][k][x][y] = z3 + q3
                    #print("q3 {}".format(q3 + z3))

    return output


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




