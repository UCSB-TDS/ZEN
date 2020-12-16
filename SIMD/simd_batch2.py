"""
SIMD Example
    Given four integer vectors A, B, C, and D, where each vector has the length of 1000. 
    We aim to compute two dot products A \cdot B and C \cdot D.

    A naive implementation [in xxx function] requires 2000 multiplications.
    The SIMD implementation [in xxx function] requires 1000 multiplications.

by Boyuan Feng
"""

import numpy as np


def naiveDotProduct(A, B, C, D):
    """
    @input A, B, C, D: a vector of integers
    @return res1, res2: two integer values. res1 = A \cdot B, res2 = C \cdot D.
    """
    # res1 = A \cdot B
    res1 = 0
    for i in range(len(A)):
        res1 += A[i] * B[i]

    # res2 = C \cdot D
    res2 = 0
    for i in range(len(C)):
        res2 += C[i] * D[i]

    return res1, res2

def encode(A, B):
    """
    @input A: a vector of integers
    @input B: a vector of integers
    @return C: a vector of integers that encodes A and B
    """
    C = [0]*len(A)
    for i in range(len(A)):
        C[i] = A[i] + B[i] * 2**26
    return C

def decode(C):
    """
    @input C: an integer value
    @return A, B: the decoded integer value
    """
    A = C % 2**26
    B = C // 2**52

    return A, B


def SIMDDotProduct(A, B, C, D):
    """
    @input A, B, C, D: a vector of integers
    @return res1, res2: two integer values. res1 = A \cdot B, res2 = C \cdot D.
    """
    # Encode A and C into X
    X = encode(A, C)
    # Encode B and D into Y
    Y = encode(B, D)

    res = 0
    for i in range(len(X)):
        res += X[i] * Y[i]

    res1, res2 = decode(res)

    return res1, res2


A = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
B = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
C = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
D = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()


res1_naive, res2_naive = naiveDotProduct(A, B, C, D)
print("res1_naive: ", res1_naive, ", res2_naive: ", res2_naive)

res1_SIMD, res2_SIMD = SIMDDotProduct(A, B, C, D)
print("res1_SIMD: ", res1_SIMD, ", res2_SIMD: ", res2_SIMD)


assert res1_naive == res1_SIMD, "Error: res1_naive != res1_SIMD"
assert res2_naive == res2_SIMD, "Error: res2_naive != res2_SIMD"











