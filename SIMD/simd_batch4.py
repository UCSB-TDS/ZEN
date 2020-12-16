"""
SIMD Example
    Given  integer vectors A, B, C, D, E, F, G, and H, where each vector has the length of 1000. 
    We aim to compute four dot products A \cdot B, C \cdot D, E \cdot F, G \cdot H.

    A naive implementation [in xxx function] requires 4000 multiplications.
    The SIMD implementation [in xxx function] requires 1000 multiplications.

by Boyuan Feng
"""

import numpy as np


def naiveDotProduct(A, B, C, D, E, F, G, H):
    """
    @input A ~ H: a vector of integers
    @return res1 ~ res4: four integer values. res1 = A \cdot B, res2 = C \cdot D, res3 = E \cdot F, res4 = G \cdot H.
    """
    # res1 = A \cdot B
    res1 = 0
    for i in range(len(A)):
        res1 += A[i] * B[i]

    # res2 = C \cdot D
    res2 = 0
    for i in range(len(C)):
        res2 += C[i] * D[i]

    # res3 = E \cdot F
    res3 = 0
    for i in range(len(E)):
        res3 += E[i] * F[i]

    # res4 = G \cdot H
    res4= 0
    for i in range(len(G)):
        res4 += G[i] * H[i]


    return res1, res2, res3, res4

def encode(A, B, C, D):
    """
    @input A ~ D: four vector of integers
    @return E: a vector of integers that encodes A ~ D
    """
    delta = 2 ** 26

    E = [0]*len(A)
    for i in range(len(A)):
        E[i] = A[i] + B[i] * delta + C[i] * delta**3 + D[i] * delta**4
    return E

def decode(E):
    """
    @input C: an integer value
    @return A, B, C, D: the decoded integer values
    """
    delta = 2**26

    A = E % delta
    B = (E // delta**2) % delta
    C = (E // delta**6) % delta
    D = E // delta**8

    return A, B, C, D


def SIMDDotProduct(A, B, C, D, E, F, G, H):
    """
    @input A, B, C, D: a vector of integers
    @return res1, res2: two integer values. res1 = A \cdot B, res2 = C \cdot D.
    """
    # Encode A, C, E, and G into X
    X = encode(A, C, E, G)
    # Encode B, D, F, and H into Y
    Y = encode(B, D, F, H)

    res = 0
    for i in range(len(X)):
        res += X[i] * Y[i]

    res1, res2, res3, res4 = decode(res)

    return res1, res2, res3, res4


A = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
B = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
C = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
D = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
E = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
F = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
G = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()
H = np.random.randint(2**8, size=1000, dtype=np.uint8).tolist()


res1_naive, res2_naive, res3_naive, res4_naive = naiveDotProduct(A, B, C, D, E, F, G, H)
print("res1_naive: ", res1_naive, ", res2_naive: ", res2_naive, ", res3_naive: ", res3_naive, ", res4_naive: ", res4_naive)

res1_SIMD, res2_SIMD, res3_SIMD, res4_SIMD = SIMDDotProduct(A, B, C, D, E, F, G, H)
print("res1_SIMD: ", res1_SIMD, ", res2_SIMD: ", res2_SIMD, ", res3_SIMD: ", res3_SIMD, ", res4_SIMD: ", res4_SIMD)


assert res1_naive == res1_SIMD, "Error: res1_naive != res1_SIMD"
assert res2_naive == res2_SIMD, "Error: res2_naive != res2_SIMD"
assert res3_naive == res3_SIMD, "Error: res3_naive != res3_SIMD"
assert res4_naive == res4_SIMD, "Error: res4_naive != res4_SIMD"











