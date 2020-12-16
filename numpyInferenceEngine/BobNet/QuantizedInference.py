import numpy as np

debug = True

def FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3):
    # print("z1.shape: ", z1.shape, ", q1.shape: ", q1.shape)
    # print("z2.shape: ", z2.shape, ", q2.shape: ", q2.shape)
    # This version uses floating-point s_i, while using integer q_i and z_i
    # M will be integer in the next version.
    # @Param:
    #   q1, z1, s1: integer value, zero value, and scale factor for weight
    #   q2, z2, s2: integer value, zero value, and scale factor for input feature
    #   q2, z2, s2: integer value, zero value, and scale factor for output feature
    print("q1.shape: ", q1.shape, ", q2.shape: ", q2.shape)

    print("q1 before shift is : \n {}".format(q1))
    if debug:
        print("(q2-z2).shape: ", (q2-z2).shape, ", (q1 - z1[:, None]).shape: ", (q1 - z1[:, None]).shape)
    M = s1 * s2 / s3
    if debug:
        print("M: ", M, ", type(M): ", M.dtype)
    M = (M * 2**24).astype(np.int32)
    # print("q1: ", q1, ", z1: ", z1, ", q1 - z1: ", q1 - z1[:, None])
    q1 = q1.astype(np.int32)
    z1 = z1.astype(np.int32)

    if debug:
        print("M after shift: ", M, ", type(M): ", M.dtype)
    #M~=4000
    #M~=24000
    q3 = (np.matmul(q2 - z2, (q1 - z1[:, None]).T))
    if debug:
        print("q3 before shift: ", q3, ", type(q3): ", q3.dtype)
    q3 = (M*q3 + z3 * 2 **24) // 2**24
    if debug:
        print("q3 after shift: ", q3, ", type(q3): ", q3.dtype)

    #q3 = z3 + q3
    return q3.astype(np.int32)


# (q3 - z3) * 2^24 equality == assert M * np.matmul(q2 - z2, (q1 - z1[:, None]).T)





