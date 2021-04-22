"""
Implementation of utility functions.

Written by Boyuan Feng
"""
import numpy as np

def extract_Uint_Weight(weight):
    q = weight.int_repr().numpy().astype(np.int32)
    z = weight.q_per_channel_zero_points().numpy().astype(np.int32)
    s = weight.q_per_channel_scales().numpy()
    assert (z == np.zeros(z.shape)).all(), 'Warning: zero poing is not zero'
    z = 128
    q += 128
    q = q.astype(np.int32)
    return q, z, s

def dump_txt(q, z, s, prefix):
    np.savetxt(prefix+"_q.txt", q.flatten(), fmt='%u', delimiter=',')
    # print(z, s)
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
