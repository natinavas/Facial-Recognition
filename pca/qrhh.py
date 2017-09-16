from math import sqrt
import numpy as np


def QRHH (A):
    m = len(A)
    n = len(A[0])
    Q = np.identity(m)
    for k in range(m):
        z = A[k,k:m]
        print (np.array(-np.sign(z[0])*np.linalg.norm(z,2) - z[0])
        assert(1==2)
        v = np.concatenate( np.array(-np.sign(z[0])*np.linalg.norm(z,2) - z[0]),np.array(-z[1:]))
        print ("z")
        print(z)
        print (len(z))
        print ("v")
        print(v)
        print(len(v))
        assert(1==2)
    #     v = np.divide(v, sqrt(np.transpose(v) * v))
    #
    #     for j in range (n):
    #         A[k:m, j] = A[k:m, j] - v * (2 * (np.transpose(v) * A[k:m, j]))
    #
    #     for j in range (m):
    #         Q[k:m, j] = Q[k:m, j] - v * (2 * (np.transpose(v) * Q[k:m, j]))
    #
    # Q = np.transpose(Q)
    # R = np.triu(A)
    assert(1==2)
    return Q, R


