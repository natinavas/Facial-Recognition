import numpy as np
import householder as hh

def qr_Householder(A):
    n = np.ma.size(A, 0)
    R = A
    Q = np.identity(n,dtype=None)

    for k in range(1,n-1):
        P = hh.k_householder(A,k)
        R = P*R
        Q = Q*np.ma.transpose(P)

    return Q,R