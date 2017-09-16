import numpy as np
import GS
from methods import Householder as hh
from math import sqrt
from methods import GS
from methods import qr

def qr_Householder(A):
    n = np.ma.size(A, 0)
    Q = np.identity(n,dtype=None)
    original_A = A

    for k in range(1,n):
        P = hh.k_householder(A,k)
        A = P.dot(A)
        Q = Q.dot(P)

    R = Q.transpose().dot(original_A)

    return Q,R

def is_diag(A, error):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if(i != j) and (np.abs(A[i,j]) > error):
                return False
    return True

def get_eig(A):
    #Q, R = householder(A)

    Q, R = qr_Householder(A)
    assert np.allclose(A, Q.dot(R))
    #A = mult_matrix(R, Q)
    A = R.dot(Q)
    C = Q

    while not is_diag(A, 0.00001):
        #Q, R = householder(A)
        Q, R = qr_Householder(A)
        #A = mult_matrix(R, Q)
        #C = mult_matrix(C, Q)
        assert np.allclose(A, Q.dot(R))
        A = R.dot(Q)
        C = C.dot(Q)
    eigvalues = np.diagonal(R)
    # Order the eigenvalues
    indices = eigvalues.argsort()[::-1]
    eigvalues = sorted(eigvalues, key=int, reverse=True)
    C = np.matrix(C)
    C = C[:][indices]
    return eigvalues, C
