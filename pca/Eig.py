import numpy as np
from qr import Householder as hh
from qr import GrahamSchmidt as gs

def is_diag(A, error):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if(i != j) and (np.abs(A[i,j]) > error):
                return False
    return True

def get_eig(A, eig_method):
    Q, R = eig_method(A)
    assert np.allclose(A, Q.dot(R))
    A = R.dot(Q)
    C = Q

    while not is_diag(A, 0.00001):
        Q, R = eig_method(A)
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
