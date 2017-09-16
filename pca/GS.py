import numpy as np
from copy import deepcopy

def gram_schmidt(A):

    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    u = np.zeros(A.shape)
    #u[:,0] = col(deepcopy(A[:,0]))
    u[:, 0] = col(A[:, 0]).reshape(u[0].size)
    Q[:,0] = (u[0] / np.linalg.norm(u[0], 2))
    for i in range(1,len(A)):
        u[:,i] = A[:,i].reshape(u[0].size)
        for j in range(0, i):
            u[:,i] = u[:,i] - (inner(A[:,i], Q[:,j])  * Q[:,j] )
        Q[:,i] = np.divide(u[i], np.linalg.norm(u[i], 2))
    for i in range(0, A.shape[0]):
        for j in range(i, A.shape[1]):
                R[i,j] = inner(A[:,j], Q[:,i])
    return Q, R


def col(n):
    return n.reshape((n.size,1))

def row(n):
    return n.reshape((1,n.size))

def inner(n1,n2):
    return np.dot(row(n1),col(n2))