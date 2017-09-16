import numpy as np

def qr_Householder(A):
    n = np.ma.size(A, 0)
    Q = np.identity(n,dtype=None)
    original_A = A

    for k in range(1,n):
        P = k_householder(A,k)
        A = P.dot(A)
        Q = Q.dot(P)

    R = Q.transpose().dot(original_A)

    return Q,R

def k_householder(M,k):
    n = np.ma.size(M,0)
    P = np.identity(n,dtype=None)
    subM = M[k-1:,k-1:]
    subI = np.identity(np.ma.size(subM,0),dtype=None)

    a = subM[:,0]
    u = calculate_U(a)
    v = u/np.linalg.norm(u, ord=None, axis=None)
    subP = subI - 2*v.dot(v.transpose())
    P[k-1:,k-1:] = subP
    return P

def calculate_U(a):
    alfa = np.linalg.norm(a, ord=None, axis=None) * (-1.0) * np.sign(a[0,0])
    e = np.zeros((len(a),1))
    e[0] = 1
    return a - alfa*e
