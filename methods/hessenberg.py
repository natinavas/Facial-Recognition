import numpy as np

def hessenberg_matrix(M):
    n = np.ma.size(M, 0)
    H = M
    T = np.identity(n, dtype=None)
    for i in xrange(n-2):
        P = k_householder_reflection(H,i+1)
        H = P.dot(H).dot(P)
        T = T.dot(P)
    return H,T

def k_householder_reflection(M,k):
    n = np.ma.size(M,0)
    I = np.identity(n,dtype=None)
    a = M[:,k-1]
    W = calculate_W(a,k)
    return I - 2*W.dot(W.T)

def calculate_S(X,k):
    last = X[k:,:]
    squared = list(map(lambda x: x ** 2, last))
    return np.math.sqrt(reduce((lambda x,y: x+y), squared))*np.sign(X[k])

def calculate_R(X,k):
    last = X[k:, :]
    S = calculate_S(X,k)
    squared = list(map(lambda x: x ** 2, last))
    return np.math.sqrt(2*S*X[k] + S*S + reduce((lambda x, y: x + y), squared))

def calculate_Y(X,k):
    Y = np.zeros((len(X),1))
    for i in range(len(X)):
        if(i <= k-1):
            Y[i] = X[i]
        elif(i == k):
            Y[i] = (-1)*calculate_S(X,k)
    return Y

def calculate_W(X,k):
    Y = calculate_Y(X,k)
    R = calculate_R(X,k)
    return (1/R) * (X-Y)