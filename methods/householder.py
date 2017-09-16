import numpy as np

def k_householder(M,k):
    n = np.ma.size(M,0)
    I = np.identity(n,dtype=None)
    a = M[:,k-1]
    u = calculate_U(a,k)
    v = u/np.linalg.norm(u, ord=None, axis=None, keepdims=False)
    return I - 2*v*np.ma.transpose(v)

def calculate_U(a, k):
    alfa = np.linalg.norm(a, ord=None, axis=None, keepdims=False)
    e = np.zeros((len(a),1))
    e[k-1] = 1
    return a - alfa*e

''' Not sure if this is usefull

def calculate_S(X,k):
    last = X[k:,:]
    squared = list(map(lambda x: x ** 2, last))
    return np.math.sqrt(reduce((lambda x,y: x+y), squared))

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
    
'''

