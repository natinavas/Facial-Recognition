import numpy as np

def kernel_matrix(M,kernel_method):
    K = np.zeros(M.shape)

    for row in xrange(M.shape[0]):
        for col in xrange(M.shape[1]):
            K[row][col] = kernel_method(M[row,:],M[:,col])

    return center_k_matrix(K)

def center_k_matrix(K):
    n = K.shape[0]
    ones_over_n = np.ones(K.shape) / n

    return K - ones_over_n.dot(K) - K.dot(ones_over_n) + ones_over_n.dot(K).dot(ones_over_n)

# x is a row and y is a column
def polynomial(x,y):
    c = 0
    d = 2
    xy = x.dot(y)
    assert(xy.shape == (1,1))
    return (xy[0,0] + c)**d