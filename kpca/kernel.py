import numpy as np

'''
    K = M - M #TODO esto es un asco barto

    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            K[row,col] = kernel_method(M[row,:],M[:,col])
'''

def kernel_matrix(M,kernel_method, TRAINING_IMAGES):

    degree = 2
    K = (np.dot(M, M.T) / TRAINING_IMAGES + 1) ** degree

    return center_k_matrix(K, TRAINING_IMAGES)

def center_k_matrix(K, TRAINING_IMAGES):


    unoM = np.ones([TRAINING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    return K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))



    # n = K.shape[0]
    # ones_over_n = np.ones(K.shape) / n
    #
    # return K - ones_over_n.dot(K) - K.dot(ones_over_n) + ones_over_n.dot(K.dot(ones_over_n))

# x is a row and y is a column
def polynomial(x,y):
    c = 0
    d = 2
    xy = x.dot(y)
    assert(xy.shape == (1,1))
    return (xy[0,0] + c)**d