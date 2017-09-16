import numpy as np

def gram_schmidt(A):
    """ Gram Schmidt method for obtaining a QR decomposition """
    # http://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
    Q = np.zeros(A.shape)
    R = np.zeros(A.shape)
    u = np.zeros(A.shape)
    # Assign the values of the first column of A to the first column of u
    u[:, 0] = col(A[:, 0]).reshape(u[:,0].size)
    # Calculate e1 for the Q matrix
    # e1 = u1 / ||u1||
    Q[:,0] = (u[:, 0] / np.linalg.norm(u[:,0], 2))
    # Iterate to obtain all values of Q and R
    for i in range(1,len(A)):
        # u(k+1) = a(k+1) - (a(k+1).e1)e1 - ... - (a(k+1).ek)ek
        # e(k+1) = u(k+1) / ||u(k+1)||
        u[:,i] = A[:,i].reshape(u[:,0].size)
        for j in range(0, i):
            u[:,i] = u[:,i] - (inner(A[:,i], Q[:,j])  * Q[:,j] )
        Q[:,i] = np.divide(u[:,i], np.linalg.norm(u[:,i], 2))
    # Generate R
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