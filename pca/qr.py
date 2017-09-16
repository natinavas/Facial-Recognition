from math import sqrt
from pprint import pprint
import numpy.linalg as la
import numpy as np
import GS


def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = zip(*N)

    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

def trans_matrix(M):
    """Take the transpose of a matrix."""
    n = len(M)
    return [[ M[i][j] for i in range(n)] for j in range(n)]

def norm(x):
    """Return the Euclidean norm of the vector x."""
    return sqrt(sum([x_i**2 for x_i in x]))

def Q_i(Q_min, i, j, k):
    """Construct the Q_t matrix by left-top padding the matrix Q                                                      
    with elements from the identity matrix."""
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]

def householder(A):
    """Performs a Householder Reflections based QR Decomposition of the                                               
    matrix A. The function returns Q, an orthogonal matrix and R, an                                                  
    upper triangular matrix such that A = QR."""
    n = len(A)

    # Set R equal to A, and create Q as a zero matrix of the same size
    R = A
    Q = [[0.0] * n for i in xrange(n)]

    # The Householder procedure
    for k in range(n-1):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
        # Create identity matrix of same size as A
        I = [[float(i == j) for i in xrange(n)] for j in xrange(n)]

        # Create the vectors x, e and the scalar alpha
        # Python does not have a sgn function, so we use cmp instead
        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -cmp(x[0],0) * norm(x)

        # Using anonymous functions, we create u and v
        u = map(lambda p,q: p + alpha * q, x, e)
        norm_u = norm(u)
        v = map(lambda p: p/norm_u, u)

        # Create the Q minor matrix
        Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in xrange(n-k)] for j in xrange(n-k) ]

        # "Pad out" the Q minor matrix with elements from the identity
        Q_t = [[ Q_i(Q_min,i,j,k) for i in xrange(n)] for j in xrange(n)]

        # If this is the first run through, right multiply by A,
        # else right multiply by Q
        if k == 0:
            Q = Q_t
            R = mult_matrix(Q_t,A)
        else:
            Q = mult_matrix(Q_t,Q)
            R = mult_matrix(Q_t,R)

    # Since Q is defined as the product of transposes of Q_t,
    # we need to take the transpose upon returning it
    return trans_matrix(Q), R

def is_diag(A, error):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if(i != j) and (np.abs(A[i][j]) > error):
                return False
    return True

def get_eig(A):
    #Q, R = householder(A)

    Q, R = GS.gram_schmidt(A)
    assert np.allclose(A, Q.dot(R))
    #A = mult_matrix(R, Q)
    A = R.dot(Q)
    C = Q

    while not is_diag(A, 0.00001):
        #Q, R = householder(A)
        Q, R = GS.gram_schmidt(A)
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


#Example

# A = [[2, 3, 1], [3, 1, 0], [1, 0, 2]]
# eigvec, eigval = get_eig(A)
#
#
# print("eigvec:")
# pprint(eigvec)
# print("eigvals:")
# pprint(eigval)
# print("los posta:")
# posta = la.eig(A)
# pprint(posta[1])
# pprint(posta[0])