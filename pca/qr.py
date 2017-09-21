import numpy as np

from methods import hessenberg as hess

def is_diag(A, error):
    """ Check if a matrix only haves ceros or numbers
    smaller than error under the diagonal"""
    for i in range(len(A)):
        for j in range(len(A[0])):
            if(i != j) and (np.abs(A[i,j]) > error):
                return False
    return True

def eig_qr(A, qr_method):
    """ Method used to calculate eigen vectors and eigen values"""
    Q, R = qr_method(A)
    resta = np.subtract(A, Q.dot(R))
    # assert np.allclose(A, Q.dot(R))
    A = R.dot(Q)
    C = Q
    # Iterate until the matrix obtained is a diagonal matrix
    while not is_diag(A, 0.00001):
        Q, R = qr_method(A)
        assert np.allclose(A, Q.dot(R))
        A = R.dot(Q)
        C = C.dot(Q)
    eigvalues = np.diagonal(R)

    # Order the eigenvalues
    # Save the new order of the old indices in order to reorder the eigen vectors (C)
    indices = sorted(range(len(eigvalues)), reverse=True, key=lambda x: abs(eigvalues[x]))
    eigvalues = sorted(eigvalues, key=abs, reverse=True)
    C = np.matrix(C)
    C = C[:,indices]
    return eigvalues, C

# Assumes that M matrix is symmetric
def eig_qr_shifted(M, qr_method):
    n = np.ma.size(M, 0)
    H, t = hess.hessenberg_matrix(M)
    B = H
    eigvec = np.identity(n, dtype=None)
    n -= 1
    error = 0.001

    while n > 0:
        # print n
        I = np.identity(n+1, dtype=None)
        while np.abs(B[n,n-1]) >= error:
            print np.abs(B[n,n-1])
            s = shift(B[n-1,n-1],B[n,n],B[n-1,n])
            Q,R = qr_method(B - s*I)

            #adapt Q to original size
            i = np.identity(np.ma.size(M, 0), dtype=None)
            i[0:n+1,0:n+1] = Q
            eigvec = eigvec.dot(i)

            B = R.dot(Q) + s*I

        H[0:n + 1, 0:n + 1] = B
        n -= 1
        B = H[0:n + 1, 0:n + 1]

    return np.diag(H), t.dot(eigvec)



def shift(a,b,c):
    r1,r2 = np.roots([1, -(a+b), (a*b - c)])
    ra = np.abs(r1-b)
    rb = np.abs(r2-b)
    if ra < rb:
        return r1
    return r2