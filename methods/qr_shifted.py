import numpy as np
import hessenberg as hess

# Assumes that M matrix is symmetric
def eig_qr_shifted(M, qr_method):
    n = np.ma.size(M, 0)
    H, t = hess.hessenberg_matrix(M)
    B = H
    eigvec = np.identity(n, dtype=None)
    n -= 1
    error = 0.00000001

    while n > 0:
        I = np.identity(n+1, dtype=None)
        while np.abs(B[n,n-1]) >= error:
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
    r1,r2 = np.roots([1, (a*b), (a*b - c*c)])
    r1 = np.abs(r1-b)
    r2 = np.abs(r2-b)
    if r1 < r2:
        return r1
    return r2