import numpy as np
import hessenberg as hess

# Assumes that M matrix is symmetric
def eig_qr_shifted(M, qr_method):
    n = np.ma.size(M, 0)
    M,t = hess.hessenberg_matrix(M)
    B = M
    n -= 1
    error = 0.00001

    while n > 0:
        I = np.identity(n+1, dtype=None)
        while np.abs(B[n,n-1]) >= error:
            s = shift(B[n-1,n-1],B[n,n],B[n-1,n])
            Q,R = qr_method(B - s*I)
            B = R.dot(Q) + s*I
        M[0:n+1,0:n+1] = B
        n-= 1
        B = M[0:n+1,0:n+1]

    return np.diag(M)


# def eig_qr_shifted_rec(M, qr_method):
#     n = np.ma.size(M, 0)
#     eigval = np.zeros((1, n))
#     error = 0.00001
#
#     if n == 1:
#         eigval[0,0] = M[0,0]
#     else:
#         I = np.identity(n, dtype=None)
#         H,t = hess.hessenberg_matrix(M)
#
#         while np.abs(H[n-1,n-2]) > error:
#             mu = shift(H[n-2,n-2],H[n-1,n-1],H[n-2,n-1])
#             Q,R = qr_method(H - mu*I)
#             H = R.dot(Q) + mu*I
#
#         eigval[0,n-1] = H[n-1,n-1]
#         eigval[0,0:n-1] = eig_qr_shifted_rec(H[0:n-1,0:n-1],qr_method)
#
#     return eigval


def shift(a,b,c):
    r1,r2 = np.roots([1, (a*b), (a*b - c*c)])
    r1 = np.abs(r1-b)
    r2 = np.abs(r2-b)
    if r1 < r2:
        return r1
    return r2