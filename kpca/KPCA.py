import numpy as np
import kernel as kernel
from pca import Eig as eig
from qr import Householder as hh

x = np.matrix([1,2,3])
y = np.matrix([4,2,0]).T
M = np.matrix([[1,2,3],[3,6,5],[0,8,1]])
K = kernel.kernel_matrix(M,kernel.polynomial)

eigval,eigvec = eig.get_eig(K,hh.qr_Householder)

sqrt_eig_values = map(lambda x: np.sqrt(np.abs(x)),eigval)
for i in xrange(len(eigval)):
    eigvec[:,i] /= sqrt_eig_values[i]