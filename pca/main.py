import numpy

N = numpy.random.randn(1000,2)
Matrix = numpy.matrix('1 5; 1 10')
X = N * Matrix

mean = X.mean(axis=1)
X = X - mean

Cx = (1/(len(N)-1)) * numpy.transpose(X) * X
E = numpy.linalg.eig(Cx)
[U,s,Vh] = numpy.linalg.svd(X)

S = numpy.diag(s)

A = Vh.H * S

V = Vh.H

print('-------\n', E[1])
print(Vh.H, '\n-------')
# print('A\n ', A)

Cz = A * numpy.transpose(A)

print('AAAAAAAAAA')
print(Cx)
print('----------')
print(Cz)


algo = A[:,0]*A[:,0].H
algo2 = algo[0,0] / Cx[0,0]

print(algo2)