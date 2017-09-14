import numpy
from PIL import Image
import qr
THRESHOLD = 0.9

# Load images
images = list(None for i in range(400))
for i in range(1,41):
    for j in range(1,11):
        dir= "/Users/natinavas/Documents/ITBA/MNA/orl_faces/s"+str(i)+"/"+str(j)+".pgm"
        print(list(Image.open(dir).getdata()))
        images[(i-1)*10+(j-1)]=list(Image.open(dir).getdata())

# Create matrix out of images
m = numpy.matrix(images)
matrix = numpy.transpose(m)

print('2----')
# print(matrix)

#Calculate mean for rows
mean = matrix.mean(axis=1)
centered_matrix = matrix - mean
print('3----')

#Calculate the covariance matrix
#TODO: hacer bien
covariance_matrix = numpy.cov(centered_matrix)
print('4----')

# Calculate eigen values
#TODO: hacer bien
#OBS: se esta asumiendo que los eigen values estan ordenados.
eig_values, eig_vectors = numpy.linalg.eig(covariance_matrix)

#Custom eig with QR (householder). eig values returned in descending order
#eig_values, eig_vectors = qr.get_eig(covariance_matrix)
print('5----')

# Get best eigenvalues
sum_eig_values = sum(eig_values)
actual_sum = 0
i = 0
print('6----')
while(actual_sum/sum_eig_values < THRESHOLD):
    actual_sum += eig_values[i]
    i+=1
best_eig_vectors = eig_vectors[:, 0:i]
print('7----')
# Project values on eigen vectors
projected_values = numpy.transpose(centered_matrix)*best_eig_vectors

#print(projected_values)
