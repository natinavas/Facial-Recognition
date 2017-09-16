import numpy as np
from PIL import Image
import qr
THRESHOLD = 0.9

# Load images
print('Loading images')
images = list(None for i in range(20))
for i in range(1,3):
    for j in range(1,11):
        dir= "/Users/natinavas/Documents/ITBA/MNA/orl_faces/s"+str(i)+"/"+str(j)+".pgm"
        # dir= "../att_faces/orl_faces/s"+str(i)+"/"+str(j)+".pgm"
        images[(i-1)*10+(j-1)]=list(Image.open(dir).getdata())

# Create matrix out of images
matrix = np.matrix(images)

#Calculate mean for rows
mean = matrix.mean(axis=1)
centered_matrix = matrix - mean

#Calculate the covariance matrix
print('Calculating covariance matrix')
covariance_matrix = centered_matrix.dot(centered_matrix.T)\

# Calculate eigen values and eigen vectors
print('Calculating eigen values and eigen vectors')
eig_values, eig_vectors = qr.get_eig(covariance_matrix)


# Get best eigenvalues
print('Getting representative eigen values')
sum_eig_values = sum(eig_values)
actual_sum = 0
i = 0
while(actual_sum/sum_eig_values < THRESHOLD):
    actual_sum += eig_values[i]
    i+=1
best_eig_vectors = eig_vectors[:, 0:i]

# Project values on eigen vectors
print('Projecting values on eigen vectors')
projected_values = np.transpose(centered_matrix)*best_eig_vectors


