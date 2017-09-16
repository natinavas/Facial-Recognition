import numpy as np
from PIL import Image
import qr
import argparse

DEFAULT_PATH = "../att_faces/orl_faces/s"

parser = argparse.ArgumentParser(description='Facial Recognition software with PCA')
parser.add_argument('--images', '-i', type=str, default=DEFAULT_PATH, dest="images")
parser.add_argument('--image_type', '-it', type=str, default='.pgm', dest="image_type")
args = parser.parse_args()
THRESHOLD = 0.9

#Training set size

TRAINING_SET_SIZE = 5
AMOUNT_OF_FACES = 3

# Load images
print('Loading images')
images = list(None for i in range(15))
for i in range(1,AMOUNT_OF_FACES + 1):
    for j in range(1,TRAINING_SET_SIZE + 1):
        dir= args.images +str(i)+"/"+str(j)+ args.image_type
        images[(i-1)*TRAINING_SET_SIZE+(j-1)]=list(Image.open(dir).getdata())

# Create matrix out of images
matrix = np.matrix(images)
#Calculate mean for rows
mean = matrix.mean(axis=1)
centered_matrix = matrix - mean

# Calculate the covariance matrix
# Calculate centered matrix * transposed centered matrix to get a
# similar matrix to the one of the covariance
print('Calculating covariance matrix')
covariance_matrix = centered_matrix.dot(centered_matrix.T)

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

import scipy.misc
scipy.misc.imsave('outfile.pgm', projected_values)