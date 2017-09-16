import argparse
import numpy as np
import scipy.misc
from PIL import Image
import qr

"""http://www.face-rec.org/algorithms/pca/jcn.pdf"""

# Define parser and get input values
DEFAULT_PATH = "../att_faces/s"
parser = argparse.ArgumentParser(description='Facial Recognition software with PCA')
parser.add_argument('--images', '-i', type=str, default=DEFAULT_PATH, dest="images")
parser.add_argument('--image_type', '-it', type=str, default='.pgm', dest="image_type")
parser.add_argument('--training_set_size', '-tss', type=int, dest="training_set_size")
args = parser.parse_args()
THRESHOLD = 0.9

#Training set characteristics
# TODO: get from image folder
AMOUNT_OF_FACES = 5
# TODO get from parameters
TRAINING_SET_SIZE = 5

# Load images
print('Loading images')
images = list(None for i in range(TRAINING_SET_SIZE * AMOUNT_OF_FACES))
for i in range(1,AMOUNT_OF_FACES + 1):
    for j in range(1,TRAINING_SET_SIZE + 1):
        dir= args.images +str(i)+"/"+str(j)+ args.image_type
        images[(i-1)*TRAINING_SET_SIZE+(j-1)]=list(Image.open(dir).getdata())

# Create matrix out of images
matrix = np.matrix(images)
# Calculate mean for rows
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

# Calculate images
# http://blog.manfredas.com/eigenfaces-tutorial/
matrix = matrix.T
eigen_faces = np.zeros(matrix.shape)

for face in range(AMOUNT_OF_FACES * TRAINING_SET_SIZE):
    eigen_faces[:, face] = matrix.dot(np.ravel(eig_vectors[:, face]))

i = 0
for face in eigen_faces.T:
    i+=1
    reshaped_face = np.reshape(face, [112, 92])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1)
    axes.imshow(np.reshape(reshaped_face, [112, 92]), cmap='gray')
    outfile = "outfile" + str(i) + ".pgm"
    scipy.misc.imsave(outfile, reshaped_face)
