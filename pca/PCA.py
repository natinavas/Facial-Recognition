import argparse
import numpy as np
import scipy.misc
from PIL import Image
import Eig
import svmclf
import eucclass
from qr import Householder as hh
from qr import GrahamSchmidt as gs
import matplotlib.pyplot as plt

"""http://www.face-rec.org/algorithms/pca/jcn.pdf"""

# Define parser and get input values
parser = argparse.ArgumentParser(description='Facial Recognition software with PCA')
parser.add_argument('--images', '-i', type=str, default="../att_faces/s", dest="images")
parser.add_argument('--image_type', '-it', type=str, default='.pgm', dest="image_type")
parser.add_argument('--training_set_size', '-tss', type=int, default=6, dest="training_set_size")
parser.add_argument('--eig_method', '-em', type=str, dest="method", default="householder")
parser.add_argument('--verbose', '-v', type=bool, default=True, dest="verbose")
parser.add_argument('--type', '-t', type=str, default="lineal", dest="type")
args = parser.parse_args()
THRESHOLD = 0.9

# Check parameters
if not args.method == "householder" and not args.method == "gramschmidt":
    raise ValueError("Eigen method is not supported, choose householder or gramschmidt")

#Training set characteristics
# TODO: get from image folder
AMOUNT_OF_FACES = 40
TRAINING_SET_SIZE = args.training_set_size
training_classes=np.zeros(AMOUNT_OF_FACES*TRAINING_SET_SIZE)


# Load images
if(args.verbose):
    print('Loading images')
images = list(None for i in range(TRAINING_SET_SIZE * AMOUNT_OF_FACES))
for i in range(1,AMOUNT_OF_FACES + 1):
    for j in range(1,TRAINING_SET_SIZE + 1):
        dir= args.images +str(i)+"/"+str(j)+ args.image_type
        images[(i-1)*TRAINING_SET_SIZE+(j-1)]=list(Image.open(dir).getdata())
        training_classes[(i-1)*TRAINING_SET_SIZE+(j-1)]=i

# Create matrix out of images
matrix = (np.matrix(images)).T/255

# Calculate mean different faces
mean = matrix.mean(axis=1)

#TODO guardar mean menos kbezamente
mean_face = np.reshape(mean, [112, 92])
# fig, axes = plt.subplots(1, 1)
# axes.imshow(np.reshape(reshaped_face, [112, 92]), cmap='gray')
outfile = "outfilemean" + ".pgm"
scipy.misc.imsave(outfile, mean_face)

# Divide by standard deviation
# standard_deviation = np.std(matrix, axis=1)
standard_deviation = 1 #TODO

# TODO
centered_matrix = (matrix - mean)/standard_deviation
# centered_matrix = matrix
# Calculate the covariance matrix
# Calculate centered matrix * transposed centered matrix to get a
# similar matrix to the one of the covariance
if(args.verbose):
    print('Calculating covariance matrix')

covariance_matrix = (centered_matrix.T).dot(centered_matrix)

# Calculate eigen values and eigen vectors
if(args.verbose):
    print('Calculating eigen values and eigen vectors')
if args.method == "householder":
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    #eig_values, eig_vectors = Eig.get_eig(covariance_matrix, hh.qr_Householder)
elif args.method == "gramschmidt":
    # eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    eig_values, eig_vectors = Eig.get_eig(covariance_matrix, gs.qr_Gram_Schmidt)
else:
    raise ValueError("The method is not supported")

# Get best eigenvalues
if(args.verbose):
    print('Getting representative eigen values')
sum_eig_values = sum(np.absolute(eig_values))
actual_sum = 0
i = 0
while(actual_sum/sum_eig_values < THRESHOLD):
    actual_sum += abs(eig_values[i])
    i+=1
best_eig_vectors = eig_vectors[:, 0:i]

# Calculate images
# http://blog.manfredas.com/eigenfaces-tutorial/
#eigen_faces = np.zeros((len(centered_matrix), len(best_eig_vectors)))

# for face in range(len(best_eig_vectors)):
#     eigen_faces[:, face] = centered_matrix.dot(np.ravel(best_eig_vectors[face, :]))
eigen_faces = centered_matrix.dot(best_eig_vectors)

# Project values on eigen vectors
if(args.verbose):
    print('Projecting values on eigen vectors')
projected_values = eigen_faces.T.dot(centered_matrix)

# Write image files
i = 0
for face in eigen_faces.T:
    i+=1
    reshaped_face = np.reshape(face, [112, 92])
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(np.reshape(reshaped_face, [112, 92]), cmap='gray')
    outfile = "outfile" + str(i) + ".pgm"
    scipy.misc.imsave(outfile, reshaped_face)

# Load test images
if(args.verbose):
    print('Loading testing images')
test_images = list(None for i in range(AMOUNT_OF_FACES*(10-TRAINING_SET_SIZE)))

testing_class= np.zeros(AMOUNT_OF_FACES*(10-TRAINING_SET_SIZE))
for i in range(1,AMOUNT_OF_FACES + 1):
    for j in range(TRAINING_SET_SIZE+1,11):
        dir= args.images +str(i)+"/"+str(j)+ args.image_type
        print("i {} j {} index {}".format(i,j,(i-1)*(10-TRAINING_SET_SIZE)+(j- TRAINING_SET_SIZE)))
        test_images[(i-1)*(10-TRAINING_SET_SIZE)+(j- TRAINING_SET_SIZE)-1]=list(Image.open(dir).getdata())
        testing_class[(i-1)*(10-TRAINING_SET_SIZE)+(j-TRAINING_SET_SIZE)-1]=i

test_matrix = np.matrix(test_images).T/255
test_matrix = test_matrix - mean

testing_set = eigen_faces.T.dot(test_matrix)


#training_classes = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]


svmclf.svmclassify(training_set=projected_values.T, training_class=training_classes, testing_set=testing_set.T, testing_class=testing_class)