import numpy as np

import qr
from classification import svmclf
from utils import ImageHandler as imageHandler
from utils import Parser as arguments

"""http://www.face-rec.org/algorithms/pca/jcn.pdf"""

def calculate_eigen(qr_method, eig_method):
    # TODO agregar eig_method de qr shifted
    if qr_method == "householder":
        from methods import Householder as hh
        eig_values, eig_vectors = qr.eig_qr(covariance_matrix, hh.qr_Householder)
    elif qr_method== "gramschmidt":
        from methods import GrahamSchmidt as gs
        eig_values, eig_vectors = qr.eig_qr(covariance_matrix, gs.qr_Gram_Schmidt)
    else:
        raise ValueError("The method is not supported")
    return eig_values, eig_vectors

def best_eig_vectors(eig_values, eig_vectors, threshold):
    sum_eig_values = sum(np.absolute(eig_values))
    actual_sum = 0
    i = 0
    while (actual_sum / sum_eig_values < threshold):
        actual_sum += abs(eig_values[i])
        i += 1
    best = eig_vectors[:, 0:i]
    return best

##########

THRESHOLD = 0.95 # Proportion of representation when choosing the best eigen vectors

args = arguments.get_arguments()

#Training set characteristics
TRAINING_INDIVIDUALS = 40 #Amount of different individuals that will be classified
TRAINING_SET_SIZE = args.training_set_size #Amount of training images for each individual

# Testing set characteristics
TESTING_SET_SIZE = args.testing_set_size
TESTING_INDIVIDUALS = TRAINING_INDIVIDUALS #TODO poner por parametros ambos y con las excepciones correspondientes

# Load images
if (args.verbose):
    print('Loading images')
images, training_classes = imageHandler.load_training_images(individual_count=TRAINING_INDIVIDUALS, training_size=TRAINING_SET_SIZE,
                                                             image_dir=args.images, image_type=args.image_type)

# Create matrix out of images
matrix = (np.matrix(images)).T
# matrix = (np.matrix(images)).T/255.

# Calculate mean different faces
mean = matrix.mean(axis=1)
imageHandler.save_image(mean, "mean.pgm")

# Divide by standard deviation
# standard_deviation = np.std(matrix, axis=1)
standard_deviation = 1 #TODO
centered_matrix = (matrix - mean)/standard_deviation

# Calculate the covariance matrix
# Calculate centered matrix * transposed centered matrix to get a similar matrix to the one of the covariance
if(args.verbose):
    print('Calculating covariance matrix')
covariance_matrix = (centered_matrix.T).dot(centered_matrix)

# Calculate eigen values and eigen vectors
if (args.verbose):
    print('Calculating eigen values and eigen vectors')
eig_values, eig_vectors = calculate_eigen(args.qr_method, args.eig_method)

# Get best eigenvalues
if(args.verbose):
    print('Getting representative eigen values')
best_eig_vectors = best_eig_vectors(eig_values, eig_vectors, THRESHOLD)

# Calculate images
# http://blog.manfredas.com/eigenfaces-tutorial/
eigen_faces = centered_matrix.dot(best_eig_vectors)

# Project values on eigen vectors
if(args.verbose):
    print('Projecting values on eigen vectors')
projected_values = eigen_faces.T.dot(centered_matrix)

# Write image files
imageHandler.save_images(images=eigen_faces.T)

# Load test images
if(args.verbose):
    print('Loading testing images')
test_images, testing_class = imageHandler.load_testing_images(individual_count=TESTING_INDIVIDUALS, testing_size=TESTING_SET_SIZE, image_dir=args.images, image_type=args.image_type)

# Generate matrices from loaded images
test_matrix = np.matrix(test_images).T
# test_matrix = np.matrix(test_images).T/255.
test_matrix = test_matrix - mean

testing_set = eigen_faces.T.dot(test_matrix)

svmclf.svmclassify(training_set=projected_values.T, training_class=training_classes, testing_set=testing_set.T, testing_class=testing_class)
