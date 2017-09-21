import numpy as np
from classification import svmclf
from kpca import kernel
from methods.GrahamSchmidt import col, row
from pca import qr
from pca.qr import eig_qr_shifted, eig_qr
from utils import ImageHandler as imageHandler
from utils import ArgumentParser as arguments
from methods import Householder as hh
from methods import GrahamSchmidt as gs
from utils import timer

"""http://www.face-rec.org/algorithms/pca/jcn.pdf"""

def calculate_eigen(matrix, qr_method, eig_method):
    if(eig_method == "qr"):
        if qr_method == "householder":
            eig_values, eig_vectors = eig_qr(matrix, hh.qr_Householder)
        elif qr_method== "gramschmidt":
            eig_values, eig_vectors = qr.eig_qr(matrix, gs.qr_Gram_Schmidt)
        else:
            raise ValueError("The method is not supported")
    elif(eig_method == "qr_shifted"):
        if qr_method == "householder":
            eig_values, eig_vectors = eig_qr_shifted(matrix, hh.qr_Householder)
        elif qr_method== "gramschmidt":
            eig_values, eig_vectors = eig_qr_shifted(matrix, gs.qr_Gram_Schmidt)
        else:
            raise ValueError("The method is not supported")
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

def calculate_kernel_eigen(matrix, eig_method, qr_method):
    # K = kernel.kernel_matrix(matrix, kernel.polynomial, TRAINING_IMAGES)

    # images_matrix = np.matrix.astype(np.ndarray)


    K = matrix.dot(matrix.transpose())

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = ((K[i,j]/TRAINING_IMAGES)+1)**2

    # K = (K + K.T)/2.0

    unoM = np.ones([TRAINING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    K = K - np.dot(unoM, K) - np.dot(K, unoM) + np.dot(unoM, np.dot(K, unoM))


    eig_values, eig_vectors = calculate_eigen(matrix=K, eig_method=eig_method, qr_method=qr_method)
    sqrt_eig_values = map(lambda x: np.sqrt(np.abs(x)), eig_values)
    for i in xrange(len(eig_values)):
        eig_vectors[:, i] /= sqrt_eig_values[i]
    return eig_values, eig_vectors, K


##########

THRESHOLD = 0.95 # Proportion of representation when choosing the best eigen vectors

args = arguments.get_arguments()

# Training set characteristics
TRAINING_INDIVIDUALS = 40 #Amount of different individuals that will be classified
TRAINING_SET_SIZE = args.training_set_size #Amount of training images for each individual

TRAINING_IMAGES = TRAINING_INDIVIDUALS * TRAINING_SET_SIZE

# Testing set characteristics
TESTING_SET_SIZE = args.testing_set_size
TESTING_INDIVIDUALS = TRAINING_INDIVIDUALS #TODO poner por parametros ambos y con las excepciones correspondientes

TESTING_IMAGES = TESTING_INDIVIDUALS * TESTING_SET_SIZE

# Load images
if (args.verbose):
    print('Loading images')
images, training_classes = imageHandler.load_training_images(individual_count=TRAINING_INDIVIDUALS, training_size=TRAINING_SET_SIZE,
                                                             image_dir=args.images, image_type=args.image_type)

# Create matrix out of images
matrix = ((np.matrix(images)) - 127.5) / 127.5

# Calculate mean different faces
# mean = matrix.mean(axis=1)
# imageHandler.save_image(mean, "mean.pgm")

# Divide by standard deviation
# standard_deviation = np.std(matrix, axis=1)
# standard_deviation = 1 #TODO
# centered_matrix = (matrix - mean)/standard_deviation

# Calculate the covariance matrix
# Calculate centered matrix * transposed centered matrix to get a similar matrix to the one of the covariance
#if(args.verbose):
#    print('Calculating covariance matrix')
#covariance_matrix = (centered_matrix.T).dot(centered_matrix)

# Calculate eigen values and eigen vectors
# if (args.verbose):
#     print('Calculating eigen values and eigen vectors')
# if args.type == "pca":
#     eig_values, eig_vectors = calculate_eigen(matrix=covariance_matrix, qr_method=args.qr_method, eig_method=args.eig_method)
# elif args.type == "kpca":
#     eig_values, eig_vectors = calculate_kernel_eigen(matrix=covariance_matrix, qr_method=args.qr_method, eig_method=args.eig_method)
# else:
#     raise ValueError("The type is not supported")


if (args.verbose):
    print('Calculating eigen values and eigen vectors')
eig_values, eig_vectors, K = calculate_kernel_eigen(matrix=matrix, qr_method=args.qr_method,
                                                 eig_method=args.eig_method)





# Get best eigenvalues
# if(args.verbose):
#     print('Getting representative eigen values')
# best_eig_vectors = best_eig_vectors(eig_values, eig_vectors, THRESHOLD)
#
# if(args.verbose):
#     print('number of eigen faces used: ', best_eig_vectors.shape[1])

eigen_faces = np.dot(K.T, eig_vectors)



# Calculate images
# http://blog.manfredas.com/eigenfaces-tutorial/
# eigen_faces = centered_matrix.dot(best_eig_vectors)

# Normalize eigen faces optimization
# row_sums = np.linalg.norm(eigen_faces, axis=0)
# eigen_faces = np.divide(eigen_faces,row(row_sums))

# Project values on eigen vectors
# if(args.verbose):
#     print('Projecting values on eigen vectors')
# projected_values = eigen_faces.T.dot(centered_matrix)

# Write image files
# imageHandler.save_images(images=eigen_faces.T)

# Load test images
if(args.verbose):
    print('Loading testing images')
test_images, testing_class = imageHandler.load_testing_images(individual_count=TESTING_INDIVIDUALS, testing_size=TESTING_SET_SIZE, image_dir=args.images, image_type=args.image_type)


test_matrix = np.matrix(test_images)
test_matrix = (test_matrix - 127.5) / 127.5


unoM = np.ones([TRAINING_IMAGES, TRAINING_IMAGES])/TRAINING_IMAGES

unoML = np.ones([TESTING_IMAGES, TRAINING_IMAGES])/TRAINING_IMAGES
Ktest = (np.asarray(np.dot(test_matrix, matrix.T))/TRAINING_INDIVIDUALS+1)**2
Ktest = Ktest - np.dot(unoML, K) - np.dot(Ktest, unoM) + np.dot(unoML, np.dot(K, unoM))
testing_projection = np.dot(Ktest, eig_vectors)



# Generate matrices from loaded images
# test_matrix = np.matrix(test_images).T/255.
# test_matrix = test_matrix - mean
#
# testing_set = eigen_faces.T.dot(test_matrix)

svmclf.svmclassify(training_set=eigen_faces, training_class=training_classes, testing_set=testing_projection, testing_class=testing_class)
