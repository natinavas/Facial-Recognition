import numpy as np
from sklearn import svm

from PIL import Image
from scipy.misc import toimage

from classification import svmclf
from kpca import kpca
from methods.GramSchmidt import col, row
import qr
from utils import ImageHandler as imageHandler
from utils import ArgumentParser as arguments
from methods import Householder as hh
from methods import GramSchmidt as gs

"""http://www.face-rec.org/algorithms/pca/jcn.pdf"""

def calculate_eigen(matrix, qr_method, eig_method):
    if(eig_method == "qr"):
        if qr_method == "householder":
            eig_values, eig_vectors = qr.eig_qr(matrix, hh.qr_Householder)
        elif qr_method== "gramschmidt":
            eig_values, eig_vectors = qr.eig_qr(matrix, gs.qr_Gram_Schmidt)
        else:
            raise ValueError("The method is not supported")
    elif(eig_method == "qr_shifted"):
        if qr_method == "householder":
            eig_values, eig_vectors = qr.eig_qr_shifted(matrix, hh.qr_Householder)
        elif qr_method== "gramschmidt":
            eig_values, eig_vectors = qr.eig_qr_shifted(matrix, gs.qr_Gram_Schmidt)
        else:
            raise ValueError("The method is not supported")
    else:
        raise ValueError("The method is not supported")
    return eig_values, eig_vectors

def get_best_eig_vectors(eig_values, eig_vectors, threshold):
    sum_eig_values = sum(np.absolute(eig_values))
    actual_sum = 0
    i = 0
    while (actual_sum / sum_eig_values < threshold):
        actual_sum += abs(eig_values[i])
        i += 1
    best = eig_vectors[:, 0:i]
    return best

def calculate_kernel_eigen(matrix, eig_method, qr_method):
    K = kpca.kernel_matrix(matrix, kpca.polynomial)
    eig_values, eig_vectors = calculate_eigen(matrix=K, eig_method=eig_method, qr_method=qr_method)
    sqrt_eig_values = map(lambda x: np.sqrt(np.abs(x)), eig_values)
    for i in xrange(len(eig_values)):
        eig_vectors[:, i] /= sqrt_eig_values[i]
    return eig_values, eig_vectors

def train():
    THRESHOLD = 0.95 # Proportion of representation when choosing the best eigen vectors

    args = arguments.get_arguments()

    # Training set characteristics
    TRAINING_SET_SIZE = args.training_set_size #Amount of training images for each individual
    # Testing set characteristics
    TESTING_SET_SIZE = args.testing_set_size #Amount of testing images for each individual

    # Load images
    if (args.verbose):
        print('Loading images')
    training_images, training_classes, testing_images, testing_classes, profile, profiles_map = imageHandler.load_images(training_size=TRAINING_SET_SIZE, testing_size=TESTING_SET_SIZE, image_dir=args.images)

    # Create matrix out of images
    matrix = (np.matrix(training_images)).T / 255.

    # Calculate mean different faces
    mean = matrix.mean(axis=1)
    imageHandler.save_image(mean, "mean.pgm")

    # Center matrix
    centered_matrix = (matrix - mean)

    # Calculate the covariance matrix
    # Calculate centered matrix * transposed centered matrix to get a similar matrix to the one of the covariance
    if(args.verbose):
        print('Calculating covariance matrix')
    covariance_matrix = (centered_matrix.T).dot(centered_matrix)

    # Calculate eigen values and eigen vectors
    if (args.verbose):
        print('Calculating eigen values and eigen vectors')
    if args.type == "pca":
        eig_values, eig_vectors = calculate_eigen(matrix=covariance_matrix, qr_method=args.qr_method, eig_method=args.eig_method)
    elif args.type == "kpca":
        eig_values, eig_vectors = calculate_kernel_eigen(matrix=covariance_matrix, qr_method=args.qr_method, eig_method=args.eig_method)
    else:
        raise ValueError("The type is not supported")

    # Get best eigenvalues
    if(args.verbose):
        print('Getting representative eigen values')
    best_eig_vectors = get_best_eig_vectors(eig_values, eig_vectors, THRESHOLD)

    # Calculate images
    # http://blog.manfredas.com/eigenfaces-tutorial/
    eigen_faces = centered_matrix.dot(best_eig_vectors)

    # Normalize eigen faces optimization
    row_sums = np.linalg.norm(eigen_faces, axis=0)
    eigen_faces = np.divide(eigen_faces,row(row_sums))

    # Project values on eigen vectors
    if(args.verbose):
        print('Projecting values on eigen vectors')
    projected_values = eigen_faces.T.dot(centered_matrix)

    # Write image files
    imageHandler.save_images(images=eigen_faces.T)

    # Generate matrices from loaded images
    test_matrix = np.matrix(testing_images).T/255.
    test_matrix = test_matrix - mean

    testing_set = eigen_faces.T.dot(test_matrix)


    #Test images
    clf = svm.LinearSVC()

    # Training classifier with provided data set+group
    clf.fit(projected_values.T, training_classes)
    classifications = clf.score(testing_set.T, testing_classes)

    return mean, eigen_faces, clf, classifications, profile, profiles_map

    #svmclf.svmclassify(training_set=projected_values.T, training_class=training_classes, testing_set=testing_set.T, testing_class=testing_classes)

def test(mean, eigen_faces, clf, image):
    testing_image = np.matrix(image).T/255.
    testing_image = testing_image - mean
    testing_set = eigen_faces.T.dot(testing_image)

    return clf.predict(testing_set.T)
