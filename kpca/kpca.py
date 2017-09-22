import numpy as np
from methods.GramSchmidt import col, row
from pca import qr
from pca.qr import eig_qr_shifted, eig_qr
from utils import ImageHandler as imageHandler
from utils import ArgumentParser as arguments
from methods import Householder as hh
from methods import GramSchmidt as gs
from sklearn import svm

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

def get_best_eig_vectors(eig_values, eig_vectors, threshold):
    sum_eig_values = sum(np.absolute(eig_values))
    actual_sum = 0
    i = 0
    while (actual_sum / sum_eig_values < threshold):
        actual_sum += abs(eig_values[i])
        i += 1
    best = eig_vectors[:, 0:i]
    return best

def calculate_kernel_eigen(matrix, eig_method, qr_method, TRAINING_IMAGES):
    K = matrix.dot(matrix.transpose())

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = ((K[i,j]/TRAINING_IMAGES)+1)**2

    ones_training = np.ones([TRAINING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    K = K - np.dot(ones_training, K) - np.dot(K, ones_training) + np.dot(ones_training, np.dot(K, ones_training))

    eig_values, eig_vectors = calculate_eigen(matrix=K, eig_method=eig_method, qr_method=qr_method)
    sqrt_eig_values = map(lambda x: np.sqrt(np.abs(x)), eig_values)
    for i in xrange(len(eig_values)):
        eig_vectors[:, i] /= sqrt_eig_values[i]
    return eig_values, eig_vectors, K


##########

def train_kpca():

    THRESHOLD = 0.95 # Proportion of representation when choosing the best eigen vectors

    args = arguments.get_arguments()

    # Testing set characteristics
    TESTING_SET_SIZE = args.testing_set_size


    # Training set characteristics
    TRAINING_SET_SIZE = args.training_set_size  # Amount of training images for each individual

    # Load images
    if (args.verbose):
        print('Loading images')
    training_images, training_classes, testing_images, testing_classes = imageHandler.load_images(training_size=TRAINING_SET_SIZE,
                                                                 testing_size=TESTING_SET_SIZE, image_dir=args.images)


    TRAINING_IMAGES = len(training_images)


    # Create matrix out of images
    matrix = ((np.matrix(training_images)) - 127.5) / 127.5

    # Calculate K matrix and get eigen values and eigen vectors
    if (args.verbose):
        print('Calculating eigen values and eigen vectors')
    eig_values, eig_vectors, K = calculate_kernel_eigen(matrix=matrix, qr_method=args.qr_method,
                                                     eig_method=args.eig_method, TRAINING_IMAGES=TRAINING_IMAGES)

    # Get best eigenvalues
    if(args.verbose):
        print('Getting representative eigen values')
    eigen_faces = get_best_eig_vectors(eig_values, eig_vectors, THRESHOLD)
    if(args.verbose):
        print('number of eigen faces used: ', eigen_faces.shape[1])

    training_projection = np.dot(K.T, eigen_faces)

    clf = svm.LinearSVC()
    clf.fit(training_projection, training_classes)

    testing_projection = get_testing_data(matrix, eigen_faces, testing_images, K)

    classifications = clf.score(testing_projection, testing_classes)

    return clf, eigen_faces, matrix, K, classifications


def test_kpca(clf, eigen_faces, matrix, K, test_image):
    testing_projection = get_testing_data(matrix, eigen_faces, test_image, K)

    classification = clf.predict(testing_projection)

    return classification

def get_testing_data(matrix, eigen_faces, test_images, K):
    TRAINING_IMAGES = matrix.shape[0]
    TESTING_IMAGES = len(test_images)

    test_matrix = np.matrix(test_images)
    test_matrix = (test_matrix - 127.5) / 127.5

    ones_training = np.ones([TRAINING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    ones_testing = np.ones([TESTING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    Ktest = (np.asarray(np.dot(test_matrix, matrix.T)) / TRAINING_IMAGES + 1) ** 2
    Ktest = Ktest - np.dot(ones_testing, K) - np.dot(Ktest, ones_training) + np.dot(ones_testing,
                                                                                    np.dot(K, ones_training))
    testing_projection = np.dot(Ktest, eigen_faces)

    return testing_projection