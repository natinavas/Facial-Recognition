import numpy as np
# from classification import svmclf
from kpca import kernel
from methods.GrahamSchmidt import col, row
from pca import qr
from pca.qr import eig_qr_shifted, eig_qr
from utils import ImageHandler as imageHandler
from utils import ArgumentParser as arguments
from methods import Householder as hh
from methods import GramSchmidt as gs
from utils import timer
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
    best_eig_vectors = get_best_eig_vectors(eig_values, eig_vectors, THRESHOLD)
    if(args.verbose):
        print('number of eigen faces used: ', best_eig_vectors.shape[1])

    eigen_faces = np.dot(K.T, best_eig_vectors)

    clf = svm.LinearSVC()
    clf.fit(eigen_faces, training_classes)

    return clf, best_eig_vectors, matrix, K


def test_kpca(clf, best_eig_vectors, matrix, K, test_image):
    #TODO: check shape!!
    TRAINING_IMAGES = matrix.shape[0]

    args = arguments.get_arguments()
    TESTING_SET_SIZE = args.testing_set_size

    args = arguments.get_arguments()

    # Load test images
    # if(args.verbose):
    #     print('Loading testing images')
    # test_images, testing_class = imageHandler.load_testing_images(individual_count=TESTING_INDIVIDUALS, testing_size=TESTING_SET_SIZE, image_dir=args.images, image_type=args.image_type)

    #TODO: check shape!!
    TESTING_IMAGES = len(test_image)

    test_matrix = np.matrix(test_image)
    test_matrix = (test_matrix - 127.5) / 127.5

    ones_training = np.ones([TRAINING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    ones_testing = np.ones([TESTING_IMAGES, TRAINING_IMAGES]) / TRAINING_IMAGES
    Ktest = (np.asarray(np.dot(test_matrix, matrix.T))/TRAINING_IMAGES+1)**2
    Ktest = Ktest - np.dot(ones_testing, K) - np.dot(Ktest, ones_training) + np.dot(ones_testing, np.dot(K, ones_training))
    testing_projection = np.dot(Ktest, best_eig_vectors)

    classification = clf.predict(testing_projection)

    return classification