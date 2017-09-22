from PIL import Image

from pca import pca
from utils import ArgumentParser as arguments
from kpca import kpca

args = arguments.get_arguments()


if args.type == "kpca":

    print("doing kpca")
    clf, best_eig_vectors, matrix, K, success_rate = kpca.train_kpca()

    print "\n----------------------------------------"
    print "Correct Classifications", success_rate*100, "%"
    print "----------------------------------------"


    # while True:
    test_image = list(None for i in range(1))
    dir = "../att_faces/s3/8.pgm"
    test_image[0] = list(Image.open(dir).getdata())
    classification = kpca.test_kpca(clf, best_eig_vectors, matrix, K, test_image)
    print("\n----------------")
    print("classification: ")
    print(classification)
    print("----------------")
elif args.type == "pca":

    print("doing pca")
    mean, eigen_faces, clf, classifications = pca.train()


    print "\n----------------------------------------"
    print "Correct Classifications", classifications*100, "%"
    print "----------------------------------------"

    #while True:
    #test_image = list(None for i in range(1))
    dir = "../att_faces/s3/8.pgm"
    test_image = list(Image.open(dir).getdata())
    classification = pca.test(mean, eigen_faces, clf, test_image)
    print("\n----------------")
    print("classification: ")
    print(classification)
    print("----------------")