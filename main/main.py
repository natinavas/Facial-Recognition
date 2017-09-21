from PIL import Image
from utils import ArgumentParser as arguments
from kpca import kernel_main

args = arguments.get_arguments()


if args.type == "kpca":
    clf, best_eig_vectors, matrix, K = kernel_main.train_kpca()

    # while True:
    test_image = list(None for i in range(1))
    dir = "../att_faces/s3/8.pgm"
    test_image[0] = list(Image.open(dir).getdata())
    classification = kernel_main.test_kpca(clf, best_eig_vectors, matrix, K, test_image)
    print("")
    print("----------------")
    print("classification: ")
    print(classification)
    print("----------------")