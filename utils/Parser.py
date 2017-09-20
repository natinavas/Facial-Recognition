import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Facial Recognition software with PCA')
    parser.add_argument('--images', '-i', type=str, default="../att_faces/s", dest="images")
    parser.add_argument('--image_type', '-it', type=str, default='.pgm', dest="image_type")
    parser.add_argument('--training_set_size', '-tss', type=int, default=6, dest="training_set_size")
    parser.add_argument('--testint_set_size', '-tess', type=int, default=4, dest="testing_set_size")
    parser.add_argument('--qr_method', '-qrm', type=str, dest="qr_method", default="householder")
    parser.add_argument('--eig_method', '-em', type=str, dest="eig_method", default="qr")
    parser.add_argument('--verbose', '-v', type=bool, default=True, dest="verbose")
    parser.add_argument('--type', '-t', type=str, default="lineal", dest="type")
    args = parser.parse_args()

    # Check parameters
    if not args.qr_method == "householder" and not args.qr_method == "gramschmidt":
        raise ValueError("Qr decomposition method is not supported, choose householder or gramschmidt")
    if not args.eig_method == "qr" and not args.eig_method == "qr_shifted":
        raise ValueError("Eigen method is not supported, choose qr or qr_shifted")
    if not args.type == "pca" and not args.type == "kpca":
        raise ValueError("Pca type is not supported, choose pca or kpca")

    return args