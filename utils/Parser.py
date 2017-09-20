
# Define parser and get input values
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Facial Recognition software with PCA')
    parser.add_argument('--images', '-i', type=str, default="../att_faces/s", dest="images")
    parser.add_argument('--image_type', '-it', type=str, default='.pgm', dest="image_type")
    parser.add_argument('--training_set_size', '-tss', type=int, default=6, dest="training_set_size")
    parser.add_argument('--eig_method', '-em', type=str, dest="method", default="householder")
    parser.add_argument('--verbose', '-v', type=bool, default=True, dest="verbose")
    parser.add_argument('--type', '-t', type=str, default="lineal", dest="type")
    args = parser.parse_args()

    # Check parameters
    if not args.method == "householder" and not args.method == "gramschmidt":
        raise ValueError("Eigen method is not supported, choose householder or gramschmidt")

    return args