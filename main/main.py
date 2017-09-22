import tkFileDialog
from Tkinter import Tk

from PIL import Image

from pca import pca
from utils import ArgumentParser as arguments
from kpca import kpca
from utils.ProfileHandler import Profile

args = arguments.get_arguments()


if args.type == "kpca":

    print("doing kpca")
    clf, best_eig_vectors, matrix, K, success_rate, profiles, profile_map = kpca.train_kpca()

    print "\n----------------------------------------"
    print "Correct Classifications", success_rate*100, "%"
    print "----------------------------------------"


    # while True:
    # test_image = list(None for i in range(1))
    # dir = "../att_faces/s3/8.pgm"
    # test_image[0] = list(Image.open(dir).getdata())
    # classification = kpca.test_kpca(clf, best_eig_vectors, matrix, K, test_image)
    # print("\n----------------")
    # print("classification: ")
    # print(classification)
    # print("----------------")
    opts = {}
    opts['filetypes'] = [('PGM Images', '.pgm')]
    root = Tk()
    root.withdraw()
    s = tkFileDialog.askopenfilename(initialdir='../att_faces', **opts)
    test_image = list(None for i in range(1))
    test_image[0] = list(Image.open(s).getdata())
    classification = kpca.test_kpca(clf, best_eig_vectors, matrix, K, test_image)
    profiles[profile_map[classification[0]]].display_portrait()


elif args.type == "pca":

    print("doing pca")
    mean, eigen_faces, clf, classifications, profiles, profile_map = pca.train()


    print "\n----------------------------------------"
    print "Correct Classifications", classifications*100, "%"
    print "----------------------------------------"

    # #while True:
    # #test_image = list(None for i in range(1))
    # dir = "../att_faces/s3/8.pgm"
    # test_image = list(Image.open(dir).getdata())
    # classification = pca.test(mean, eigen_faces, clf, test_image)
    # print("\n----------------")
    # print("classification: ")
    # print(classification)
    # print("----------------")
    opts = {}
    opts['filetypes'] = [('PGM Images', '.pgm')]
    root = Tk()
    root.withdraw()
    s = tkFileDialog.askopenfilename(initialdir='../att_faces', **opts)
    test_image = list(Image.open(s).getdata())
    classification = pca.test(mean, eigen_faces, clf, test_image)

    profiles[profile_map[classification[0]]].display_portrait()

