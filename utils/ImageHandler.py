import csv
import numpy as np
import os
import tkFileDialog
from Tkinter import Tk, Toplevel

import scipy
import scipy.misc
from PIL import Image

from utils.ProfileHandler import Profile


def load_images(training_size, testing_size, image_dir):
    training_classes = list()
    training_images = list()
    testing_images = list()
    testing_classes = list()
    profiles = list()
    profile_map = dict()


    for subdir, dirs, files in os.walk(image_dir):
        counter = 0
        first = True
        for file in files:
            if not subdir.startswith('.') and not file.startswith('.') and not file.endswith('.csv'):
                image_class = os.path.basename(os.path.normpath(subdir))
                path = subdir + "/" + file
                if(first):
                    first = False
                    profile_path = subdir + "/profile"
                    profiles.append(load_profile(path, profile_path))
                    profile_map[image_class] = len(profiles)-1

                if counter < training_size:
                    training_classes.append(image_class)
                    training_images.append(list(Image.open(path).getdata()))
                elif counter < testing_size + training_size:
                    testing_classes.append(image_class)
                    testing_images.append(list(Image.open(path).getdata()))
                counter+=1

    return training_images, training_classes, testing_images, testing_classes, profiles, profile_map

def load_testing_images(individual_count, testing_size, image_dir, image_type='.pgm'):
    test_images = list(None for i in range(individual_count * testing_size))

    testing_class = np.zeros(individual_count * testing_size)
    for i in range(1, individual_count + 1):
        for j in range(10 - testing_size + 1, 11):
            dir = image_dir + str(i) + "/" + str(j) + image_type
            test_images[(i - 1) * testing_size + (j - (10 - testing_size)) - 1] = list(
                Image.open(dir).getdata())
            testing_class[(i - 1) * testing_size + (j - (10 - testing_size)) - 1] = i

    return test_images, testing_class

def save_images(images, file_name='outfile', image_type='.pgm'):
    i = 0
    for face in images:
        i += 1
        reshaped_face = np.reshape(face, [112, 92])
        outfile = "output/" + file_name + str(i) + image_type
        scipy.misc.imsave(outfile, reshaped_face)

def save_image(image_matrix, image_name):
    mean_face = np.reshape(image_matrix, [112, 92])
    scipy.misc.imsave(image_name, mean_face)

def load_profile(image_dir, profile_dir, file_type = ".csv"):
    csv_dir = profile_dir + file_type
    reader = csv.reader(open(csv_dir))
    for row in reader:
        return Profile(row[0], int(row[1]), row[2], int(row[3]), image_dir, row[4])
