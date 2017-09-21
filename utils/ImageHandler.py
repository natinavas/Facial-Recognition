import numpy as np
import os
import scipy
import scipy.misc
from PIL import Image


def load_images(training_size, testing_size, image_dir):
    training_classes = list()
    training_images = list()
    testing_images = list()
    testing_classes = list()

    for subdir, dirs, files in os.walk(image_dir):
        counter = 0
        for file in files:
            if not subdir.startswith('.') and not file.startswith('.'):
                image_class = os.path.basename(os.path.normpath(subdir))
                path = subdir + "/" + file
                if counter < training_size:
                    training_classes.append(image_class)
                    training_images.append(list(Image.open(path).getdata()))
                elif counter < testing_size + training_size:
                    testing_classes.append(image_class)
                    testing_images.append(list(Image.open(path).getdata()))
                counter+=1

    return training_images, training_classes, testing_images, testing_classes

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