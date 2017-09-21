import numpy as np
import scipy
import scipy.misc
from PIL import Image

def load_training_images(individual_count, training_size, image_dir, image_type='.pgm'):
    training_classes = np.zeros(individual_count * training_size)
    images = list(None for i in range(training_size * individual_count))
    for i in range(1, individual_count + 1):
        for j in range(1, training_size + 1):
            dir = image_dir + str(i) + "/" + str(j) + image_type
            images[(i - 1) * training_size + (j - 1)] = list(Image.open(dir).getdata())
            training_classes[(i - 1) * training_size + (j - 1)] = i

    return images, training_classes

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
        outfile = file_name + str(i) + image_type
        scipy.misc.imsave(outfile, reshaped_face)

def save_image(image_matrix, image_name):
    mean_face = np.reshape(image_matrix, [112, 92])
    scipy.misc.imsave(image_name, mean_face)