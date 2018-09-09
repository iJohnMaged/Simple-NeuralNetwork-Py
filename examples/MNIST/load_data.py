"""
This module contains useful methods to load the MNIST dataset for my neural network, and also prepare them in a format
that can be used with pygame to visualize the data.

Source of the MNIST database: http://yann.lecun.com/exdb/mnist/
"""


import os
import numpy as np

files = {
    'X_train': 'train-images.idx3-ubyte',
    'y_train': 'train-labels.idx1-ubyte',
    'X_test': 't10k-images.idx3-ubyte',
    'y_test': 't10k-labels.idx1-ubyte'
}

# The folder containing the data files
data_path = 'DATA'


def read_file(file_name, train = True):

    """
    A method to read a MNIST database file.
    :param file_name: the file to read
    :param train: set this option to true in case of reading training labels
    :return: A list of numpy.ndarray
             In case of images: Each array contains 784 elements. (Pixels of image)
             In case of training labels: Each array contains 10 elements with index i (i being the true label) is set to 1
             In case of test labels: returns a list of ints, each int is the ground truth value for test images.
    """

    with open(f'{data_path}{os.sep}{file_name}', 'rb') as f:
        data = []
        magic_number = int.from_bytes(f.read(4), 'big')
        items_number = int.from_bytes(f.read(4), 'big')

        # 2051: Images
        # 2049: Labels
        if magic_number == 2051:
            # Consume the row/col bytes (Rows = Cols = 28)
            f.read(8)
            for i in range(items_number):
                # Read 784 (28x28) bytes from the file, each byte represents a single pixel in image
                data_arr = np.frombuffer(f.read(784), dtype=np.uint8).reshape(784, 1)
                # Normalize the data to have a value between 0 and 1
                data.append(data_arr / 255)

        else:
            for i in range(items_number):
                target = int.from_bytes(f.read(1), 'big')
                if train:
                    target_arr = np.zeros((10, 1))
                    target_arr[target, 0] = 1
                    data.append(target_arr.reshape(len(target_arr), 1))
                else:
                    data.append(target)

    return data


def extract_images(file_name):

    """
    This module process the MNIST data to extract images in a way that works with PyGame.
    :param file_name: file to process
    :return: list of numpy.ndarray each array is a pixel array of an image and can be used with PyGame.
    """
    with open(f'{data_path}{os.sep}{file_name}', 'rb') as f:
        images = []
        f.read(4)
        images_number = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        for i in range(images_number):
            img = np.zeros((rows, cols, 3))
            for i in range(rows):
                for j in range(cols):
                    byte = int.from_bytes(f.read(1), 'big')
                    img[j, i] = (byte, byte, byte)
            images.append(img)

    return images


# Loading the data
X_train = read_file(files['X_train'])
y_train = read_file(files['y_train'])
X_test = read_file(files['X_test'])
y_test = read_file(files['y_test'], False)


# This can take a while.. I suggest running this once and save the data
# Then comment all the following code and uncomment the loading section!

# Process the data to images
# X_train_images = extract_images(files['X_train'])
# X_test_images = extract_images(files['X_test'])

# Save images on disk:
# np.save(f'{data_path}{os.sep}X_train_images', X_train_images)
# np.save(f'{data_path}{os.sep}X_test_images', X_test_images)

# >>>> UNCOMMENT THIS AFTER SAVING <<<<

# Loading for disk:
X_train_images = np.load(f'{data_path}{os.sep}X_train_images.npy')
X_test_images = np.load(f'{data_path}{os.sep}X_test_images.npy')
