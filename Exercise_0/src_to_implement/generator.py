import os.path
import json
import random
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import math


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.file_path = file_path  # declaring instances
        self.batch_size = batch_size
        self.image_size = image_size  # e.g., [32, 32, 3]
        self.rotation = rotation
        self.mirroring = mirroring  # indicating whether to randomly mirror (flip) images horizontally for augmentation tis also increases data variety and helps prevent overfitting.
        self.shuffle = shuffle  # shuffle after each epoch

        with open(label_path, 'r') as f:  # json files contain images mapping, to help with labeling
            self.labels = json.load(f)

        self.image_files = [f for f in os.listdir(file_path) if f.endswith('.npy')]  # ists of all image filenames

        self.current_index = 0  # index to keep track of batche this helps track

        self.epoch = 0  # Epoch counter(full passees counter)

        self.indices = np.arange(len(self.image_files))  # Prepare indices for shuffling

        if self.shuffle:
            np.random.shuffle(self.indices)

        # Class label names
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

    def next(self):
        images = []  # initalizes empty lists for images and labels
        labels = []

        for _ in range(self.batch_size):
            if self.current_index >= len(
                    self.indices):  # check inside loop after every image fetch to chec if all images have been checked
                self.current_index = 0
                self.epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.indices)

            idx = self.indices[self.current_index]  # gets the shuffled index for current image
            file_name = self.image_files[idx]  # file image fro the image
            label = self.labels[os.path.splitext(file_name)[0]]  # lokos for the labels for the image

            img = np.load(os.path.join(self.file_path, file_name))  # loads images
            img = resize(img, self.image_size, preserve_range=True)  # image resize

            if self.mirroring and random.choice(
                    [True, False]):  # if mirroring is chosn then image if flipped horizontally
                img = np.fliplr(img)

            if self.rotation:  # if rotation selcted it is rotated by 90 degrees
                k = random.choice([0, 1, 2, 3])
                img = np.rot90(img, k)

            images.append(img)
            labels.append(label)

            self.current_index += 1  # increment after check & append

        return np.array(images), np.array(labels)

    def augment(self, img):  # this increase variety and reduce overfitting.

        # TODO: implement augmentation function
        if self.mirroring and np.random.rand() > 0.5:  # this function takes a single image as an input and performs a random transformation
            # (mirroring and/or rotation) on it and outputs the transformed image
            img = np.fliplr(img)

        if self.rotation:
            k = np.random.choice([0, 1, 2, 3])  # Rotate 0, 90, 180, 270 degrees
            img = np.rot90(img, k)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch  # tracking training progress and knowing how many times the dataset has been iterated over.

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict.get(x, "Unknown")  # logging class names instead of numeric labels

    def show(self):
        images, labels = self.next()

        n_images = len(images)
        # Calculate grid size (square-like)
        cols = min(10, n_images)  # max 10 images per row for readability
        rows = math.ceil(n_images / cols)

        plt.figure(figsize=(cols * 2, rows * 2))  # adjust figure size accordingly

        for i in range(n_images):  # loops throgh each image in the batch
            plt.subplot(rows, cols, i + 1)  # convets image to unsigned 8 bith integer
            plt.imshow(images[i].astype(np.uint8))
            plt.title(self.class_name(labels[i]), fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

