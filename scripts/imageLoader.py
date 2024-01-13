import os
import cv2
from sklearn.utils import shuffle
import numpy as np


class ImageLoader:
    def __init__(self, test_directory=r"../data/test/", train_directory=r"../data/train/"):
        self.test_directory = test_directory
        self.train_directory = train_directory
        self.Images = []
        self.Labels = []
        self.label = 0

    def get_images(self, directory):
        for labels in os.listdir(directory):
            if labels == 'buildings':
                self.label = 0
            elif labels == 'forest':
                self.label = 1
            elif labels == 'glacier':
                self.label = 2
            elif labels == 'mountain':
                self.label = 3
            elif labels == 'sea':
                self.label = 4
            elif labels == 'street':
                self.label = 5

            for image_file in os.listdir(directory + labels):
                image = cv2.imread(directory + labels + r'/' + image_file)
                image = cv2.resize(image, (100, 100))
                self.Images.append(image)
                self.Labels.append(self.label)

        return shuffle(self.Images, self.Labels, random_state=817328462)

    def get_test_data(self):
        x_test, y_test = self.get_images(self.test_directory)
        test_images = np.array(x_test) / 255
        test_labels = np.array(y_test)
        return test_images, test_labels

    def get_train_data(self):
        x_train, y_train = self.get_images(self.train_directory)
        train_images = np.array(x_train) / 255
        train_labels = np.array(y_train)
        return train_images, train_labels
