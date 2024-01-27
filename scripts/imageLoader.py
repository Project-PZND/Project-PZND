import os
import cv2
from sklearn.utils import shuffle
import numpy as np
from skimage.color import rgb2gray
import tensorflow as tf
import config as cfg
import matplotlib.pyplot as plt


class ImageLoader:
    def __init__(self, test_directory=cfg.Images.test_directory, train_directory=cfg.Images.train_directory,
                 greyscale=False):
        self.test_directory = test_directory
        self.train_directory = train_directory
        self.greyscale = greyscale
        self.Images = []
        self.Labels = []

    def get_images(self, directory):
        for label in os.listdir(directory):
            label_number = cfg.Labels.label_mapping.get(label)

            for image_file in os.listdir(directory + label):
                image_path = os.path.join(directory, label, image_file)
                image = cv2.imread(image_path)
                if self.greyscale is True:
                    image = rgb2gray(image)

                image = cv2.resize(image, (100, 100))
                self.Images.append(image)
                self.Labels.append(label_number)

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

    def get_tensor_train(self, validation_split=0.2, batch_size=32, image_size=(100, 100)):
        tensor_train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)
        return tensor_train_ds

    def get_tensor_val(self, validation_split=0.2, batch_size=32, image_size=(100, 100)):
        tensor_val_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size)
        return tensor_val_ds

    def get_tensor_test(self, batch_size=32, image_size=(100, 100)):
        tensor_test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_directory,
            seed=123,
            image_size=image_size,
            batch_size=batch_size)
        return tensor_test_ds

    def plot_images(self, dataset, nrows=3, ncols=3):
        if self.greyscale:
            cmap = 'gray'
        else:
            cmap = None
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(nrows*ncols):
                px = images[i].numpy()
                if max(px[0][0]) > 1:
                    px = px.astype('uint8')
                plt.subplot(nrows, ncols, i + 1)
                plt.imshow(px, cmap=cmap)
                plt.axis("off")
            plt.show()
