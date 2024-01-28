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

    @staticmethod
    def get_color_mode(grayscale):
        return "grayscale" if grayscale else "rgb"

    def get_images(self, directory):
        for label in os.listdir(directory):
            label_number = cfg.Labels.label_mapping.get(label)

            for image_file in os.listdir(directory + label):
                image_path = os.path.join(directory, label, image_file)
                image = cv2.imread(image_path)

                if self.greyscale is True:
                    image = rgb2gray(image)

                image = cv2.resize(image, (cfg.ImageLoadSize.width, cfg.ImageLoadSize.height))
                self.Images.append(image)
                self.Labels.append(label_number)

        return shuffle(self.Images, self.Labels, random_state=817328462)

    def _get_data(self, directory):
        images, labels = self.get_images(directory)
        return np.array(images) / 255, np.array(labels)

    def get_test_data(self):
        return self._get_data(self.test_directory)

    def get_train_data(self):
        return self._get_data(self.train_directory)

    def get_tensor_train(self, validation_split=0.2, batch_size=32, image_size=(100, 100)):
        tensor_train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=self.get_color_mode(self.greyscale)
        )
        return tensor_train_ds

    def get_tensor_val(self, validation_split=0.2, batch_size=32, image_size=(100, 100)):
        tensor_val_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=self.get_color_mode(self.greyscale)
        )
        return tensor_val_ds

    def get_tensor_test(self, batch_size=32, image_size=(100, 100)):
        tensor_test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_directory,
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=self.get_color_mode(self.greyscale)
        )
        return tensor_test_ds

    def plot_images(self, dataset, num_of_rows=3, num_of_cols=3):
        class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        if self.greyscale:
            cmap = 'gray'
        else:
            cmap = None
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(num_of_rows * num_of_cols):
                px = images[i].numpy()
                if max(px[0][0]) > 1:
                    px = px.astype('uint8')
                plt.subplot(num_of_rows, num_of_cols, i + 1)
                plt.imshow(px, cmap=cmap)
                plt.title(class_names[labels[i]])
                plt.axis("off")
            plt.show()
