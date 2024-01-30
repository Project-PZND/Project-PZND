from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import config as cfg


class KNNModel:
    """
        A class for k-nearest neighbors (KNN) classifier model.

        All parameters can be set in config.py

        Attributes:
        - k (int): The number of neighbors to consider (10 by default).
        - model: The KNeighborsClassifier model with specified k value.

        Methods:
        - flatten(tensor_dataset): Flattens a TensorFlow dataset of images and labels into 1D arrays.
        - evaluate(train, test): Evaluates the KNN model on training and test datasets, displaying confusion matrix and metrics.
        """
    def __init__(self, k=10):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    @staticmethod
    def flatten(tensor_dataset):
        images, labels = tuple(zip(*tensor_dataset.unbatch()))
        images = np.array(images)
        labels = np.array(labels)
        n_samples, nx, ny, n_rgb = images.shape
        flat = images.reshape(n_samples, nx * ny * n_rgb)
        return flat, labels

    def evaluate(self, train, test):
        x_train, y_train = self.flatten(train)
        x_test, y_test = self.flatten(test)

        print('\nFitting KNNModel...')

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                      display_labels=cfg.Labels.label_mapping.keys())
        disp.plot()
        plt.title("KNNModel")

        print('\n---------------------------')
        print(f'{self.__class__.__name__}:')
        print('Accuracy: {}%'.format(round(100 * accuracy, 2)))
