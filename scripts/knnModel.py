from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import config as cfg


class KNNModel:

    def __init__(self, k=3):
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

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                      display_labels=cfg.Labels.label_mapping.keys())
        disp.plot()
        plt.title("KNNModel")
        plt.show()

        print(classification_report(y_pred, y_test))
        print('Accuracy: {}%'.format(round(100 * accuracy, 2)))
