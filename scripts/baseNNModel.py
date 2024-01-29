import tensorflow as tf
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class BaseNNModel:

    def __init__(self):
        self.model = None

    def train_model(self, train_data, validation_data=None, epochs=10):
        x_train = train_data[0]
        y_train = train_data[1]

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            epochs=epochs
        )

    def test_model(self, test_data):
        x_test = test_data[0]
        y_test = test_data[1]

        y_predicted_probabilities = self.model.predict(x_test)
        y_predicted = np.array(list(map(lambda results: np.argmax(results), y_predicted_probabilities)))

        print('Accuracy: {}%'.format(round(100 * accuracy_score(y_test, y_predicted), 2)))

        cf_matrix = confusion_matrix(y_test, y_predicted)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix,
            display_labels=cfg.Labels.label_mapping.keys()
        )
        disp.plot()
        plt.show()