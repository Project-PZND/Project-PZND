import tensorflow as tf
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class BaseNNModel:
    """
        A base class for neural network models.

        All parameters can be set in config.py.

        Methods:
        - train_model(train_data, validation_data, epochs): Trains the neural network model on training data.
                      default values: validation_data = None, epochs = 10
        - test_model(test_data): Evaluates the neural network model on test data, displaying a confusion matrix and metrics.
    """
    def __init__(self):
        self.model = None
        self.history = None

    def train_model(self, train_data, validation_data=None, epochs=10):

        print(f"\nFitting {self.__class__.__name__}...")

        x_train = train_data[0]
        y_train = train_data[1]

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            epochs=epochs
        )

        plt.figure()

        if validation_data:
            plt.plot(self.history.history['loss'], label='Train')
            plt.plot(self.history.history['val_loss'], label='Test')
            plt.title(f'{self.__class__.__name__} loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['Training loss', 'Validation loss'], loc='upper left')

        else:
            plt.plot(self.history.history['loss'], label='Train')
            plt.title(f'{self.__class__.__name__} loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['Training loss'], loc='upper left')

    def test_model(self, test_data):
        x_test = test_data[0]
        y_test = test_data[1]

        print('\n---------------------------')

        y_predicted_probabilities = self.model.predict(x_test)
        y_predicted = np.array(list(map(lambda results: np.argmax(results), y_predicted_probabilities)))

        cf_matrix = confusion_matrix(y_test, y_predicted)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix,
            display_labels=cfg.Labels.label_mapping.keys()
        )
        disp.plot()
        plt.title(str(self.__class__.__name__))

        score, accuracy = self.model.evaluate(
            x=x_test,
            y=y_test,
            verbose=0
        )

        print(f'{self.__class__.__name__}:')
        print('Loss: {}'.format(score))
        print('Accuracy: {}%'.format(round(100 * accuracy, 2)))
