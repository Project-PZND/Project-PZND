import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config as cfg


class NNModel:
    def __init__(self, input_shape, dense_layers, activations, epochs):
        self.dense_layers = dense_layers
        self.activations = activations
        self.epochs = epochs
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            *[tf.keras.layers.Dense(
                dense_layer, activation=activation) for dense_layer, activation in
                zip(self.dense_layers, self.dense_layers)]
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def fit_model(self, train_data, validation_data=None):

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
            epochs=self.epochs,
        )

    def test_model(self, test_data):
        x_test = test_data[0]
        y_test = test_data[1]


    # def evaluate(self):
    #     score, accuracy = self.model.evaluate(
    #         self.test_ds,
    #         batch_size=self.batch_size,
    #         verbose=0
    #     )
    #     print('Loss: {}'.format(score))
    #     print('Accuracy: {}%'.format(round(100 * accuracy, 2)))

    def plot_loss_curve(self):
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Test')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def confusion_matrix(self, test_tensor):
        y_predicted_probabilities = self.model.predict(test_tensor)
        y_predicted_classes = np.array(list(map(lambda results: np.argmax(results), y_predicted_probabilities)))
        y_true_classes = np.concatenate([y for x, y in test_tensor], axis=0)

        # print('Accuracy: {}%'.format(round(100 * accuracy_score(test_labels, predict_test_cnn_1), 2)))

        cf_matrix = confusion_matrix(y_true_classes, y_predicted_classes)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix,
            display_labels=cfg.Labels.label_mapping.keys()
        )
        disp.plot()
        plt.show()
