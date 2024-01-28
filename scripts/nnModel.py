import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import config as cfg


class NNModel:
    def __init__(self, input_shape, dense_layer, activations,
                 train_ds, test_ds, val_ds, epochs, batch_size):
        self.dense_layer = dense_layer
        self.activations = activations
        self.train_ds = train_ds,
        self.test_ds = test_ds,
        self.val_ds = val_ds,
        self.epochs = epochs,
        self.batch_size = batch_size,
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            [tf.keras.layers.Dense(self.dense_layer[i],
                                   activation=self.activations[i]) for i in range(len(self.dense_layer))]])

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def train(self):
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            verbose=0
        )

    def evaluate(self):
        score, accuracy = self.model.evaluate(
            self.test_ds,
            batch_size=self.batch_size,
            verbose=0
        )
        print('Loss: {}'.format(score))
        print('Accuracy: {}%'.format(round(100 * accuracy, 2)))

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
        print(y_predicted_classes)
        y_true_classes = np.concatenate([y for x, y in test_tensor], axis=0)
        print(y_true_classes)

        # print('Accuracy: {}%'.format(round(100 * accuracy_score(test_labels, predict_test_cnn_1), 2)))

        cf_matrix = confusion_matrix(y_true_classes, y_predicted_classes)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cf_matrix,
            display_labels=cfg.Labels.label_mapping.keys()
        )
        disp.plot()
        plt.show()
