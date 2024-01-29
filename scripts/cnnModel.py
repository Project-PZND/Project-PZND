import tensorflow as tf
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


class CNNModel(tf.keras.Sequential):

    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

    def test_model(self, train_data, test_data, validation_data=None):
        x_train = train_data[0]
        y_train = train_data[1]
        x_test = test_data[0]
        y_test = test_data[1]

        self.model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        self.model.fit(
            x=x_train,
            y=y_train,
            validation_data=validation_data,
            epochs=10
        )

        # history_cnn_1 = self.model.fit(
        #     train_ds,
        #     validation_data=val_ds,
        #     epochs=20,
        #     verbose=0
        # )

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
