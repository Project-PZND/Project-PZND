import tensorflow as tf
from baseNNModel import BaseNNModel


class NNModel(BaseNNModel):
    """
        A neural network model for image classification, built on top of the BaseNNModel.

        All parameters can be set in config.py.

        Attributes:
        - dense_layers (tuple): Tuple specifying the number of neurons in each dense layer. Default is (128, 64, 6).
        - activations (tuple): Tuple specifying the activation function for each dense layer. Default is ('relu', 'relu', 'softmax').
        - input_shape (tuple): Tuple specifying the input shape of the images. Default is (150, 150, 3)
        - model: The Sequential model defining the neural network architecture.

        Methods:
        - train_model(train_data, validation_data): Trains the neural network model on the provided training data.
        - evaluate_model(test_data): Evaluates the neural network model on the provided test data.
        - save_model(model_path): Saves the trained model to the specified file path.
    """
    def __init__(self, dense_layers=(128, 64, 6), activations=('relu', 'relu', 'softmax'), input_shape=(150, 150, 3)):
        super().__init__()
        self.dense_layers = dense_layers
        self.activations = activations
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            *[tf.keras.layers.Dense(
                dense_layer, activation=activation) for dense_layer, activation in
                zip(self.dense_layers, self.activations)]
        ])
