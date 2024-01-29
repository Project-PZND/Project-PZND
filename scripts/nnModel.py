import tensorflow as tf
from baseNNModel import BaseNNModel
import config as cfg


class NNModel(BaseNNModel):
    def __init__(self, input_shape, dense_layers, activations, epochs):
        super().__init__()
        self.dense_layers = dense_layers
        self.activations = activations
        self.epochs = epochs
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            *[tf.keras.layers.Dense(
                dense_layer, activation=activation) for dense_layer, activation in
                zip(self.dense_layers, self.dense_layers)]
        ])

