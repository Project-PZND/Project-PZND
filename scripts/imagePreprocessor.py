import tensorflow as tf
import config as cfg


class ImageDataPreprocessor:
    def __init__(self, target_size=(100, 100), normalize=True, augmentation=True):
        self.target_size = target_size
        self.normalize = normalize
        self.augmentation = augmentation

    @staticmethod
    def __normalize_image():
        normalize = tf.keras.layers.Rescaling(1. / 255)
        return normalize

    @staticmethod
    def __augment_image():
        rotate = cfg.AugmentImageParams.rotate
        zoom = cfg.AugmentImageParams.zoom
        flip = cfg.AugmentImageParams.flip
        contrast = cfg.AugmentImageParams.contrast
        brightness = cfg.AugmentImageParams.brightness

        augment = tf.keras.Sequential()

        if rotate:
            augment.add(tf.keras.layers.RandomRotation(rotate))
        if zoom:
            augment.add(tf.keras.layers.RandomZoom(height_factor=zoom, fill_mode='nearest'))
        if flip:
            augment.add(tf.keras.layers.RandomFlip(flip))
        if contrast:
            augment.add(tf.keras.layers.RandomContrast(contrast))
        if brightness:
            augment.add(tf.keras.layers.RandomBrightness(brightness))

        return augment

    def __resize_image(self):
        resize = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1])
        return resize

    def preprocess(self, dataset):
        preprocess = tf.keras.Sequential()

        # Resize image
        if self.target_size:
            preprocess.add(self.__resize_image())

        # Normalize image
        if self.normalize:
            preprocess.add(self.__normalize_image())

        # Augment image
        if self.augmentation:
            preprocess.add(self.__augment_image())

        dataset = dataset.map(lambda x, y: (preprocess(x, training=True), y))
        return dataset
