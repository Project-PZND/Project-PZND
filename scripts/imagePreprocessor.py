import tensorflow as tf
import config as cfg


class ImageDataPreprocessor:
    def __init__(self, target_size=(100, 100), augmentation=True):
        self.target_size = target_size
        self.augmentation = augmentation

    @staticmethod
    def _augment_image():
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

    def _resize_image(self):
        resize = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1])
        return resize

    def preprocess(self, data):

        preprocess = tf.keras.Sequential()

        # Resize image
        if self.target_size:
            preprocess.add(self._resize_image())

        # Augment image
        if self.augmentation:
            preprocess.add(self._augment_image())

        preprocessed_data = None
        if isinstance(data, tf.data.Dataset):
            preprocessed_data = data.map(lambda x, y: (preprocess(x, training=True), y))
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            preprocessed_data = (preprocess(data[0]), data[1])

        print('Processing data..')

        return preprocessed_data
