import tensorflow as tf

class ImageDataPreprocessor:
    def __init__(self, target_size=(100, 100), normalize=True, augmentation=True):
        self.target_size = target_size
        self.normalize = normalize
        self.augmentation = augmentation

    def resize_image(self):
        resize = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1])
        return resize

    def normalize_image(self):
        normalize = tf.keras.layers.Rescaling(1./255)
        return normalize

    def augment_image(self, rotate=0.15, zoom=(.2, .1), flip="horizontal", contrast=0.3,
                      brightness=0.001):
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

    def preprocess(self, dataset):
        preprocess = tf.keras.Sequential()

        # Resize image
        if self.target_size:
            resizing = tf.keras.layers.Resizing(self.target_size[0], self.target_size[1])
            preprocess.add(resizing)

        # Normalize image
        if self.normalize:
            normalizing = tf.keras.layers.Rescaling(1./255)
            preprocess.add(normalizing)

        # Augment image
        if self.augmentation:
            augment = self.augment_image()
            preprocess.add(augment)

        dataset = dataset.map(lambda x, y: (preprocess(x, training=True), y))
        return dataset


