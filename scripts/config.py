class Images:
    train_directory = r"../data/train_images/"
    test_directory = r"../data/test_images/"


class Labels:
    label_mapping = {
        'buildings': 0,
        'forest': 1,
        'glacier': 2,
        'mountain': 3,
        'sea': 4,
        'street': 5
    }


# Default params for image augmentation
class AugmentImageParams:
    rotate = 0.10
    zoom = (.15, .1)
    flip = "horizontal"
    contrast = 0.1
    brightness = 0.0001


# Default size of loaded image (base images are 150x150 pixels)
class ImageLoadSize:
    height = 100
    width = 100
