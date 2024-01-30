import matplotlib.pyplot as plt
from imageLoader import ImageLoader
from imagePreprocessor import ImageDataPreprocessor
from cnnModel import CNNModel
from knnModel import KNNModel
from nnModel import NNModel

image_loader = ImageLoader()

image_processor = ImageDataPreprocessor(target_size=(150, 150), augmentation=False)

x_test, y_test = image_loader.get_test_data()
x_train, y_train = image_loader.get_train_data()

image_loader.plot_images((x_train, y_train))

train_tensor = image_loader.get_tensor_train()
test_tensor = image_loader.get_tensor_test()
val_tensor = image_loader.get_tensor_val()

cnn = CNNModel()
nn = NNModel()
knn = KNNModel()

cnn.train_model((x_train, y_train), validation_data=(x_test, y_test), epochs=5)
nn.train_model((x_train, y_train), validation_data=(x_test, y_test), epochs=5)

knn.evaluate(train_tensor, test_tensor)

cnn.test_model((x_test, y_test))
nn.test_model((x_test, y_test))

plt.show()
