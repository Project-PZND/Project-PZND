from imagePreprocessor import ImageDataPreprocessor
from imageLoader import ImageLoader
from cnnModel import CNNModel
from knnModel import KNNModel
from nnModel import NNModel

images_loader = ImageLoader()

x_test, y_test = images_loader.get_test_data()
x_train, y_train = images_loader.get_train_data()

train_tensor = images_loader.get_tensor_train()
test_tensor = images_loader.get_tensor_test()
val_tensor = images_loader.get_tensor_val()

cnn = CNNModel()
nn = NNModel()
knn = KNNModel()

cnn.train_model((x_train, y_train), validation_data=(x_test, y_test), epochs=10)
nn.train_model((x_train, y_train), validation_data=(x_test, y_test), epochs=10)

knn.evaluate(train_tensor, test_tensor)

cnn.test_model((x_test, y_test))
nn.test_model((x_test, y_test))
