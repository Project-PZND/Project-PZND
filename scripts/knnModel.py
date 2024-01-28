from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from imageLoader import ImageLoader
import matplotlib.pyplot as plt


class KNNModel(ImageLoader):

    def __init__(self, k=3):
        super().__init__()
        self.train_data = ImageLoader().get_train_data()
        self.test_data = ImageLoader().get_test_data()
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def flatten(self):
        train_images = self.train_data[0]
        test_images = self.test_data[0]

        train_nsamples, train_nx, train_ny, train_nrgb = train_images.shape
        test_nsamples, test_nx, test_ny, test_nrgb = test_images.shape

        train_flat = train_images.reshape(train_nsamples, train_nx*train_ny*train_nrgb)
        test_flat = test_images.reshape(test_nsamples, test_nx*test_ny*test_nrgb)
        return train_flat, test_flat

    def evaluate(self):
        x_train = self.flatten()[0]
        y_train = self.train_data[1]
        x_test = self.flatten()[1]
        y_test = self.test_data[1]

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1, 2, 3, 4, 5])

        print(classification_report(y_pred, y_test))
        disp.plot()
        plt.show()
        return accuracy
