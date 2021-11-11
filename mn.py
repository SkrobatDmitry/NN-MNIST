import numpy as np
from mnist import MNIST


class DataLoader:
    def __init__(self):
        self.mn = MNIST('dataset', gz=True)

    def get_training(self):
        images, labels = self.mn.load_training()
        return self.process_data(images, labels)

    def get_testing(self):
        images, labels = self.mn.load_testing()
        return self.process_data(images, labels)

    @staticmethod
    def process_data(x, y):
        x = np.array(x) / 255.
        y = np.eye(11)[y]
        return x, y
