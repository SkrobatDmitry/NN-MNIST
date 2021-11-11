import numpy as np


class ReLU:
    def __init__(self): pass

    @staticmethod
    def activation(x):
        return np.maximum(0, x)

    @staticmethod
    def gradient(x):
        return x > 0


class Softmax:
    def __init__(self): pass

    @staticmethod
    def activation(x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def gradient(self, x):
        x = self.activation(x)
        return x * (1 - x)


class CrossEntropy:
    def __init__(self): pass

    @staticmethod
    def loss(y, o):
        o = np.clip(o, 1e-15, 1 - 1e-15)
        return -y * np.log(o) - (1 - y) * np.log(1 - o)

    @staticmethod
    def gradient(y, o):
        o = np.clip(o, 1e-15, 1 - 1e-15)
        return -(y / o) + (1 - y) / (1 - o)
