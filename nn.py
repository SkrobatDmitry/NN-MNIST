import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.parameters = self.init_parameters(784, 128, 11)
        self.velocity = self.init_velocity()

    def forward(self, x):
        pass

    def backward(self, x, error):
        pass

    def init_parameters(self, n_in, n_h, n_out):
        return {'w1': self.init_weights(n_in, n_h), 'b1': self.init_biases(n_h),
                'w2': self.init_weights(n_h, n_out), 'b2': self.init_biases(n_out)}

    def init_velocity(self):
        return {'w1': np.zeros(self.parameters['w1'].shape), 'b1': np.zeros(self.parameters['b1'].shape),
                'w2': np.zeros(self.parameters['w2'].shape), 'b2': np.zeros(self.parameters['b2'].shape)}

    @staticmethod
    def init_weights(n_in, n_out):
        limit = np.sqrt(2 / (n_in - 1))
        return np.random.uniform(-limit, limit, (n_out, n_in))

    @staticmethod
    def init_biases(n_out):
        return np.zeros((n_out, 1))
