import numpy as np
from functions import ReLU, Softmax


class NeuralNetwork:
    def __init__(self):
        self.parameters = self.init_parameters(784, 128, 11)
        self.velocity = self.init_velocity()
        self.cache = {}

    def forward(self, x):
        z1 = self.linear_function(self.parameters['w1'], x, self.parameters['b1'])
        h = ReLU.activation(z1)

        z2 = self.linear_function(self.parameters['w2'], h, self.parameters['b2'])
        o = Softmax.activation(z2)

        self.cache = {'z1': z1, 'h': h, 'z2': z2, 'o': o}
        return o

    def backward(self, x, error, mu=.9, learning_rate=1e-3):
        db2 = Softmax().gradient(self.cache['z2']) * error
        dw2 = np.dot(self.cache['h'], db2.T)

        db1 = ReLU.gradient(self.cache['z1']) * np.dot(db2.T, self.parameters['w2']).T
        dw1 = np.dot(x, db1.T)

        # Update parameters
        velocity_prev = self.velocity

        self.velocity['w1'] = mu * self.velocity['w1'] - learning_rate * dw1
        self.velocity['b1'] = mu * self.velocity['b1'] - learning_rate * np.mean(db1)
        self.velocity['w2'] = mu * self.velocity['w2'] - learning_rate * dw2
        self.velocity['b2'] = mu * self.velocity['b2'] - learning_rate * np.mean(db2)

        self.parameters['w1'] += -mu * velocity_prev['w1'] + (1 + mu) * self.velocity['w1']
        self.parameters['b1'] += -mu * velocity_prev['b1'] + (1 + mu) * self.velocity['b1']
        self.parameters['w2'] += -mu * velocity_prev['w2'] + (1 + mu) * self.velocity['w2']
        self.parameters['b2'] += -mu * velocity_prev['b2'] + (1 + mu) * self.velocity['b2']

    @staticmethod
    def linear_function(w, x, b):
        return np.dot(w, x) + b

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
