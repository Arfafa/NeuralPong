import numpy as np


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class Neuron:

    def __init__(self, weights_number, learning_rate):
        self.weights = np.random.normal(size=weights_number)
        self.bias = np.random.normal()

        self.learning_rate = learning_rate

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

    def update(self, dLdypred, inputs, sum_out):

        aux = self.learning_rate*dLdypred*deriv_sigmoid(sum_out)

        self.bias -= aux

        new_weights = []

        for weight, x in zip(self.weights, inputs):
            weight -= aux*x
            new_weights.append(weight)

        self.weights = np.array(new_weights)


class NeuralNetwork:

    def __init__(self, hidden_neurons_number, learning_rate):
        self.learning_rate = learning_rate

        self.hidden_neurons = []

        for i in range(hidden_neurons_number):
            self.hidden_neurons.append(Neuron(2, learning_rate))

        self.out_neuron = Neuron(hidden_neurons_number, learning_rate)

    def feedforward(self, x):
        # x is a numpy array with 2 elements.

        out_hidden_layer = []
        for neuron in self.hidden_neurons:
            out_hidden_layer.append(neuron.feedforward(x))

        out_hidden_layer = np.array(out_hidden_layer)
        resp = self.out_neuron.feedforward(out_hidden_layer)

        return resp

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        '''

        for x, y_true in zip(data, all_y_trues):

            totals = []
            out_neurons = []
            for neuron in self.hidden_neurons:
                total = np.dot(neuron.weights, x) + neuron.bias
                out_neuron = sigmoid(total)

                totals.append(total)
                out_neurons.append(out_neuron)

            totals = np.array(totals)
            out_neurons = np.array(out_neurons)

            total_out_layer = np.dot(self.out_neuron.weights, out_neurons)
            total_out_layer += self.out_neuron.bias
            out_out_layer = sigmoid(total_out_layer)

            y_pred = out_out_layer

            d_L_d_ypred = -2 * (y_true - y_pred)

            hidden_derivs = []

            for weight in self.out_neuron.weights:
                resp = weight*deriv_sigmoid(total_out_layer)
                hidden_derivs.append(resp)

            hidden_derivs = np.array(hidden_derivs)

            for neuron, total, hidden_deriv in zip(self.hidden_neurons,
                                                   totals,
                                                   hidden_derivs):

                neuron.update(d_L_d_ypred*hidden_deriv, x, total)

            self.out_neuron.update(d_L_d_ypred, out_neurons, total_out_layer)
