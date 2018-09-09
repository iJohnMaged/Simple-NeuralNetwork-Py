import numpy as np
from activation_functions import SIGMOID
import pickle


class NeuralNetwork:

    def __init__(self, input_num, hidden_num, output_num, activation_func = SIGMOID, lr = 0.1):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        self.weights_ih = None
        self.weights_ho = None
        self.bias_o = None
        self.bias_h = None

        self.reset()

        self._lr = lr
        self.set_activation_func(activation_func)

    def reset(self):
        self.weights_ih = np.random.uniform(-1, 1, (self.hidden_num, self.input_num))
        self.weights_ho = np.random.uniform(-1, 1, (self.output_num, self.hidden_num))
        self.bias_h = np.random.uniform(-1, 1, (self.hidden_num, 1))
        self.bias_o = np.random.uniform(-1, 1, (self.output_num, 1))

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    def set_activation_func(self, func):
        self.activation_func = func
        self.npfunc = np.vectorize(self.activation_func.func)
        self.npdfunc = np.vectorize(self.activation_func.dfunc)

    def predict(self, input, get_max = False):

        if type(input) is list:
            input_matrix = np.array(input).reshape((len(input), 1))
        elif type(input) is np.ndarray:
            input_matrix = input
        else:
            raise ValueError('Input should be a list or np array')

        hidden = np.dot(self.weights_ih, input_matrix)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.npfunc(hidden)

        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.npfunc(output)

        if get_max:
            return output, output.argmax()

        return output

    def train(self, inputs, targets):

        if type(inputs) is list:
            input_matrix = np.array(inputs).reshape((len(inputs), 1))
        elif type(inputs) is np.ndarray:
            input_matrix = inputs
        else:
            raise ValueError('Input should be a list or np array')

        # Generate the outputs of the hidden nodes
        hidden = np.dot(self.weights_ih, input_matrix)
        hidden = np.add(hidden, self.bias_h)

        hidden_copy = None
        if self.activation_func.use_x_vals:
            hidden_copy = hidden.copy()

        hidden = self.npfunc(hidden)

        # Generate the outputs
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)

        output_copy = None
        if self.activation_func.use_x_vals:
            output_copy = output.copy()

        output = self.npfunc(output)

        targets_matrix = np.array(targets).reshape(len(targets), 1)

        # Calculate the error
        # E = TARGETS - OUTPUTS
        output_error = np.subtract(targets_matrix, output)

        if self.activation_func.use_x_vals:
            output_gradient = self.npdfunc(output_copy)
        else:
            output_gradient = self.npdfunc(output)

        # Calculate output gradient
        output_gradient = np.multiply(self._lr, output_gradient)
        output_gradient = np.multiply(output_error, output_gradient)
        hidden_T = hidden.T

        # Calculate the deltas for hidden-output layer
        weights_ho_deltas = np.dot(output_gradient, hidden_T)

        # Adjust the weights
        self.weights_ho = np.add(self.weights_ho, weights_ho_deltas)
        self.bias_o = np.add(self.bias_o, output_gradient)

        # Hidden layer error
        weights_ho_T = self.weights_ho.T
        hidden_error = np.dot(weights_ho_T, output_error)

        # Calculate hidden layer gradient

        if self.activation_func.use_x_vals:
            hidden_gradient = self.npdfunc(hidden_copy)
        else:
            hidden_gradient = self.npdfunc(hidden)

        hidden_gradient = np.multiply(hidden_gradient, self._lr)
        hidden_gradient = np.multiply(hidden_gradient, hidden_error)

        input_T = input_matrix.T

        weights_ih_deltas = np.dot(hidden_gradient, input_T)

        # Adjust the weights
        self.weights_ih = np.add(self.weights_ih, weights_ih_deltas)
        self.bias_h = np.add(self.bias_h, hidden_gradient)


    def mutate(self, func):
        npfunc = np.vectorize(func)
        self.weights_ih = npfunc(self.weights_ih)
        self.weights_ho = npfunc(self.weights_ho)
        self.bias_h = npfunc(self.bias_h)
        self.bias_o = npfunc(self.bias_o)

    def copy(self):
        nn = NeuralNetwork(self.input_num, self.hidden_num, self.output_num, self.activation_func)
        nn.weights_ho = self.weights_ho.copy()
        nn.weights_ih = self.weights_ih.copy()
        nn.bias_o = self.bias_o.copy()
        nn.bias_h = self.bias_h.copy()
        nn.lr = self._lr
        return nn

    def serialize(self):
        return pickle.dumps(self)

    def save_to_file(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'rb') as f:
            nn = pickle.load(f)
            return nn

    @staticmethod
    def deserialize(data):
        nn = pickle.loads(data)
        return nn
