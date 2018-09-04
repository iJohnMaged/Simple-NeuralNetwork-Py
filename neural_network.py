from matrix import Matrix
import math
import random


class ActivationFunction:
    # A simple class to hold an activation function and its derivative.
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc


sigmoid = ActivationFunction(lambda x: 1 / (1 + math.exp(-x)), lambda y: y * (1 - y))
tanh = ActivationFunction(lambda x: math.tanh(x), lambda y: 1 - (y * y))


class NeuralNetwork:

    def __init__(self, input_num, hidden_num, output_num, activation_func = sigmoid):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        self.weights_ih = Matrix(self.hidden_num, self.input_num)
        self.weights_ho = Matrix(self.output_num, self.hidden_num)

        self.bias_h = Matrix(self.hidden_num, 1)
        self.bias_o = Matrix(self.output_num, 1)

        self._lr = 0.1
        self.activation_func = activation_func

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        self._lr = lr

    def set_activation_func(self, func):
        self.activation_func = func

    def predict(self, input):

        input_matrix = Matrix.from_list(input)

        hidden = Matrix.dot(self.weights_ih, input_matrix)
        hidden.add(self.bias_h)
        hidden.map(self.activation_func.func)

        output = Matrix.dot(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(self.activation_func.func)

        return output.to_list()

    def train(self, inputs, targets):

        # Generate the outputs of the hidden nodes.
        input_matrix = Matrix.from_list(inputs)

        hidden = Matrix.dot(self.weights_ih, input_matrix)
        hidden.add(self.bias_h)
        hidden.map(self.activation_func.func)

        # Generate the outputs of the outputs nodes
        outputs = Matrix.dot(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(self.activation_func.func)

        targets_matrix = Matrix.from_list(targets)

        # Calculate the error following the formula: E = targets - outputs
        output_error = Matrix.subtract(targets_matrix, outputs)

        # Calculate gradient
        gradient = Matrix.mmap(outputs, self.activation_func.dfunc)
        gradient.scale(self._lr)
        gradient.multiply(output_error)
        hidden_T = Matrix.transpose(hidden)
        # Calculate deltas for hidden-output connection
        weights_ho_deltas = Matrix.dot(gradient, hidden_T)

        # Adjust the weights and bias
        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradient)

        # Hidden layer error
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_error = Matrix.dot(weights_ho_t, output_error)

        # Calculate hidden layer gradient
        hidden_gradient = Matrix.mmap(hidden, self.activation_func.dfunc)
        hidden_gradient.scale(self._lr)
        hidden_gradient.multiply(hidden_error)
        input_T = Matrix.transpose(input_matrix)
        weights_ih_deltas = Matrix.dot(hidden_gradient, input_T)

        # Adjust the weights and bias for input-hidden connection
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)

    def mutate(self, func):
        # Accepts an arbitrary function to mutate the neural network
        self.weights_ih.map(func)
        self.weights_ho.map(func)
        self.bias_h.map(func)
        self.bias_o.map(func)

    def copy(self):
        nn = NeuralNetwork(self.input_num, self.hidden_num, self.output_num, self.activation_func)
        nn.weights_ho = self.weights_ho.copy()
        nn.weights_ih = self.weights_ih.copy()
        nn.bias_o = self.bias_o.copy()
        nn.bias_h = self.bias_h.copy()
        nn.lr = self._lr
        return nn


def main():
    training_data = [{
        'input': [0, 0],
        'target': [0]
        },
        {
        'input': [1, 0],
        'target': [1]
        },
        {
        'input': [0, 1],
        'target': [1]
        },
        {
        'input': [1, 1],
        'target': [0]
    }]

    nn = NeuralNetwork(2, 2, 1)

    for i in range(1000000):
        data = random.choice(training_data)
        nn.train(data['input'], data['target'])
        # for data in training_data:
        #     nn.train(data['input'], data['target'])
        # random.shuffle(training_data)

    print(nn.predict([0, 0]))
    print(nn.predict([0, 1]))
    print(nn.predict([1, 0]))
    print(nn.predict([1, 1]))

if __name__ == '__main__':
    main()