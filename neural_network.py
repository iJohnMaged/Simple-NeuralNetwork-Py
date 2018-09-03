from matrix import Matrix
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNetwork:

    def __init__(self, input_num, hidden_num, output_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        self.weights_ih = Matrix(self.hidden_num, self.input_num)
        self.weights_ho = Matrix(self.output_num, self.hidden_num)

        self.bias_h = Matrix(self.hidden_num, 1)
        self.bias_o = Matrix(self.output_num, 1)

        self.lr = 0.25

    def feed_forward(self, input):

        input_matrix = Matrix.from_list(input)

        hidden = Matrix.dot(self.weights_ih, input_matrix)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        outputs = Matrix.dot(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)
        return outputs

    def train(self, inputs, targets):

        # input_matrix = Matrix.from_list(inputs)
        input_matrix = Matrix.from_list(inputs)

        hidden = Matrix.dot(self.weights_ih, input_matrix)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        outputs = Matrix.dot(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)

        targets_matrix = Matrix.from_list(targets)
        output_error = Matrix.subtract(targets_matrix, outputs)
        weights_ho_t = Matrix.transpose(self.weights_ho)
        hidden_error = Matrix.dot(weights_ho_t, output_error)

        gradient = Matrix.mmap(outputs, dsigmoid)
        gradient.scale(self.lr)
        gradient.multiply(output_error)
        hidden_T = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.dot(gradient, hidden_T)

        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradient)

        hidden_gradient = Matrix.mmap(hidden, dsigmoid)
        hidden_gradient.scale(self.lr)
        hidden_gradient.multiply(hidden_error)
        input_T = Matrix.transpose(input_matrix)
        weights_ih_deltas = Matrix.dot(hidden_gradient, input_T)
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)


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

print(nn.feed_forward([0, 0]))
print(nn.feed_forward([0, 1]))
print(nn.feed_forward([1, 0]))
print(nn.feed_forward([1, 1]))
