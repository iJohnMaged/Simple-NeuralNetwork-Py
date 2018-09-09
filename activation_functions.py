"""
This modules contains a few useful activation functions.

You can read more about it here: https://en.wikipedia.org/wiki/Activation_function#Comparison_of_activation_functions
"""

import math


class ActivationFunction:
    # A simple class to hold an activation function and its derivative.
    def __init__(self, func, dfunc, range=(0, 0), use_x_vals=False):
        self.func = func
        self.dfunc = dfunc
        self.range = range
        self.use_x_vals = use_x_vals


# This could be turned into lambda functions but I'm keeping it this way for the pickle module to work.

def sig(x):
    return 1 / (1 + math.exp(-x))


def dsig(y):
    return y * (1 - y)


def tanh(x):
    return math.tanh(x)


def dtanh(y):
    return 1 - (y * y)


def arctan(x):
    return math.atan(x)


def darctan(y):
    return 1 / (y**2 + 1)


def softsign(x):
    return x / (1 + math.fabs(x))


def dsoftsign(y):
    return 1 / (math.pow((math.fabs(y) + 1), 2))


def relu(x):
    return max(0, x)


def drelu(y):
    return 0 if y < 0 else 1


def leaky_relu(x):
    return 0.01 * x if x < 0 else x


def dleaky_relu(y):
    return 0.01 if y < 0 else 1


def softplus(x):
    return math.log(1 + math.exp(x))


def dsoftplus(y):
    return 1 / (1 + math.exp(-y))


def gaussian(x):
    return math.exp(-1 * (x * x))


def dgaussian(y):
    return -2 * y * math.exp(-1 * (y * y))


SIGMOID = ActivationFunction(sig, dsig, (0, 1))
TANH = ActivationFunction(tanh, dtanh)
ARCTAN = ActivationFunction(arctan, darctan, range=(-math.pi/2, math.pi/2), use_x_vals=True)
SOFTSIGN = ActivationFunction(softsign, dsoftsign, range=(-1, 1), use_x_vals=True)
RELU = ActivationFunction(relu, drelu, use_x_vals=True)
LEAKY_RELU = ActivationFunction(leaky_relu, dleaky_relu, use_x_vals=True)
SOFTPLUS = ActivationFunction(softplus, dsoftplus, use_x_vals=True)
GAUSSIAN = ActivationFunction(gaussian, dgaussian, range=(0, 1), use_x_vals=True)

A_FUNCTIONS = {
    'sigmoid': SIGMOID,
    'tanh': TANH,
    'arctan': ARCTAN,
    'softsign': SOFTSIGN,
    'relu': RELU,
    'leaky_relu': LEAKY_RELU,
    'softplus': SOFTPLUS,
    'gaussian': GAUSSIAN
}
