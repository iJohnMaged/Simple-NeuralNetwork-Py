import math


class ActivationFunction:
    # A simple class to hold an activation function and its derivative.
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc


def sig(x):
    return 1 / (1 + math.exp(-x))


def dsig(y):
    return y * (1 - y)


def tanh(x):
    return math.tanh(x)


def dtanh(y):
    return 1 - (y * y)


SIGMOID = ActivationFunction(sig, dsig)
TANH = ActivationFunction(tanh, dtanh)
