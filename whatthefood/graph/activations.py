import numpy as np

from whatthefood.graph.node import Node


class ReLU(Node):
    def __init__(self, x, alpha=0.):
        super(ReLU, self).__init__(x.shape, x.batched, x)

        self.alpha = alpha

    def do(self, x):
        return (x >= 0) * x + (x < 0) * self.alpha * x

    def backpropagate(self, grad, x):
        return (x >= 0) * grad + (x < 0) * self.alpha * grad,


def leaky_relu(alpha=0.2):
    def builder(x):
        return ReLU(x, alpha)

    return builder


class Sigmoid(Node):
    def __init__(self, x):
        super(Sigmoid, self).__init__(x.shape, x.batched, x)

    def do(self, x):
        return 1 / (1 + np.exp(-x))

    def backpropagate(self, grad, x):
        v = self.do(x)
        return v * (1 - v) * grad,


class Softmax(Node):
    def __init__(self, x):
        assert len(x.shape) > 0
        super(Softmax, self).__init__(x.shape, x.batched, x)

    def do(self, x):
        e = np.exp(x)
        s = np.repeat(np.sum(e, axis=-1), e.shape[-1]).reshape(e.shape)
        s.resize(e.shape)
        return e / s

    def backpropagate(self, grad, x):
        e = np.exp(x)
        s = np.repeat(np.sum(e, axis=-1), e.shape[-1]).reshape(e.shape)

        eg = e * grad

        common_part = np.repeat(np.sum(eg, axis=-1), e.shape[-1]).reshape(e.shape)
        common_part.resize(e.shape)

        return (eg * s - e * common_part) / np.square(s),
