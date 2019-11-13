import numpy as np

from whatthefood.graph.node import Node


class Sum(Node):
    def __init__(self, x, y):
        assert x.shape == y.shape
        super(Sum, self).__init__(x.shape, x.batched or y.batched, x, y)

        self.batched_args = x.batched, y.batched

    def do(self, x, y):
        return x + y

    def backpropagate(self, grad, x, y):
        grad_x = np.sum(grad, axis=0) if not self.batched_args[0] and self.batched else grad
        grad_y = np.sum(grad, axis=0) if not self.batched_args[0] and self.batched else grad

        return grad_x, grad_y


class Reshape(Node):
    def __init__(self, x, shape):
        super(Reshape, self).__init__(shape, x.batched, x)

    def do(self, x):
        return x.reshape(self.shape if not self.batched else (x.shape[0], *self.shape))

    def backpropagate(self, grad, x):
        return grad.reshape(x.shape)


def flatten(x):
    return Reshape(x, shape=(-1,))


class MultiplyByScalar(Node):
    def __init__(self, x, scalar):
        assert scalar.shape == () and not scalar.batched
        super(MultiplyByScalar, self).__init__(x.shape, x.batched, x, scalar)

    def do(self, x, scalar):
        return x * scalar

    def backpropagate(self, grad, x, scalar):
        return grad * scalar, np.sum(grad * x)
