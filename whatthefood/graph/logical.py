from whatthefood.graph.node import Node

import numpy as np


class GT(Node):
    def __init__(self, x, y):
        assert y.shape == () or y.shape == x.shape
        assert x.batched == y.batched
        super(GT, self).__init__(x.shape, x.batched, x, y)

    def do(self, x, y):
        return x > y

    def backpropagate(self, grad, x, y):
        return np.zeros_like(x), np.zeros_like(y)


class Equal(Node):
    def __init__(self, x, y):
        assert y.shape == () or y.shape == x.shape
        assert x.batched == y.batched
        super(Equal, self).__init__(x.shape, x.batched, x, y)

    def do(self, x, y):
        return x == y

    def backpropagate(self, grad, x, y):
        return np.zeros_like(x), np.zeros_like(y)


class ArgMax(Node):
    def __init__(self, x):
        assert len(x.shape) > 0
        super(ArgMax, self).__init__(x.shape[:-1], x.batched, x)

    def do(self, x):
        return np.argmax(x, axis=-1)

    def backpropagate(self, grad, x):
        return np.zeros_like(x)
