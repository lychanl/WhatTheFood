import numpy as np

from whatthefood.graph.node import Node


class Matmul(Node):
    def __init__(self, x, y):
        assert x.batched and len(x.shape) == 1 or not x.batched and len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape[-1] == y.shape[0]

        super(Matmul, self).__init__(
            (x.shape[0], y.shape[1]) if not x.batched else (y.shape[1]),
            x.batched,
            x, y
        )

    def do(self, x, y):
        return np.matmul(x, y)

    def backpropagate(self, grad, x, y):
        return np.matmul(grad, y.T), np.matmul(x.T, grad)
