from whatthefood.graph.node import Node
from whatthefood.ext import conv as conv_impl

import numpy as np


class Convolution(Node):
    def __init__(self, x, filters, step=1):
        assert len(x.shape) == 3 and len(filters.shape) == 4
        assert filters.shape[2] == x.shape[2]
        assert not filters.batched
        shape = (
            (x.shape[0] - filters.shape[0]) // step + 1,
            (x.shape[1] - filters.shape[1]) // step + 1,
            filters.shape[3]
        )
        super(Convolution, self).__init__(shape, x.batched, x, filters)

        self.step = step

    def do(self, x, filters):
        if x.dtype != np.float32:
            x = np.cast[np.float32](x)
        if filters.dtype != np.float32:
            filters = np.cast[np.float32](filters)
        if not self.batched:
            x = np.array([x], np.float32)

        out = conv_impl.conv2d(x, filters, self.step)
        return out if self.batched else out[0]

    def backpropagate(self, grad, x, filters):

        if x.dtype != np.float32:
            x = np.cast[np.float32](x)
        if filters.dtype != np.float32:
            filters = np.cast[np.float32](filters)
        if grad.dtype != np.float32:
            grad = np.cast[np.float32](grad)
        if not self.batched:
            x = np.array([x], np.float32)
            grad = np.array([grad], np.float32)

        grad_x, grad_filters = conv_impl.conv2d_grad(x, filters, grad, self.step)
        return grad_x if self.batched else grad_x[0], grad_filters
