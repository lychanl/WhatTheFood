import numpy as np

from whatthefood.graph.node import Node


class MaxPooling2d(Node):
    def __init__(self, x, window_size, step=None):
        assert len(x.shape) == 3
        if step is None:
            step = window_size
        super(MaxPooling2d, self).__init__(
            tuple((s - window_size) // step + 1 for s in x.shape[:-1]) + (x.shape[-1],),
            x.batched,
            x
        )
        self.window_size = window_size
        self.step = step

    def do(self, x):
        out = np.ndarray(self.shape if not self.batched else (x.shape[0],) + self.shape)

        for i in range(0, x.shape[-3] - self.window_size + 1, self.step):
            for j in range(0, x.shape[-2] - self.window_size + 1, self.step):
                out_slice = (slice(None), i // self.step, j // self.step, slice(None))\
                    if self.batched else (i // self.step, j // self.step, slice(None))
                in_slice = (slice(None), slice(i, i + self.window_size), slice(j, j + self.window_size), slice(None))\
                    if self.batched else (slice(i, i + self.window_size), slice(j, j + self.window_size), slice(None))
                out[out_slice] = np.max(x[in_slice], axis=(-3, -2))

        return out

    def backpropagate(self, grad, x):
        y = self.do(x)
        out = np.zeros(x.shape)

        for i in range(0, x.shape[-3] - self.window_size + 1, self.step):
            for j in range(0, x.shape[-2] - self.window_size + 1, self.step):
                out_slice = (slice(None), i // self.step, j // self.step, slice(None))\
                    if self.batched else (i // self.step, j // self.step, slice(None))
                in_slice = (slice(None), slice(i, i + self.window_size), slice(j, j + self.window_size), slice(None))\
                    if self.batched else (slice(i, i + self.window_size), slice(j, j + self.window_size), slice(None))
                out[in_slice] += (y[out_slice] == x[in_slice]) * grad[out_slice]

        return out,
