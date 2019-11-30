from whatthefood.graph.node import Node

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
        out = np.ndarray(self.shape if not self.batched else (x.shape[0],) + self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x_slice = x[self._get_x_slice(i, j, filters)]
                x_slice = self._reshape_in_to_full(x_slice)
                if not self.batched:
                    out[i, j, :] = np.sum(x_slice * filters, axis=(0, 1, 2))
                else:
                    out[:, i, j, :] = np.sum(x_slice * filters, axis=(1, 2, 3))

        return out

    def backpropagate(self, grad, x, filters):
        grad_x = np.zeros_like(x, dtype=np.float32)
        grad_filters = np.zeros_like(filters, dtype=np.float32)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x_slice = self._get_x_slice(i, j, filters)
                g = grad[self._get_output_slice(i, j)]

                gx = np.sum(
                    self._reshape_out_to_full(g) * filters,
                    axis=-1)
                grad_x[x_slice] += gx

                gf = self._reshape_out_to_full(g) * self._reshape_in_to_full(x[x_slice])
                if self.batched:
                    gf = np.sum(gf, axis=0)
                grad_filters += gf

        return grad_x, grad_filters

    def _reshape_out_to_full(self, out):
        return out.reshape(out.shape[:-1] + (1,) + out.shape[-1:])

    def _reshape_filters_to_full(self, f):
        return f

    def _reshape_in_to_full(self, x):
        return x.reshape(x.shape + (1,))

    def _get_output_slice(self, i, j):
        s = (slice(i, i+1), slice(j, j+1), slice(None))

        if self.batched:
            s = (slice(None),) + s

        return s

    def _get_x_slice(self, i, j, filters):
        s = (
            slice(i * self.step, i * self.step + filters.shape[0]),
            slice(j * self.step, j * self.step + filters.shape[1]),
            slice(None)
        )

        if self.batched:
            s = (slice(None),) + s

        return s
