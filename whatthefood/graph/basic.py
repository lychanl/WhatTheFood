from collections.abc import Sequence

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
        grad_y = np.sum(grad, axis=0) if not self.batched_args[1] and self.batched else grad

        return grad_x, grad_y


class Difference(Node):
    def __init__(self, x, y):
        assert x.shape == y.shape
        super(Difference, self).__init__(x.shape, x.batched or y.batched, x, y)

        self.batched_args = x.batched, y.batched

    def do(self, x, y):
        return x - y

    def backpropagate(self, grad, x, y):
        grad_x = np.sum(grad, axis=0) if not self.batched_args[0] and self.batched else grad
        grad_y = -(np.sum(grad, axis=0) if not self.batched_args[0] and self.batched else grad)

        return grad_x, grad_y


class Reshape(Node):
    def __init__(self, x, shape):
        if -1 in shape:
            shape = tuple(s if s != -1 else np.prod(x.shape) // -np.prod(shape) for s in shape)
        super(Reshape, self).__init__(shape, x.batched, x)

    def do(self, x):
        return x.reshape(self.shape if not self.batched else (x.shape[0], *self.shape))

    def backpropagate(self, grad, x):
        return grad.reshape(x.shape)


class Slice(Node):
    def __init__(self, x, id_from, id_to):
        assert len(id_from) <= len(x.shape)
        assert len(id_from) == len(id_to)
        for i_f, i_t, d in zip(id_from, id_to, x.shape):
            assert i_f >= 0
            assert i_t > 0
            assert i_f < i_t
            assert i_t <= d

        self.id_from = id_from
        self.id_to = id_to

        super(Slice, self).__init__(tuple(i_t - i_f for i_t, i_f in zip(id_to, id_from)), x.batched, x)

        self.slice = tuple(slice(i_f, i_t) for i_t, i_f in zip(id_to, id_from))
        if x.batched:
            self.slice = (slice(None), *self.slice)

    def do(self, x):
        return x[self.slice]

    def backpropagate(self, grad, x):
        out = np.zeros_like(x)
        out[self.slice] = grad

        return out,


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


class Reduce(Node):
    def __init__(self, x, axis=None, reduce_batching=None):
        if axis is not None:
            if isinstance(axis, Sequence):
                assert max(axis) < len(x.shape) and min(axis) >= -len(x.shape)
                out_shape = tuple(d for i, d in enumerate(x.shape) if i not in axis and i - len(x.shape) not in axis)
            else:
                assert -len(x.shape) <= axis < len(x.shape)
                out_shape = x.shape[:axis] + x.shape[(axis + 1):]
                axis = axis,
        else:
            out_shape = ()
            axis = tuple(range(len(x.shape)))

        self.axis = axis

        if reduce_batching:
            assert x.batched

        if reduce_batching is None:
            reduce_batching = x.batched

        if x.batched:
            self.axis = tuple([a + 1 if a >= 0 else a for a in self.axis])
            if reduce_batching:
                self.axis = (0,) + self.axis

        super(Reduce, self).__init__(
            out_shape,
            x.batched and not reduce_batching,
            x
        )

        self.reduce_batching = reduce_batching
        self.grad_shape = tuple(d if i not in axis and i - len(x.shape) not in axis else 1 for i, d in enumerate(x.shape))
        if self.batched and x.batched:
            self.grad_shape = (-1,) + self.grad_shape
        elif x.batched:
            self.grad_shape = (1,) + self.grad_shape


class ReduceSum(Reduce):
    def do(self, x):
        return np.sum(x, self.axis)

    def backpropagate(self, grad, x):
        return np.broadcast_to(grad.reshape(self.grad_shape), x.shape),


class ReduceMean(Reduce):
    def do(self, x):
        return np.mean(x, self.axis)

    def _get_divisor(self, x):
        v = 1
        for d in self.axis:
            v *= x.shape[d]

        return v

    def backpropagate(self, grad, x):
        return np.broadcast_to(grad.reshape(self.grad_shape), x.shape) / self._get_divisor(x),


class Square(Node):
    def __init__(self, x):
        super(Square, self).__init__(x.shape, x.batched, x)

    def do(self, x):
        return np.square(x)

    def backpropagate(self, grad, x):
        return 2 * x * grad
