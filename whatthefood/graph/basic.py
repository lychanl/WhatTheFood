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


def flatten(x):
    return Reshape(x, shape=(-1,))


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


class Concatenate(Node):
    def __init__(self, inputs, axis=0):
        for i1, i2 in zip(inputs[:-1], inputs[1:]):
            s1 = list(i1.shape)
            s2 = list(i2.shape)
            s1[axis] = 0
            s2[axis] = 0
            assert tuple(s1) == tuple(s2)
            assert i1.batched == i2.batched

        out_shape = (*inputs[0].shape[:axis], sum(i.shape[axis] for i in inputs))
        if axis != -1:
            out_shape = (*out_shape, *out_shape[(axis+1):])

        super(Concatenate, self).__init__(out_shape, inputs[0].batched, *inputs)
        self.axis = axis if not self.batched or axis < 0 else axis + 1

        start = 0
        self.slices = []
        for inp in inputs:
            s = tuple(
                slice(start, start + inp.shape[a]) if a == axis or a - len(self.shape) == axis else slice(None)
                for a, d in enumerate(inp.shape)
            )

            if self.batched:
                s = (slice(None), *s)

            start += inp.shape[self.axis]
            self.slices.append(s)

    def do(self, *inputs):
        return np.concatenate(inputs, self.axis)

    def backpropagate(self, grad, *inputs):
        return tuple(grad[s] for s in self.slices)


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
            if not isinstance(axis, Sequence):
                axis = (axis,)
            assert max(axis) < len(x.shape) and min(axis) >= -len(x.shape)
            out_shape = tuple(d for i, d in enumerate(x.shape) if i not in axis and i - len(x.shape) not in axis)
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
        return 2 * x * grad,


class Multiply(Node):
    def __init__(self, x, y):
        assert len(x.shape) == len(y.shape)
        assert x.batched == y.batched
        for sx, sy in zip(x.shape, y.shape):
            assert sx == sy or (sx == 1 if x.size < y.size else sy == 1)

        super(Multiply, self).__init__(x.shape if len(x.shape) > len(y.shape) else y.shape, x.batched, x, y)

    def do(self, x, y):
        return x * y

    def backpropagate(self, grad, x, y):
        yg = x * grad
        xg = y * grad

        if xg.size > x.size:
            xg = np.sum(xg, axis=tuple(i for i, s in enumerate(x.shape) if s == 1), keepdims=True)
        if yg.size > y.size:
            yg = np.sum(yg, axis=tuple(i for i, s in enumerate(y.shape) if s == 1), keepdims=True)

        return xg, yg
