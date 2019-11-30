import whatthefood.graph as graph
from whatthefood.nn.layers.layer import Layer


class Dense(Layer):
    def __init__(self, x, n, bias=True, activation=None, *activation_args, **activation_kwargs):
        if isinstance(x, Layer):
            x = x.output

        assert len(x.shape) == 1

        self.kernel = graph.Variable((x.shape[0], n))

        if bias:
            self.bias = graph.Variable((n,))

        y = x * self.kernel

        if bias:
            y = y + self.bias

        if activation:
            y = activation(y, *activation_args, **activation_kwargs)

        super(Dense, self).__init__(y, (self.kernel, self.bias) if self.bias else (self.kernel,))

