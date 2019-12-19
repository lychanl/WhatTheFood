import whatthefood.graph as graph
from whatthefood.nn.layers.layer import Layer


class Dense(Layer):
    def __init__(self, x, n, bias=True, activation=None, *activation_args, **activation_kwargs):
        if isinstance(x, Layer):
            x = x.output

        assert len(x.shape) == 1

        self.kernel = graph.Variable((x.shape[0], n))

        self.bias = graph.Variable((n,)) if bias else None

        y = graph.Matmul(x, self.kernel)

        if bias:
            y = graph.Sum(y, self.bias)

        if activation:
            y = activation(y, *activation_args, **activation_kwargs)

        super(Dense, self).__init__((x,), y, (self.kernel, self.bias) if self.bias else (self.kernel,))

