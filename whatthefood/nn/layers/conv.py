import whatthefood.graph as graph
from whatthefood.nn.layers.layer import Layer


class Convolution(Layer):
    def __init__(
            self, x, filters, filter_size, step, bias=True,
            activation=None, *activation_args, **activation_kwargs):
        if isinstance(x, Layer):
            x = x.output

        self.kernel = graph.Variable((filter_size, filter_size, filters))

        y = graph.Convolution(x, self.kernel, step)

        self.bias = graph.Variable(y.shape) if bias else None

        if bias:
            y = graph.Sum(y, self.bias)

        if activation:
            y = activation(y, *activation_args, **activation_kwargs)

        super(Convolution, self).__init__(y, [self.kernel, self.bias])
