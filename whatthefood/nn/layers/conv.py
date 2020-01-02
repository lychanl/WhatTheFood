import whatthefood.graph as graph
from whatthefood.nn.layers.layer import Layer


class Convolution(Layer):
    def __init__(
            self, x, filters, filter_size, step, bias=True,
            activation=None, padding='VALID', *activation_args, **activation_kwargs):
        if isinstance(x, Layer):
            x = x.output

        self.kernel = graph.Variable((filter_size, filter_size, x.shape[2], filters))

        y = graph.Convolution(x, self.kernel, step, padding)

        self.bias = graph.Variable((filters,)) if bias else None

        if bias:
            y = graph.Sum(y, self.bias)

        if activation:
            y = activation(y, *activation_args, **activation_kwargs)

        super(Convolution, self).__init__((x,), y, [self.kernel, self.bias])

    def get_input_size(self):
        return self.kernel.size
