from whatthefood.graph import Grad, Placeholder, Variable, run
from whatthefood.nn.layers import Layer


class Model(object):
    def __init__(self):
        self.layers = []
        self.variables = []
        self.non_layer_nodes = []
        self.output = None
        self.inputs = []
        self.grad = None

    def add(self, builder, *args, current_output_as_argument=True, set_output=True, **kwargs):
        if current_output_as_argument and self.output is not None:
            args = (self.output, *args)
        obj = builder(*args, **kwargs)

        if isinstance(obj, Placeholder):
            self._add_input(obj)

        if isinstance(obj, Variable):
            self.variables.append(obj)

        if isinstance(obj, Layer):
            self.layers.append(obj)
            self.variables.extend(obj.vars)

            if set_output:
                self.output = obj.output
        else:
            self.non_layer_nodes.append(obj)

            if set_output:
                self.output = obj

        return obj

    def _add_input(self, obj):
        self.inputs.append(obj)

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) == len(self.inputs)
        assert not args or not kwargs

        input_dict = kwargs if kwargs else {i: v for i, v in zip(self.inputs, args)}

        return run(self.output, input_dict)
