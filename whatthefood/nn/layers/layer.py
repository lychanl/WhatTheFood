import numpy as np


class Layer(object):
    def __init__(self, inputs, output, vars):
        self.inputs = inputs
        self.output = output
        self.vars = vars

    def initialize_weights(self, initializer):
        inputs_size = np.sum([np.prod(i.shape) for i in self.inputs])
        for v in self.vars:
            v.value = initializer.get_weights(v.shape, inputs_size)
