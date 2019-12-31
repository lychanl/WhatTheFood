import numpy as np


class Layer(object):
    def __init__(self, inputs, output, vars):
        self.inputs = inputs
        self.output = output
        self.vars = vars

    def get_input_size(self):
        return np.sum([np.prod(i.shape) for i in self.inputs])

    def initialize_weights(self, initializer):
        inputs_size = self.get_input_size()
        for v in self.vars:
            v.value = initializer.get_weights(v.shape, inputs_size)
