from collections.abc import Sequence

import numpy as np

from whatthefood.graph import Grad, Placeholder, run


class Minimizer(object):
    def __init__(self, model, loss):
        self.expected_output = Placeholder(model.output.shape, model.output.batched)
        self.loss = loss(self.expected_output, model.output)
        self.model = model

        self.vars = model.variables
        self.grad = Grad(self.loss, self.model)

    def run(self, inputs, expected_output, *args, **kwargs):
        input_dict = None
        if inputs is None:
            input_dict = {}
        elif isinstance(inputs, np.ndarray) and len(self.model.inputs) == 1:
            input_dict = {self.model.inputs[0]: inputs}
        elif isinstance(inputs, dict):
            input_dict = inputs
        elif isinstance(inputs, Sequence):
            input_dict = {i: v for i, v in zip(self.model.inputs, inputs)}

        input_dict[self.expected_output] = expected_output
        loss, grads = run((self.loss, self.grad), input_dict)

        self._run(grads, *args, **kwargs)

        return loss

    def _run(self, grads, *args, **kwargs):
        raise NotImplementedError
