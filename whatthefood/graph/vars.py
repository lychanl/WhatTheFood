from whatthefood.graph.node import Node

import numpy as np


class Variable(Node):
    def __init__(self, shape):
        super(Variable, self).__init__(shape, False)

        self.value = np.ndarray(shape)

    def do(self):
        return self.value

    def backpropagate(self, grad):
        return ()


class Placeholder(Node):
    def __init__(self, shape, batched):
        super(Placeholder, self).__init__(shape, batched)

    def backpropagate(self, grad):
        return ()


class Constant(Node):
    def __init__(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        super(Constant, self).__init__(np.shape(value), False)

        self.value = value

    def do(self):
        return self.value

    def backpropagate(self, grad):
        return ()
