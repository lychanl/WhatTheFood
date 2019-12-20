import numpy as np


class Node:
    def __init__(self, shape, batched, *inputs):
        self.batched = batched
        self.shape = shape
        self.inputs = list(inputs)

    @property
    def size(self):
        return np.prod(self.shape)
