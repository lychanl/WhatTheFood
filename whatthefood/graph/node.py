import numpy as np


class Node:
    def __init__(self, shape, batched, *inputs):
        self.batched = batched
        self.shape = shape
        self.inputs = list(inputs)

        self.tensor = None

    @property
    def size(self):
        return np.prod(self.shape)

    def build_tf(self, tf, sess):
        if self.tensor is None:
            input_tensors = [i.build_tf(tf, sess) for i in self.inputs]
            self.tensor = self._build_tf(tf, *input_tensors)
            self.initialize(tf, sess)

        return self.tensor

    def _build_tf(self, tf, *inputs):
        raise NotImplementedError

    def initialize(self, tf, sess):
        pass

    def __getstate__(self):
        state = dict(self.__dict__)
        state['tensor'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tensor = None

    def clear_tf(self):
        self.tensor = None
