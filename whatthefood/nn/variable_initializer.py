import numpy as np


class GaussianInitializer(object):
    def __init__(self):
        pass

    def get_weights(self, shape, layer_input_size=None):
        std = np.sqrt(2 / layer_input_size) if layer_input_size else 1.

        return np.random.normal(0., std, shape)
