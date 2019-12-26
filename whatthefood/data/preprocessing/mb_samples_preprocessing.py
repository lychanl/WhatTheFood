import numpy as np


class AddNoisePreprocessor(object):
    def __init__(self, mean=0., std=1., limits=None):
        self.mean = mean
        self.std = std
        self.limits = limits

    def process(self, inp, out):
        inp = inp + np.random.normal(self.mean, self.std, inp.shape)
        if self.limits:
            inp = np.clip(inp, self.limits[0], self.limits[1])
        return inp, out


class FlipPreprocessor(object):
    def process(self, inp, out):
        if np.random.uniform() < 0.5:
            inp = np.flip(inp, axis=0)
            out = np.flip(out, axis=0)
        if np.random.uniform() < 0.5:
            inp = np.flip(inp, axis=1)
            out = np.flip(out, axis=1)
        return inp, out
