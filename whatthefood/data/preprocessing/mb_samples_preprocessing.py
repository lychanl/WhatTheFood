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


def yolo_flip_out(out, axis):
    new_out = np.flip(np.copy(out), axis=axis)

    if len(out.shape) == 3 and axis == 0:
        new_out[:,:,1] = 1 - new_out[:,:,1]
    elif len(out.shape) == 4 and axis == 1:
        new_out[:,:,:,1] = 1 - new_out[:,:,:,1]

    elif len(out.shape) == 3 and axis == 1:
        new_out[:,:,2] = 1 - new_out[:,:,2]
    elif len(out.shape) == 4 and axis == 2:
        new_out[:,:,:,2] = 1 - new_out[:,:,:,2]

    return new_out


class FlipPreprocessor(object):
    def process(self, inp, out):
        if np.random.uniform() < 0.5:
            inp = np.flip(inp, axis=0)
            out = yolo_flip_out(out, axis=0)
        if np.random.uniform() < 0.5:
            inp = np.flip(inp, axis=1)
            out = yolo_flip_out(out, axis=1)
        return inp, out
