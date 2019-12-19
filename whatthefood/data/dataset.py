import numpy as np


class Dataset(object):
    def __init__(self, ids, inputs, outputs, classes=None):
        assert len(ids) == len(inputs) == len(outputs)

        self.ids = ids
        self.inputs = inputs
        self.outputs = outputs
        self.classes = classes

    def __len__(self):
        return len(self.ids)

    def get_batch(self, size):
        samples = np.random.random_integers(0, len(self.inputs), size)

        inputs = np.ndarray((size,) + self.inputs[0].shape)
        outputs = np.ndarray((size,) + self.inputs[0].shape)

        for i, s in enumerate(samples):
            inputs[i] = self.inputs[s]
            outputs[i] = self.outputs[s]

        return inputs, outputs
