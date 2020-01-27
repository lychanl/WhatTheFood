import numpy as np


class Dataset(object):
    def __init__(self, ids, inputs, outputs, classes=None, processors=None):
        assert len(ids) == len(inputs) == len(outputs)

        self.ids = ids
        self.inputs = inputs
        self.outputs = outputs
        self.classes = classes

        if processors is None:
            processors = ()
        self.processors = processors

    def __len__(self):
        return len(self.ids)

    def get_batch(self, size):
        samples = np.random.random_integers(0, len(self.inputs) - 1, size)

        inputs = np.ndarray((size,) + self.inputs[0].shape, dtype=np.float32)
        outputs = np.ndarray((size,) + self.outputs[0].shape, dtype=np.float32)

        for i, s in enumerate(samples):
            inp = self.inputs[s]
            out = self.outputs[s]
            for p in self.processors:
                inp, out = p.process(inp, out)
            inputs[i] = inp
            outputs[i] = out

        return inputs, outputs
