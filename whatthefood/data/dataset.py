class Dataset(object):
    def __init__(self, ids, inputs, outputs, classes=None):
        assert len(ids) == len(inputs) == len(outputs)

        self.ids = ids
        self.inputs = inputs
        self.outputs = outputs
        self.classes = classes

    def __len__(self):
        return len(self.ids)
