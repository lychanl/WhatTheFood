import numpy as np

from whatthefood.graph import GT, Constant, ArgMax, Reshape
from whatthefood.data.preprocessing import yolo_flip_out


def to_classes(prob):
    if prob.shape[-1] == 1:
        return Reshape(GT(prob, Constant(0.5)), prob.shape[:-1])
    else:
        return ArgMax(prob)


def get_output_mean_with_flipped(model, inputs):
    out = model(inputs)
    i1 = np.flip(inputs, axis=1)
    out1 = yolo_flip_out(model(i1), axis=1)
    i2 = np.flip(inputs, axis=2)
    out2 = yolo_flip_out(model(i2), axis=2)
    i3 = np.flip(i1, axis=2)
    out3 = yolo_flip_out(yolo_flip_out(model(i3), axis=1), axis=2)

    return np.mean([out, out1, out2, out3], axis=0)
