from whatthefood.graph import GT, Constant, ArgMax, Reshape


def to_classes(prob):
    if prob.shape[-1] == 1:
        return Reshape(GT(prob, Constant(0.5)), prob.shape[:-1])
    else:
        return ArgMax(prob)
