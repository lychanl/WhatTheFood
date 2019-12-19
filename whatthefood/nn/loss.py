from whatthefood.graph import Difference, Square, ReduceMean


def mse(expected, result):
    return ReduceMean(Square(Difference(expected, result)))
