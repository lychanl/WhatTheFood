import whatthefood.graph as graph


def accuracy(result, expected, flags=None):
    if flags is None:
        return graph.ReduceMean(graph.Equal(result, expected))
    else:
        return graph.Divide(
            graph.ReduceSum(graph.Multiply(graph.Equal(result, expected), flags)),
            graph.ReduceSum(flags))


def precision(result, expected):
    return graph.Divide(
        graph.ReduceSum(graph.Multiply(graph.Equal(result, expected), expected)),
        graph.ReduceSum(expected))
