import whatthefood.graph as graph


def accuracy(expected, result, flags=None):
    if flags is None:
        return graph.ReduceMean(graph.Equal(result, expected))
    else:
        return graph.Divide(
            graph.ReduceSum(graph.Multiply(graph.Equal(result, expected), flags)),
            graph.ReduceSum(flags))


def tp_rate(expected, result):
    return graph.Divide(
        graph.ReduceSum(graph.Multiply(graph.Equal(result, expected), expected)),
        graph.ReduceSum(expected))


def fp_rate(expected, result):
    ne = graph.Negate(expected)
    return graph.Divide(
        graph.ReduceSum(graph.Multiply(graph.Negate(graph.Equal(result, expected)), ne)),
        graph.ReduceSum(ne))

