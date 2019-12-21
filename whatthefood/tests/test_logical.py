import unittest
import numpy as np


import whatthefood.graph as graph


class TestLogical(unittest.TestCase):
    def test_gt(self):
        x1 = graph.Constant([1, 2, 3, 2, 1])
        x2 = graph.Constant([2, 1, 3, 4, 0])
        y = graph.GT(x1, x2)

        np.testing.assert_array_equal([0, 1, 0, 0, 1], graph.run(y))

    def test_equal(self):
        x1 = graph.Constant([2, 2, 3, 2, 1])
        x2 = graph.Constant([2, 1, 3, 4, 0])
        y = graph.Equal(x1, x2)

        np.testing.assert_array_equal([1, 0, 1, 0, 0], graph.run(y))

    def test_argmax(self):
        x = graph.Constant([[2, 1], [3, 4], [0, 2]])
        y = graph.ArgMax(x)

        np.testing.assert_array_equal([0, 1, 1], graph.run(y))

    def test_negate(self):
        x = graph.Constant([[1, 0], [1, 1], [0, 1]])
        y = graph.Negate(x)

        np.testing.assert_array_equal([[0, 1], [0, 0], [1, 0]], graph.run(y))
