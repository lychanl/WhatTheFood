import unittest
import numpy as np

import whatthefood.graph as graph


class TestActivations(unittest.TestCase):
    def test_relu(self):
        x = graph.Constant([[1, 2, -3, 4], [-1, 2, -3, 0]])
        y = graph.ReLU(x)
        y01 = graph.ReLU(x, 0.1)

        np.testing.assert_array_equal([[1, 2, 0, 4], [0, 2, 0, 0]], graph.run(y))
        np.testing.assert_array_equal([[1, 2, -3 * 0.1, 4], [-1 * 0.1, 2, -3 * 0.1, 0]], graph.run(y01))

    def test_relu_grad(self):
        x = graph.Constant([[1, 2, -3, 4], [-1, 2, -3, 0]])
        y = graph.ReLU(x)
        y01 = graph.ReLU(x, 0.1)

        g = graph.Grad(y, x)
        g01 = graph.Grad(y01, x)

        np.testing.assert_array_equal([[1, 1, 0, 1], [0, 1, 0, 1]], graph.run(g))
        np.testing.assert_array_equal([[1, 1, 0.1, 1], [0.1, 1, 0.1, 1]], graph.run(g01))

    def test_sigmoid(self):
        x = graph.Constant([1, 0, -1])
        y = graph.Sigmoid(x)

        np.testing.assert_array_equal(1 / (1 + np.exp([-1, 0, 1])), graph.run(y))

    def test_softmax(self):
        x_arr = np.array([[1, 2, 3], [-1, -2, -3]])
        x = graph.Constant(x_arr)
        y = graph.Softmax(x)

        s = np.repeat(np.sum(np.exp(x_arr - [[3], [-1]]), axis=-1), x_arr.shape[-1]).reshape(x_arr.shape)
        np.testing.assert_array_equal(np.exp(x_arr - [[3], [-1]]) / s, graph.run(y))
