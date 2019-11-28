import unittest
import numpy as np


import whatthefood.graph as graph


class TestOps(unittest.TestCase):
    def test_matmul(self):
        x_arr = np.array([[1, 2], [2, 3], [3, 4]])
        y_arr = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])

        x = graph.Constant(x_arr)
        y = graph.Constant(y_arr)

        m = graph.Matmul(x, y)

        np.testing.assert_array_equal(graph.run(m), np.matmul(x_arr, y_arr))

    def test_matmul_grad(self):
        x_arr = np.array([[1, 2], [2, 3], [3, 4]])
        y_arr = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])

        x = graph.Constant(x_arr)
        y = graph.Constant(y_arr)

        m = graph.Matmul(x, y)

        g = graph.Grad(m, [x, y])

        mv, (g_x, g_y) = graph.run((m, g))

        self.assertSequenceEqual(g_x.shape, x.shape)
        self.assertSequenceEqual(g_y.shape, y.shape)

        np.testing.assert_array_equal(g_x, np.matmul(np.ones_like(mv), y_arr.T))
        np.testing.assert_array_equal(g_y, np.matmul(x_arr.T, np.ones_like(mv)))

