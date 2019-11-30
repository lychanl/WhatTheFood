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

    def test_matmul_vec(self):
        x = graph.Constant([1, 2, 3])
        y = graph.Constant([[1, 2], [1, 3], [2, 4]])

        m = graph.Matmul(x, y)

        np.testing.assert_array_equal([9, 20], graph.run(m))

    def test_matmul_vec_grad(self):
        x = graph.Constant([1, 2, 3])
        y = graph.Constant([[1, 2], [1, 3], [2, 4]])

        m = graph.Matmul(x, y)
        g = graph.Grad(m, [x, y])

        g_x, g_y = graph.run(g)

        np.testing.assert_array_equal([3, 4, 6], g_x)
        np.testing.assert_array_equal([[1, 1], [2, 2], [3, 3]], g_y)

    def test_relu(self):
        x = graph.Constant([[1, 2, -3, 4], [-1, 2, -3, 0]])
        y = graph.ReLU(x)
        y01 = graph.ReLU(x, 0.1)

        np.testing.assert_array_equal([[1, 2, 0, 4], [0, 2, 0, 0]], graph.run(y))
        np.testing.assert_array_equal([[1, 2, -3*0.1, 4], [-1*0.1, 2, -3*0.1, 0]], graph.run(y01))

    def test_relu_grad(self):
        x = graph.Constant([[1, 2, -3, 4], [-1, 2, -3, 0]])
        y = graph.ReLU(x)
        y01 = graph.ReLU(x, 0.1)

        g = graph.Grad(y, x)
        g01 = graph.Grad(y01, x)

        np.testing.assert_array_equal([[1, 1, 0, 1], [0, 1, 0, 1]], graph.run(g))
        np.testing.assert_array_equal([[1, 1, 0.1, 1], [0.1, 1, 0.1, 1]], graph.run(g01))
