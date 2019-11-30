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

    def test_reduce_sum(self):
        x = graph.Constant([[[1], [2]], [[3], [4]], [[5], [6]]])
        y1 = graph.ReduceSum(x, axis=0)
        y2 = graph.ReduceSum(x, axis=(1, -1))
        y3 = graph.ReduceSum(x)

        np.testing.assert_array_equal([[9], [12]], graph.run(y1))
        np.testing.assert_array_equal([3, 7, 11], graph.run(y2))
        self.assertEqual(21, graph.run(y3))

    def test_reduce_sum_batched(self):
        x_arr = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])
        y_arr = np.array([9, 12])
        x = graph.Placeholder(shape=(3, 2, 1), batched=True)
        y1 = graph.ReduceSum(x, (0, 2), True)
        y2 = graph.ReduceSum(x, (0, 2), False)

        np.testing.assert_array_equal(y_arr * 3, graph.run(y1, {x: np.array([x_arr, 2 * x_arr])}))
        np.testing.assert_array_equal([y_arr, 2 * y_arr], graph.run(y2, {x: np.array([x_arr, 2 * x_arr])}))

    def test_reduce_sum_grad(self):
        x = graph.Constant([[[1], [2]], [[3], [4]], [[5], [6]]])
        y1 = graph.ReduceSum(x, axis=0)
        y2 = graph.ReduceSum(x, axis=(1, -1))
        y3 = graph.ReduceSum(x)

        g1 = graph.Grad(y1, x)
        g2 = graph.Grad(y2, x)
        g3 = graph.Grad(y3, x)

        np.testing.assert_array_equal(np.ones_like(x.value), graph.run(g1))
        np.testing.assert_array_equal(np.ones_like(x.value), graph.run(g2))
        np.testing.assert_array_equal(np.ones_like(x.value), graph.run(g3))

    def test_reduce_sum_grad_batched(self):
        x_arr = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])

        x = graph.Placeholder(shape=(3, 2, 1), batched=True)
        y1 = graph.ReduceSum(x, (0, 2), True)
        y2 = graph.ReduceSum(x, (0, 2), False)

        g1 = graph.Grad(y1, x)
        g2 = graph.Grad(y2, x)

        np.testing.assert_array_equal(
            [np.ones_like(x_arr), np.ones_like(x_arr)],
            graph.run(g1, {x: np.array([x_arr, 2 * x_arr])}))
        np.testing.assert_array_equal(
            [np.ones_like(x_arr), np.ones_like(x_arr)],
            graph.run(g2, {x: np.array([x_arr, 2 * x_arr])}))

    def test_reduce_mean(self):
        x = graph.Constant([[[1], [2]], [[3], [4]], [[5], [6]]])
        y1 = graph.ReduceMean(x, axis=0)
        y2 = graph.ReduceMean(x, axis=(1, -1))
        y3 = graph.ReduceMean(x)

        np.testing.assert_array_equal([[3], [4]], graph.run(y1))
        np.testing.assert_array_equal([1.5, 3.5, 5.5], graph.run(y2))
        self.assertEqual(3.5, graph.run(y3))

    def test_reduce_mean_batched(self):
        x_arr = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])
        y_arr = np.array([3, 4])
        x = graph.Placeholder(shape=(3, 2, 1), batched=True)
        y1 = graph.ReduceMean(x, (0, 2), True)
        y2 = graph.ReduceMean(x, (0, 2), False)

        np.testing.assert_array_equal(y_arr * 1.5, graph.run(y1, {x: np.array([x_arr, 2 * x_arr])}))
        np.testing.assert_array_equal([y_arr, 2 * y_arr], graph.run(y2, {x: np.array([x_arr, 2 * x_arr])}))

    def test_reduce_mean_grad(self):
        x = graph.Constant([[[1], [2]], [[3], [4]], [[5], [6]]])
        y1 = graph.ReduceMean(x, axis=0)
        y2 = graph.ReduceMean(x, axis=(1, -1))
        y3 = graph.ReduceMean(x)

        g1 = graph.Grad(y1, x)
        g2 = graph.Grad(y2, x)
        g3 = graph.Grad(y3, x)

        np.testing.assert_array_equal(np.ones_like(x.value) / 3, graph.run(g1))
        np.testing.assert_array_equal(np.ones_like(x.value) / 2, graph.run(g2))
        np.testing.assert_array_equal(np.ones_like(x.value) / 6, graph.run(g3))

    def test_reduce_mean_grad_batched(self):
        x_arr = np.array([[[1], [2]], [[3], [4]], [[5], [6]]])

        x = graph.Placeholder(shape=(3, 2, 1), batched=True)
        y1 = graph.ReduceMean(x, (0, 2), True)
        y2 = graph.ReduceMean(x, (0, 2), False)

        g1 = graph.Grad(y1, x)
        g2 = graph.Grad(y2, x)

        np.testing.assert_array_equal(
            [np.ones_like(x_arr) / 6, np.ones_like(x_arr) / 6],
            graph.run(g1, {x: np.array([x_arr, 2 * x_arr])}))
        np.testing.assert_array_equal(
            [np.ones_like(x_arr) / 3, np.ones_like(x_arr) / 3],
            graph.run(g2, {x: np.array([x_arr, 2 * x_arr])}))
