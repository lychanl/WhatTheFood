import unittest
import numpy as np


import whatthefood.graph as graph


class TestConv(unittest.TestCase):
    # 1 2 3 4
    # 1 2 3 4
    # 1 2 3 4
    # 1 2 3 4
    #
    # 1 1 1 1
    # 1 1 1 1
    # 1 1 1 1
    # 1 1 1 1
    x = np.array([
        [[1, 1], [2, 1], [3, 1], [4, 1]],
        [[1, 1], [2, 1], [3, 1], [4, 1]],
        [[1, 1], [2, 1], [3, 1], [4, 1]],
        [[1, 1], [2, 1], [3, 1], [4, 1]]
    ])

    # 1 0
    # 0 1
    #
    # 0 1
    # 1 0
    #
    #
    # 1 1
    # 1 1
    #
    # 1 2
    # 3 4
    filters = np.array([
        [
            [[1, 1], [0, 1]],
            [[0, 1], [1, 2]]
        ],
        [
            [[0, 1], [1, 3]],
            [[1, 1], [0, 4]]
        ]
    ])

    # 5 7 9
    # 5 7 9
    # 5 7 9
    #
    # 16 20 24
    # 16 20 24
    # 16 20 24
    expected = np.array([
        [[5, 16], [7, 20], [9, 24]],
        [[5, 16], [7, 20], [9, 24]],
        [[5, 16], [7, 20], [9, 24]],
    ])

    # 5 9
    # 5 9
    #
    # 16 24
    # 16 24

    expected_step_2 = np.array([
        [[5, 16], [9, 24]],
        [[5, 16], [9, 24]],
    ])

    # 2 3 3 1
    # 3 6 6 3
    # 3 6 6 3
    # 1 3 3 2
    #
    # 1 4 4 3
    # 5 12 12 7
    # 5 12 12 7
    # 4 8 8 4
    grad_x = np.array([
        [[2, 1], [3, 4], [3, 4], [1, 3]],
        [[3, 5], [6, 12], [6, 12], [3, 7]],
        [[3, 5], [6, 12], [6, 12], [3, 7]],
        [[1, 4], [3, 8], [3, 8], [2, 4]],
    ])

    # 18 27
    # 18 27
    #
    # 9 9
    # 9 9
    #
    #
    # 18 27
    # 18 27
    #
    # 9 9
    # 9 9
    grad_f = np.array([
        [
            [[18, 18], [9, 9]],
            [[27, 27], [9, 9]]
        ],
        [
            [[18, 18], [9, 9]],
            [[27, 27], [9, 9]]
        ]
    ])

    # 2 1 2 1
    # 1 2 1 2
    # 2 1 2 1
    # 1 2 1 2
    #
    # 1 3 1 3
    # 4 4 4 4
    # 1 3 1 3
    # 4 4 4 4
    grad_x_step_2 = np.array([
        [[2, 1], [1, 3], [2, 1], [1, 3]],
        [[1, 4], [2, 4], [1, 4], [2, 4]],
        [[2, 1], [1, 3], [2, 1], [1, 3]],
        [[1, 4], [2, 4], [1, 4], [2, 4]]
    ])

    # 8 12
    # 8 12
    #
    # 4 4
    # 4 4
    #
    #
    # 8 12
    # 8 12
    #
    # 4 4
    # 4 4
    grad_f_step_2 = np.array([
        [
            [[8, 8], [4, 4]],
            [[12, 12], [4, 4]]
        ],
        [
            [[8, 8], [4, 4]],
            [[12, 12], [4, 4]]
        ]
    ])

    def test_convolution(self):
        x_c = graph.Constant(self.x)
        f_c = graph.Constant(self.filters)

        conv = graph.Convolution(x_c, f_c)

        np.testing.assert_array_equal(graph.run(conv), self.expected)

    def test_convolution_batched(self):
        x_p = graph.Placeholder(self.x.shape, batched=True)
        f_c = graph.Constant(self.filters)

        conv = graph.Convolution(x_p, f_c)

        x = np.stack([self.x, 2 * self.x, 3 * self.x])
        expected = np.stack([self.expected, 2 * self.expected, 3 * self.expected])

        np.testing.assert_array_equal(graph.run(conv, {x_p: x}), expected)

    def test_convolution_grad(self):
        x_c = graph.Constant(self.x)
        f_c = graph.Constant(self.filters)

        conv = graph.Convolution(x_c, f_c)

        grad = graph.Grad(conv, [x_c, f_c])

        grad_x, grad_f = graph.run(grad)

        np.testing.assert_array_equal(grad_x, self.grad_x)
        np.testing.assert_array_equal(grad_f, self.grad_f)

    def test_convolution_grad_batched(self):
        x_p = graph.Placeholder(self.x.shape, batched=True)
        f_c = graph.Constant(self.filters)

        x = np.stack([self.x, 2 * self.x, 3 * self.x])

        conv = graph.Convolution(x_p, f_c)

        grad = graph.Grad(conv, [x_p, f_c])

        grad_x, grad_f = graph.run(grad, {x_p: x})

        np.testing.assert_array_equal(grad_x, np.stack([self.grad_x] * 3))
        np.testing.assert_array_equal(grad_f, self.grad_f * 6)

    def test_convolution_with_step(self):
        x = graph.Constant(self.x)
        f = graph.Constant(self.filters)

        conv = graph.Convolution(x, f, step=2)

        np.testing.assert_array_equal(self.expected_step_2, graph.run(conv))

    def test_convolution_grad_with_step(self):
        x_c = graph.Constant(self.x)
        f_c = graph.Constant(self.filters)

        conv = graph.Convolution(x_c, f_c, step=2)

        grad = graph.Grad(conv, [x_c, f_c])

        grad_x, grad_f = graph.run(grad)

        np.testing.assert_array_equal(grad_x, self.grad_x_step_2)
        np.testing.assert_array_equal(grad_f, self.grad_f_step_2)
