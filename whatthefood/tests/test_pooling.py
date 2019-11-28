import unittest
import numpy as np


import whatthefood.graph as graph


class TestOps(unittest.TestCase):
    x = np.array([
        [
            [0, 1],
            [2, 3],
            [1, 1],
            [4, 3],
        ],
        [
            [1, 1],
            [0, 2],
            [3, 4],
            [2, 1],
        ],
        [
            [-1, 0],
            [1, 1],
            [5, 2],
            [0, 0],
        ],
        [
            [1, 0],
            [2, 1],
            [3, 4],
            [1, 0],
        ]
    ])

    y = np.array([
        [
            [2, 3],
            [4, 4],
        ],
        [
            [2, 1],
            [5, 4],
        ]
    ])

    y_step_1 = np.array([
        [
            [2, 3],
            [3, 4],
            [4, 4],
        ],
        [
            [1, 2],
            [5, 4],
            [5, 4],
        ],
        [
            [2, 1],
            [5, 4],
            [5, 4],
        ]
    ])

    grad = np.array([
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 0],
        ],
        [
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0],
        ],
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
        ],
        [
            [0, 0],
            [1, 1],
            [0, 1],
            [0, 0],
        ]
    ])

    grad_step_1 = np.array([
        [
            [0, 0],
            [1, 1],  # 1 1
            [0, 0],
            [1, 0],
        ],
        [
            [1, 0],  # 1 0
            [0, 1],  # 0 1
            [1, 4],  # 1 4
            [0, 0],
        ],
        [
            [0, 0],
            [1, 1],
            [4, 0],
            [0, 0],
        ],
        [
            [0, 0],
            [1, 1],
            [0, 2],
            [0, 0],
        ]
    ])

    def test_max_pooling(self):
        x = graph.Constant(self.x)

        y = graph.MaxPooling2d(x, 2, 2)
        y_step_1 = graph.MaxPooling2d(x, 2, 1)

        np.testing.assert_array_equal(self.y, graph.run(y))
        np.testing.assert_array_equal(self.y_step_1, graph.run(y_step_1))

    def test_max_pooling_batched(self):
        x = graph.Placeholder(self.x.shape, batched=True)

        y = graph.MaxPooling2d(x, 2, 2)
        y_step_1 = graph.MaxPooling2d(x, 2, 1)

        np.testing.assert_array_equal([self.y, self.y * 2], graph.run(y, {x: np.array([self.x, self.x * 2])}))
        np.testing.assert_array_equal(
            [self.y_step_1, self.y_step_1 * 2],
            graph.run(y_step_1, {x: np.array([self.x, self.x * 2])})
        )

    def test_max_pooling_grad(self):
        x = graph.Placeholder(self.x.shape, batched=True)

        y = graph.MaxPooling2d(x, 2, 2)
        y_step_1 = graph.MaxPooling2d(x, 2, 1)

        gx = graph.Grad(y, x)
        gx_step_1 = graph.Grad(y_step_1, x)

        np.testing.assert_array_equal(self.grad, graph.run(gx))
        np.testing.assert_array_equal(self.grad_step_1, graph.run(gx_step_1))

    def test_max_pooling_grad_batched(self):
        x = graph.Constant(self.x)

        y = graph.MaxPooling2d(x, 2, 2)
        y_step_1 = graph.MaxPooling2d(x, 2, 1)

        gx = graph.Grad(y, x)
        gx_step_1 = graph.Grad(y_step_1, x)

        np.testing.assert_array_equal(self.grad, graph.run(gx))
        np.testing.assert_array_equal(self.grad_step_1, graph.run(gx_step_1))

