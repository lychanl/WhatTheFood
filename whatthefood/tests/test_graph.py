import unittest
import numpy as np

import whatthefood.graph as graph


class TestGraph(unittest.TestCase):
    def test_simple(self):
        c = graph.Constant(2)

        self.assertEqual(graph.run(c), 2)

    def test_multiple_simple(self):
        c1 = graph.Constant(2)
        c2 = graph.Constant(3)

        self.assertSequenceEqual(graph.run((c1, c2)), (2, 3))

    def test_non_trivial(self):
        a = graph.Constant(2)
        b = graph.Constant(3)
        c = graph.Sum(a, b)

        self.assertEqual(graph.run(c), 5)

    def test_placeholder(self):
        a = graph.Placeholder(shape=(), batched=False)
        b = graph.Constant(2)
        c = graph.Sum(a, b)

        self.assertEqual(graph.run(c, {a: 3}), 5)
        self.assertEqual(graph.run(c, {a: 4}), 6)

    def test_batched(self):
        a = graph.Placeholder(shape=(2,), batched=True)
        b = graph.Constant([1, 2])
        c = graph.Sum(a, b)

        np.testing.assert_array_equal(graph.run(c, {a: np.array([[1, 2]])}), np.array([[2, 4]]))
        np.testing.assert_array_equal(graph.run(c, {a: np.array([[1, 2], [3, 4]])}), np.array([[2, 4], [4, 6]]))

    def test_grad_simple(self):
        a = graph.Constant(1)
        b = graph.Constant(2)
        c = graph.Constant(4)

        d = graph.Sum(a, b)
        e = graph.MultiplyByScalar(d, c)

        g = graph.Grad(e, [a, b, c])

        self.assertSequenceEqual(graph.run(g), [4, 4, 3])

    def test_grad_placeholder_variable(self):
        a = graph.Constant(1)
        b = graph.Placeholder(batched=False, shape=())
        c = graph.Variable(shape=())

        c.value = 4

        d = graph.Sum(a, b)
        e = graph.MultiplyByScalar(d, c)

        g = graph.Grad(e, [a, b, c])

        self.assertSequenceEqual(graph.run(g, {b: 2}), [4, 4, 3])

    def test_grad(self):
        a = graph.Constant(1)
        b = graph.Constant(2)
        c = graph.Constant(4)

        d = graph.Sum(a, b)
        e = graph.Sum(b, c)
        f = graph.MultiplyByScalar(d, e)

        g = graph.Grad(f, [a, b, c])

        self.assertSequenceEqual(graph.run(g), [6, 9, 3])

