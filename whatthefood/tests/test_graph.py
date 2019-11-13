import unittest
import numpy as np

from whatthefood.graph import run
import whatthefood.graph as graph


class TestGraph(unittest.TestCase):
    def test_simple(self):
        c = graph.Constant(2)

        self.assertEqual(run(c), 2)

    def test_multiple_simple(self):
        c1 = graph.Constant(2)
        c2 = graph.Constant(3)

        self.assertSequenceEqual(run((c1, c2)), (2, 3))

    def test_non_trivial(self):
        a = graph.Constant(2)
        b = graph.Constant(3)
        c = graph.Sum(a, b)

        self.assertEqual(run(c), 5)

    def test_placeholder(self):
        a = graph.Placeholder(shape=(), batched=False)
        b = graph.Constant(2)
        c = graph.Sum(a, b)

        self.assertEqual(run(c, {a: 3}), 5)
        self.assertEqual(run(c, {a: 4}), 6)

    def test_batched(self):
        a = graph.Placeholder(shape=(2,), batched=True)
        b = graph.Constant([1, 2])
        c = graph.Sum(a, b)

        np.testing.assert_array_equal(run(c, {a: np.array([[1, 2]])}), np.array([[2, 4]]))
        np.testing.assert_array_equal(run(c, {a: np.array([[1, 2], [3, 4]])}), np.array([[2, 4], [4, 6]]))

    def test_grad_simple(self):
        a = graph.Constant(1)
        b = graph.Constant(2)
        c = graph.Constant(4)

        d = graph.Sum(a, b)
        e = graph.MultiplyByScalar(d, c)

        g = graph.Grad(e, [a, b, c])

        self.assertSequenceEqual(run(g), [4, 4, 3])

    def test_grad(self):
        a = graph.Constant(1)
        b = graph.Constant(2)
        c = graph.Constant(4)

        d = graph.Sum(a, b)
        e = graph.Sum(b, c)
        f = graph.MultiplyByScalar(d, e)

        g = graph.Grad(f, [a, b, c])

        self.assertSequenceEqual(run(g), [6, 9, 3])

