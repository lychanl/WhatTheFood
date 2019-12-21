import unittest

import numpy as np

import whatthefood.graph as graph
import whatthefood.classification.metrics as metrics


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        classes = graph.Constant([1, 2, 3, 2, 0, 3])
        e_classes = graph.Constant([0, 2, 3, 2, 1, 3])

        acc = metrics.accuracy(classes, e_classes)
        self.assertEqual(4/6, graph.run(acc))

    def test_accuracy_with_flags(self):
        classes = graph.Constant([1, 2, 3, 2, 0, 3])
        e_classes = graph.Constant([0, 2, 3, 2, 1, 3])
        flags = graph.Constant([0, 1, 1, 1, 1, 0])

        acc = metrics.accuracy(classes, e_classes, flags)
        self.assertEqual(0.75, graph.run(acc))

    def test_precision(self):
        classes = graph.Constant([1, 0, 0, 1, 1, 1])
        e_classes = graph.Constant([0, 1, 0, 1, 1, 1])

        pr = metrics.precision(classes, e_classes)
        self.assertEqual(0.75, graph.run(pr))


