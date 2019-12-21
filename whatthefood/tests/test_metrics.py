import unittest

import numpy as np

import whatthefood.graph as graph
import whatthefood.classification.metrics as metrics


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        classes = graph.Constant([1, 2, 3, 2, 0, 3])
        e_classes = graph.Constant([0, 2, 3, 2, 1, 3])

        acc = metrics.accuracy(e_classes, classes)
        self.assertEqual(4/6, graph.run(acc))

    def test_accuracy_with_flags(self):
        classes = graph.Constant([1, 2, 3, 2, 0, 3])
        e_classes = graph.Constant([0, 2, 3, 2, 1, 3])
        flags = graph.Constant([0, 1, 1, 1, 1, 0])

        acc = metrics.accuracy(e_classes, classes, flags)
        self.assertEqual(0.75, graph.run(acc))

    def test_tp_rate(self):
        classes = graph.Constant([1, 0, 0, 1, 1, 1])
        e_classes = graph.Constant([0, 1, 0, 1, 1, 1])

        pr = metrics.tp_rate(e_classes, classes)
        self.assertEqual(0.75, graph.run(pr))

    def test_fp_rate(self):
        classes = graph.Constant([1, 0, 0, 1, 1, 1, 1])
        e_classes = graph.Constant([0, 1, 0, 1, 1, 1, 0])

        pr = metrics.fp_rate(e_classes, classes)
        self.assertEqual(2 / 3, graph.run(pr))


