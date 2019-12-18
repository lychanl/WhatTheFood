import unittest

from whatthefood.graph import Constant, MultiplyByScalar, Placeholder, Sum, Variable
from whatthefood.nn import layers, Model


class TestModel(unittest.TestCase):
    def test_inputs(self):
        model = Model()

        self.assertEqual(len(model.inputs), 0)

        model.add(Placeholder, (1, 2), False)
        model.add(Placeholder, (2, 2), True, current_output_as_argument=False)

        self.assertEqual(len(model.inputs), 2)

        self.assertIsInstance(model.inputs[0], Placeholder)
        self.assertEqual(model.inputs[0].shape, (1, 2))
        self.assertEqual(model.inputs[0].batched, False)

        self.assertIsInstance(model.inputs[1], Placeholder)
        self.assertEqual(model.inputs[1].shape, (2, 2))
        self.assertEqual(model.inputs[1].batched, True)

    def test_output(self):
        model = Model()

        model.add(Placeholder, (), False)
        model.add(Sum, Constant(3))

        self.assertEqual(model(4), 7)

        model.add(MultiplyByScalar, Constant(4))

        self.assertEqual(model(4), 28)

    def test_variables(self):
        model = Model()

        self.assertEqual(len(model.variables), 0)

        model.add(Variable, (1, 2))

        self.assertEqual(len(model.variables), 1)

    def test_layers(self):
        model = Model()

        self.assertEqual(len(model.layers), 0)
        self.assertEqual(len(model.variables), 0)

        model.add(Placeholder, (3,), True)
        model.add(layers.Dense, 5)

        self.assertEqual(len(model.layers), 1)
        self.assertEqual(len(model.variables), len(model.layers[0].vars))

        self.assertSequenceEqual(model.variables, model.layers[0].vars)
