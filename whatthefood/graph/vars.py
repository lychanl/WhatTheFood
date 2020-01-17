from whatthefood.graph.node import Node

import numpy as np


class Variable(Node):
    def __init__(self, shape):
        super(Variable, self).__init__(shape, False)

        self.value = np.ndarray(shape)

    def update_from_tf(self, sess):
        self.value = sess.run(self.tensor)

    def do(self):
        return self.value

    def backpropagate(self, grad):
        return ()

    def _build_tf(self, tf):
        return tf.Variable(self.value, dtype=tf.float32)

    def initialize(self, tf, sess):
        sess.run(tf.compat.v1.variables_initializer([self.tensor]))


class Placeholder(Node):
    def __init__(self, shape, batched):
        super(Placeholder, self).__init__(shape, batched)

    def backpropagate(self, grad):
        return ()

    def _build_tf(self, tf):
        return tf.compat.v1.placeholder(dtype=tf.float32, shape=self.shape if not self.batched else (None, *self.shape))


class Constant(Node):
    def __init__(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        super(Constant, self).__init__(np.shape(value), False)

        self.value = value

    def do(self):
        return self.value

    def backpropagate(self, grad):
        return ()

    def _build_tf(self, tf):
        return self.value
