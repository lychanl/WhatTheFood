from whatthefood.graph.node import Node


class ReLU(Node):
    def __init__(self, x, alpha=0.):
        super(ReLU, self).__init__(x.shape, x.batched, x)

        self.alpha = alpha

    def do(self, x):
        return (x >= 0) * x + (x < 0) * self.alpha * x

    def backpropagate(self, grad, x):
        return (x >= 0) * grad + (x < 0) * self.alpha * grad,


def leaky_relu(alpha=0.2):
    def builder(x):
        return ReLU(x, alpha)

    return builder
