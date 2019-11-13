from whatthefood.graph.node import Node

from collections.abc import Sequence
import numpy as np


class Grad(Node):
    def __init__(self, y, xs):
        if not isinstance(xs, Sequence):
            xs = [xs]

        self.outputs_graph = self._build_outputs_graph(y, xs)
        self.outputs_graph_flattened = self._flatten_outputs_graph(self.outputs_graph)

        super(Grad, self).__init__(None, False, y, *self.outputs_graph_flattened)

        self.y = y
        self.xs = xs

    def do(self, *inputs):
        values = {n: v for n, v in zip(self.inputs, inputs)}
        inputs_grads = {}

        ret = [None for _ in self.xs]

        for op in self.outputs_graph_flattened:
            if op is self.y:
                grad = np.ones_like(values[self.y])
            else:
                grad = np.sum([
                    inputs_grads[o][o.inputs.index(op)]
                    for o in self.outputs_graph[op]
                ], axis=0)

            input_values = [values[i] for i in op.inputs]
            inputs_grads[op] = op.backpropagate(grad, *input_values)

            if op in self.xs:
                ret[self.xs.index(op)] = grad

        return ret

    def _build_outputs_graph(self, y, xs):
        outputs = {}
        keep = y in xs
        for i in y.inputs:
            if i in xs:
                keep = True

            from_i = self._build_outputs_graph(i, xs)

            if from_i is not None:
                keep = True

                for k, v in from_i.items():
                    if k in outputs:
                        outputs[k].extend(v)
                    else:
                        outputs[k] = v

                if i in outputs:
                    outputs[i].append(y)
                else:
                    outputs[i] = [y]

        if y not in outputs:
            outputs[y] = []
        return outputs if keep else None

    def _flatten_outputs_graph(self, outputs):
        flattened = []
        outputs = {k: list(v) for k, v in outputs.items()}

        while outputs:
            lasts = [k for k, v in outputs.items() if not v]

            for l in lasts:
                outputs.pop(l)
            for l in lasts:
                for v in outputs.values():
                    if l in v:
                        v.remove(l)

            flattened.extend(lasts)

        return flattened
