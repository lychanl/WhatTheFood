from collections.abc import Sequence

import numpy as np

from whatthefood.graph import Grad, Placeholder, run, Sum, Square, Multiply, Constant, ReduceSum
from whatthefood.graph.node import Node


class Minimizer(object):
    def __init__(self, model, loss, regularization=None):
        self.expected_output = Placeholder(model.output.shape, model.output.batched)
        self.loss = loss(self.expected_output, model.output)

        self.model = model

        self.metrics = [self.loss]

        self.vars = model.variables

        self.eval_tensor = None
        self.opt_tensor = None

        self.total_loss = Sum(self.loss, Multiply(self.create_regularization(), Constant(regularization)))\
            if regularization else self.loss
        self.grad = Grad(self.total_loss, self.model.variables)


    def create_regularization(self):
        regularization = None
        for v in self.vars:
            if regularization is None:
                regularization = ReduceSum(Square(v))
            else:
                regularization = Sum(ReduceSum(Square(v)), regularization)

        return regularization


    def get_input_dict(self, inputs, expected_output):
        input_dict = None
        if inputs is None:
            input_dict = {}
        elif isinstance(inputs, np.ndarray) and len(self.model.inputs) == 1:
            input_dict = {self.model.inputs[0]: inputs}
        elif isinstance(inputs, dict):
            input_dict = inputs
        elif isinstance(inputs, Sequence):
            input_dict = {i: v for i, v in zip(self.model.inputs, inputs)}

        input_dict[self.expected_output] = expected_output

        return input_dict

    def add_metrics(self, metrics):
        if isinstance(metrics, Node):
            self.metrics.append(metrics)
        elif isinstance(metrics, Sequence):
            for m in metrics:
                self.add_metrics(m)
        else:
            m = metrics(self.expected_output, self.model.output)
            if isinstance(m, Sequence):
                self.metrics.extend(m)
            else:
                self.metrics.append(m)

    def run(self, inputs, expected_output, *args, tf=None, tf_session=None, **kwargs):
        input_dict = self.get_input_dict(inputs, expected_output)

        if not tf_session or not tf:
            loss, grads = run((self.loss, self.grad), input_dict)

            self._run(grads, *args, **kwargs)

            return loss
        else:
            if kwargs:
                raise NotImplementedError
            input_dict = {p.build_tf(tf, tf_session): v for p, v in input_dict.items()}
            opt_tensor = self.get_tf_opt(tf, tf_session, *args, **kwargs)
            loss, _ = tf_session.run(opt_tensor, input_dict)
            return loss


    def _run(self, grads, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, ds, batch_size=None, tf_session=None):
        if not batch_size:
            batch_size = len(ds)
        total_metrics = [0 for _ in self.metrics]
        for i in range(0, len(ds), batch_size):
            size = min(batch_size, len(ds) - i)
            batch = slice(i, i + batch_size)

            input_dict = self.get_input_dict(ds.inputs[batch], ds.outputs[batch])
            m = run(self.metrics, input_dict, tf_sess=tf_session)
            for i, v in enumerate(m):
                total_metrics[i] += v * size

        return [v / len(ds) for v in total_metrics]

    def store(self, file):
        pass

    def restore(self, file):
        pass

    def get_tf_opt(self, tf, sess, *args, **kwargs):
        if self.opt_tensor is None:
            grad_ops = self.grad.build_tf(tf, sess)
            loss_op = self.loss.build_tf(tf, sess)
            self.opt_tensor = (loss_op, self._build_tf_opt(grad_ops, tf, sess, *args, **kwargs))

        return self.opt_tensor

    def _build_tf_opt(self, grads, tf, sess, *args, **kwargs):
        raise NotImplementedError

    def update_from_tf(self, sess):
        self.model.update_from_tf(sess)
        self._update_internal_from_tf(sess)

    def _update_internal_from_tf(self, sess):
        pass

    def clear_tf(self):
        self.model.clear_tf()
        self.opt_tensor = None

    def _clear_internal_tf(self):
        pass
