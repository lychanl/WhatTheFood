from whatthefood.train import Minimizer


class SGD(Minimizer):
    def __init__(self, model, loss, lr=0.1):
        super(SGD, self).__init__(model, loss)
        self.lr = lr

    def _run(self, grads, lr_decay=1., *args, **kwargs):
        for v, g in zip(self.vars, grads):
            v.value -= g * self.lr * lr_decay

    def _build_tf_opt(self, grads, tf, sess, *args, **kwargs):
        ops = []
        for v, g in zip(self.vars, grads):
            ops.append(v.assign_sub(g * self.lr))

        return tf.group(ops)
