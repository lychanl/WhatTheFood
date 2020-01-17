import numpy as np
import pickle

from whatthefood.graph import Variable
from whatthefood.train import Minimizer


class ADAM(Minimizer):
    def __init__(self, model, loss, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, limit=None):
        super(ADAM, self).__init__(model, loss)
        self.lr = lr

        self.m = [Variable(v.shape) for v in self.vars]
        for m in self.m:
            m.value = np.zeros(m.shape)
        self.v = [Variable(v.shape) for v in self.vars]
        for v in self.v:
            v.value = np.zeros(v.shape)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.limit = limit

    def _run(self, grads, lr_decay=1., *args, **kwargs):
        for i, (var, g) in enumerate(zip(self.vars, grads)):
            self.m[i].value = self.m[i].value * self.beta1 + g * (1 - self.beta1)
            self.v[i].value = self.v[i].value * self.beta2 + np.square(g) * (1 - self.beta2)

            m_corr = self.m[i].value / (1. - self.beta1)
            v_corr = self.v[i].value / (1. - self.beta2)

            change = m_corr / (np.sqrt(v_corr) + self.eps)
            if self.limit:
                change = np.clip(change, -self.limit, self.limit)

            var.value -= self.lr * lr_decay * change

    def store(self, file):
        pickle.dump((self.m, self.v), file)

    def restore(self, file):
        self.m, self.v = pickle.load(file)

    def _build_tf_opt(self, grads, tf, sess, *args, **kwargs):
        ops = []

        for i, (var, g) in enumerate(zip(self.vars, grads)):
            m = self.m[i].build_tf(tf, sess)
            v = self.v[i].build_tf(tf, sess)

            m_new = m * self.beta1 + g * (1 - self.beta1)
            v_new = v * self.beta2 + tf.square(g) * (1 - self.beta2)

            ops.append(m.assign(m_new))
            ops.append(v.assign(v_new))

            m_corr = m / (1. - self.beta1)
            v_corr = v / (1. - self.beta2)

            change = m_corr / (tf.sqrt(v_corr) + self.eps)
            if self.limit:
                raise NotImplementedError

            ops.append(var.build_tf(tf, sess).assign_sub(self.lr * change))

        return tf.group(ops)

    def _update_internal_from_tf(self, sess):
        for m in self.m:
            m.update_from_tf(sess)
        for v in self.v:
            v.update_from_tf(sess)

    def _clear_internal_tf(self):
        for m in self.m:
            m.clear_tf()
        for v in self.v:
            v.clear_tf()

