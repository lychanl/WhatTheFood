import numpy as np
import pickle

from whatthefood.train import Minimizer


class ADAM(Minimizer):
    def __init__(self, model, loss, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, limit=None):
        super(ADAM, self).__init__(model, loss)
        self.lr = lr

        self.m = [np.zeros_like(v.value) for v in self.vars]
        self.v = [np.zeros_like(v.value) for v in self.vars]

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.limit = limit

    def _run(self, grads, lr_decay=1., *args, **kwargs):
        for i, (var, g) in enumerate(zip(self.vars, grads)):
            self.m[i] = self.m[i] * self.beta1 + g * (1 - self.beta1)
            self.v[i] = self.v[i] * self.beta2 + np.square(g) * (1 - self.beta2)

            m_corr = self.m[i] / (1. - self.beta1)
            v_corr = self.v[i] / (1. - self.beta2)

            change = m_corr / (np.sqrt(v_corr) + self.eps)
            if self.limit:
                change = np.clip(change, -self.limit, self.limit)

            var.value -= self.lr * lr_decay * change

    def store(self, file):
        pickle.dump((self.m, self.v), file)

    def restore(self, file):
        self.m, self.v = pickle.load(file)
