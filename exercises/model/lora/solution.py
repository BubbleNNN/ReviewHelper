import numpy as np


class LoRALinear:
    def __init__(self, W, r, alpha, rng):
        self.W = np.array(W, dtype=np.float64, copy=True)
        out_f, in_f = self.W.shape
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = rng.normal(0.0, 1.0 / r, size=(r, in_f))
        self.B = np.zeros((out_f, r))
        self.merged = False

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        y = x @ self.W.T
        if not self.merged:
            y = y + self.scaling * (x @ self.A.T) @ self.B.T
        return y

    def merge(self):
        if not self.merged:
            self.W = self.W + self.scaling * self.B @ self.A
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.W = self.W - self.scaling * self.B @ self.A
            self.merged = False

    def num_trainable_params(self):
        return int(self.A.size + self.B.size)
