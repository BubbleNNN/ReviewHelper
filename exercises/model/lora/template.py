import numpy as np


class LoRALinear:
    """y = x W^T + (alpha / r) * x A^T B^T

    属性：W (out, in)、A (r, in)、B (out, r)、r、alpha、scaling、merged
    """

    def __init__(self, W, r, alpha, rng):
        """复制 W；A ~ N(0, 1/r)（rng.normal(0, 1/r, size=(r, in))），B = 0，scaling = alpha / r。"""
        raise NotImplementedError

    def forward(self, x):
        """x: (..., in) -> (..., out)。merged=True 时只用 W。"""
        raise NotImplementedError

    def merge(self):
        """W += scaling * B @ A；重复调用不重复叠加。"""
        raise NotImplementedError

    def unmerge(self):
        """撤销 merge；未合并时不做事。"""
        raise NotImplementedError

    def num_trainable_params(self):
        """A 与 B 的元素个数之和。"""
        raise NotImplementedError
