import numpy as np


def kl_penalty(logprob, ref_logprob, kind):
    """逐元素 KL 估计量，对齐 verl kl_penalty_forward。

    kind ∈ {"kl","k1","abs","mse","k2","low_var_kl","k3"}，其他抛 ValueError。
    k3: u = clip(ref - logprob, -20, 20)，结果 clip(exp(u) - u - 1, -10, 10)。
    """
    raise NotImplementedError


def exact_kl(p, q):
    """离散分布的 KL(p || q) = Σ p log(p/q)，p=0 的项记为 0。返回 float。"""
    raise NotImplementedError
