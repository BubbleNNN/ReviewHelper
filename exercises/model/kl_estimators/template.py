import torch


def kl_penalty(logprob, ref_logprob, kind):
    """逐元素 KL 估计量，对齐 verl kl_penalty_forward。

    Args:
        logprob:     任意形状的 float tensor，当前策略的 log π(x)（可能带梯度）
        ref_logprob: 同形状 float tensor，参考模型的 log π_ref(x)
        kind ∈ {"kl","k1","abs","mse","k2","low_var_kl","k3"}，其他抛 ValueError。
            k3: u = clamp(ref - logprob, -20, 20)，结果 clamp(exp(u) - u - 1, -10, 10)。
    Returns:
        与输入同形状、同 dtype 的 tensor，可以对 logprob 反传（全程用 torch 运算）。
    """
    raise NotImplementedError


def exact_kl(p, q):
    """离散分布的 KL(p || q) = Σ p log(p/q)，p=0 的项记为 0。

    Args:
        p, q: (V,) float tensor，概率
    Returns:
        0-d tensor，dtype 与 p 相同。不要直接调用 F.kl_div / torch.distributions.kl_divergence。
    """
    raise NotImplementedError
