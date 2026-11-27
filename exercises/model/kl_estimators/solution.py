import torch


def kl_penalty(logprob, ref_logprob, kind):
    if kind in ("kl", "k1"):
        return logprob - ref_logprob
    if kind == "abs":
        return (logprob - ref_logprob).abs()
    if kind in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if kind in ("low_var_kl", "k3"):
        u = torch.clamp(ref_logprob - logprob, -20.0, 20.0)  # u = log r，r = π_ref / π
        return torch.clamp(torch.exp(u) - u - 1.0, -10.0, 10.0)
    raise ValueError(f"unknown kl kind: {kind}")


def exact_kl(p, q):
    # xlogy(x, y) = x·log y，并且 x=0 时恰好返回 0，正好对应 0·log0 = 0 的约定
    return (torch.xlogy(p, p) - torch.xlogy(p, q)).sum()
