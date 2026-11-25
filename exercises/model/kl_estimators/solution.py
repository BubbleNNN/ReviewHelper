import numpy as np


def kl_penalty(logprob, ref_logprob, kind):
    logprob = np.asarray(logprob, dtype=np.float64)
    ref_logprob = np.asarray(ref_logprob, dtype=np.float64)
    if kind in ("kl", "k1"):
        return logprob - ref_logprob
    if kind == "abs":
        return np.abs(logprob - ref_logprob)
    if kind in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob) ** 2
    if kind in ("low_var_kl", "k3"):
        u = np.clip(ref_logprob - logprob, -20.0, 20.0)  # u = log r，r = π_ref / π
        return np.clip(np.exp(u) - u - 1.0, -10.0, 10.0)
    raise ValueError(f"unknown kl kind: {kind}")


def exact_kl(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    nz = p > 0
    return float(np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz]))))
