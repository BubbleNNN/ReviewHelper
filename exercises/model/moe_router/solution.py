import math

import numpy as np


def router_probs(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def top_k_routing(logits, k):
    probs = router_probs(logits)
    idx = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
    w = np.take_along_axis(probs, idx, axis=-1)
    return idx, w / w.sum(axis=-1, keepdims=True)


def expert_capacity(num_tokens, num_experts, k, capacity_factor):
    return int(math.ceil(capacity_factor * num_tokens * k / num_experts))


def dispatch(expert_idx, num_experts, capacity):
    expert_idx = np.asarray(expert_idx)
    T, k = expert_idx.shape
    assignments = [[] for _ in range(num_experts)]
    dropped = np.zeros((T, k), dtype=bool)
    for j in range(k):          # slot 优先
        for t in range(T):      # slot 内按 token 顺序
            e = int(expert_idx[t, j])
            if len(assignments[e]) < capacity:
                assignments[e].append(t)
            else:
                dropped[t, j] = True
    return assignments, dropped


def load_balancing_loss(logits, expert_idx, num_experts):
    expert_idx = np.asarray(expert_idx)
    f = np.bincount(expert_idx.ravel(), minlength=num_experts) / expert_idx.size
    P = router_probs(logits).mean(axis=0)
    return float(num_experts * np.sum(f * P))


def moe_forward(x, logits, experts, k, capacity):
    x = np.asarray(x, dtype=np.float64)
    idx, w = top_k_routing(logits, k)
    _, dropped = dispatch(idx, len(experts), capacity)
    y = np.zeros_like(x)
    for e, fn in enumerate(experts):
        sel = (idx == e) & ~dropped                  # (T, k)
        rows = np.nonzero(sel.any(axis=1))[0]
        if rows.size == 0:
            continue
        out = fn(x[rows])                             # 每个专家一次处理自己的全部 token
        gate = (w * sel).sum(axis=1)[rows]
        y[rows] += gate[:, None] * out
    return y
