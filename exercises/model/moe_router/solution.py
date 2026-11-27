import math

import torch


def router_probs(logits):
    return torch.softmax(logits, dim=-1)


def top_k_routing(logits, k):
    probs = router_probs(logits)
    # stable 降序排序保证并列时下标小的在前（torch.topk 不承诺并列的顺序）
    w, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    w, idx = w[:, :k], idx[:, :k]
    return idx, w / w.sum(dim=-1, keepdim=True)


def expert_capacity(num_tokens, num_experts, k, capacity_factor):
    return int(math.ceil(capacity_factor * num_tokens * k / num_experts))


def dispatch(expert_idx, num_experts, capacity):
    T, k = expert_idx.shape
    idx = expert_idx.tolist()   # 逐个元素的 python 循环，先转成 list 比逐个取 tensor 元素快
    assignments = [[] for _ in range(num_experts)]
    dropped = torch.zeros(T, k, dtype=torch.bool, device=expert_idx.device)
    for j in range(k):          # slot 优先
        for t in range(T):      # slot 内按 token 顺序
            e = idx[t][j]
            if len(assignments[e]) < capacity:
                assignments[e].append(t)
            else:
                dropped[t, j] = True
    return assignments, dropped


def load_balancing_loss(logits, expert_idx, num_experts):
    f = torch.bincount(expert_idx.flatten(), minlength=num_experts).to(logits.dtype) / expert_idx.numel()
    P = router_probs(logits).mean(dim=0)
    return num_experts * (f * P).sum()  # f 是常数（计数），梯度只走 P


def moe_forward(x, logits, experts, k, capacity):
    idx, w = top_k_routing(logits, k)
    _, dropped = dispatch(idx, len(experts), capacity)
    y = torch.zeros_like(x)
    for e, fn in enumerate(experts):
        sel = (idx == e) & ~dropped                   # (T, k) bool
        rows = sel.any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        out = fn(x[rows])                             # 每个专家一次处理自己的全部 token
        gate = (w * sel).sum(dim=1)[rows]
        y.index_add_(0, rows, gate[:, None] * out)
    return y
