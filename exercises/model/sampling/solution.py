import torch


def top_k_filter(probs, k):
    V = probs.shape[-1]
    if k <= 0 or k >= V:
        return probs / probs.sum()
    kth = torch.topk(probs, k).values[-1]  # 第 k 大的概率
    out = torch.where(probs >= kth, probs, torch.zeros_like(probs))
    return out / out.sum()


def top_p_filter(probs, p):
    if p >= 1.0:
        return probs / probs.sum()
    sorted_p, idx = torch.sort(probs, descending=True, stable=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    # 去掉“前面的 token 已经凑够 p”的位置；越过 p 的那个 token 本身保留，所以至少剩 1 个
    remove = (cum - sorted_p) >= p
    sorted_p = sorted_p.masked_fill(remove, 0.0)
    out = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
    return out / out.sum()


def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature < 1e-5:
        out = torch.zeros_like(logits)
        out[torch.argmax(logits)] = 1.0  # 并列时 argmax 返回第一个
        return out
    probs = torch.softmax(logits / temperature, dim=-1)
    probs = top_k_filter(probs, top_k)
    probs = top_p_filter(probs, top_p)
    return probs


def sample(logits, generator, temperature=1.0, top_k=0, top_p=1.0):
    probs = process_logits(logits, temperature, top_k, top_p)
    return int(torch.multinomial(probs, 1, generator=generator).item())
