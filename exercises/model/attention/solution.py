import math

import torch


def scaled_dot_product_attention(q, k, v, causal=False, key_padding_mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)  # (B, H, Tq, Tk)
    Tq, Tk = scores.shape[-2:]

    allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device)
    if causal:
        i = torch.arange(Tq, device=q.device)[:, None]
        j = torch.arange(Tk, device=q.device)[None, :]
        allowed = j <= i + (Tk - Tq)  # 右下角对齐
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[:, None, None, :]  # (B, 1, 1, Tk)，True=有效

    scores = scores.masked_fill(~allowed, float("-inf"))  # 广播到 (B, H, Tq, Tk)
    attn = torch.softmax(scores, dim=-1)  # 每行至少一个可见 key，-inf 处 exp 后严格为 0
    return attn @ v, attn
