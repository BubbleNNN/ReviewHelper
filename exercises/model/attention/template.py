import math

import torch


def scaled_dot_product_attention(q, k, v, causal=False, key_padding_mask=None):
    """缩放点积注意力。不要调用 F.scaled_dot_product_attention。

    Args:
        q: (B, H, Tq, d) float tensor
        k: (B, H, Tk, d) float tensor
        v: (B, H, Tk, dv) float tensor
        causal: True 时右下角对齐：query i 只能看 key j <= i + (Tk - Tq)
        key_padding_mask: (B, Tk) bool tensor，True=有效 token，False=padding（要被 mask 掉）；或 None
    Returns:
        out:  (B, H, Tq, dv)，dtype 与 q 相同
        attn: (B, H, Tq, Tk)，被 mask 的位置严格为 0，每行和为 1
    """
    raise NotImplementedError
