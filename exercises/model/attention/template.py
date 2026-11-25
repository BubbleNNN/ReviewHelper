import numpy as np


def scaled_dot_product_attention(q, k, v, causal=False, key_padding_mask=None):
    """缩放点积注意力。

    Args:
        q: (B, H, Tq, d)
        k: (B, H, Tk, d)
        v: (B, H, Tk, dv)
        causal: True 时右下角对齐：query i 只能看 key j <= i + (Tk - Tq)
        key_padding_mask: (B, Tk) bool，True=有效，False=padding；或 None
    Returns:
        out:  (B, H, Tq, dv)
        attn: (B, H, Tq, Tk)，被 mask 的位置为 0，每行和为 1
    """
    raise NotImplementedError
