import numpy as np


def adamw_step(param, grad, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """AdamW 单步更新，对齐 torch.optim.AdamW。

    Args:
        param: 参数数组
        grad:  同形状梯度
        state: dict，首次为空；初始化为 {"step": 0, "m": zeros, "v": zeros} 并原地更新
    Returns:
        更新后的参数数组
    """
    raise NotImplementedError
