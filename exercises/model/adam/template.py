import torch


def adamw_step(param, grad, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """AdamW 单步更新，对齐 torch.optim.AdamW（不许直接调用 torch.optim）。

    Args:
        param: 任意形状的 float tensor，当前参数
        grad:  同形状、同 dtype 的梯度 tensor
        state: dict，首次为空 {}；初始化为 {"step": 0, "m": zeros, "v": zeros} 并原地更新。
               "step" 是 python int，"m" / "v" 是与 param 同形状、同 dtype 的 tensor
        lr, betas, eps, weight_decay: python float（betas 是二元 tuple）
    Returns:
        更新后的参数 tensor，形状、dtype、device 与 param 相同（可以新建，不要求原地修改 param）
    """
    raise NotImplementedError
