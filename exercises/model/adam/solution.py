import numpy as np


def adamw_step(param, grad, state, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    if not state:
        state["step"] = 0
        state["m"] = np.zeros_like(param)
        state["v"] = np.zeros_like(param)
    b1, b2 = betas
    state["step"] += 1
    t = state["step"]

    param = param * (1 - lr * weight_decay)  # 解耦的权重衰减
    state["m"] = b1 * state["m"] + (1 - b1) * grad
    state["v"] = b2 * state["v"] + (1 - b2) * grad * grad
    m_hat = state["m"] / (1 - b1 ** t)
    v_hat = state["v"] / (1 - b2 ** t)
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)
