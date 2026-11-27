import torch


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm 前向（在最后一维归一化，方差除以 D）。

    不要调用 F.layer_norm / torch.nn.LayerNorm。

    Args:
        x:     (N, D) float tensor
        gamma: (D,) float tensor
        beta:  (D,) float tensor
        eps:   python float
    Returns:
        y:     (N, D) tensor，dtype 与 x 相同
        cache: 反向需要的中间量（内容自定，比如一个 tensor 组成的 tuple）
    """
    raise NotImplementedError


def layernorm_backward(dy, cache):
    """LayerNorm 反向，手推公式实现。

    不能用 autograd（不要调 .backward() / torch.autograd.grad），测试会在 torch.no_grad() 下调用它。

    Args:
        dy:    (N, D) float tensor，上游梯度
        cache: layernorm_forward 返回的 cache
    Returns:
        dx:     (N, D) tensor
        dgamma: (D,) tensor
        dbeta:  (D,) tensor
        dtype 都与 dy 相同。
    """
    raise NotImplementedError


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm：x / sqrt(mean(x^2) + eps) * weight。不要调用 F.rms_norm / torch.nn.RMSNorm。

    Args:
        x:      (..., D) float tensor
        weight: (D,) float tensor
        eps:    python float
    Returns:
        (..., D) tensor，dtype 与 x 相同
    """
    raise NotImplementedError
