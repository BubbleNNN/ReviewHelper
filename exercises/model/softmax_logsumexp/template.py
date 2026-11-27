import torch


def logsumexp(x, dim=-1, keepdim=False):
    """数值稳定的 log Σ exp(x)。不要调用 torch.logsumexp。

    Args:
        x: 任意形状的 float tensor，可以包含 -inf
        dim: 归约的维度（python int）
        keepdim: 是否保留被归约的维度（长度为 1），语义同 torch.logsumexp
    Returns:
        归约后的 tensor，dtype 与 x 相同；一维输入且 keepdim=False 时是 0-d tensor。
        整行 -inf 时结果为 -inf（不能是 nan）。
    """
    raise NotImplementedError


def softmax(x, dim=-1):
    """数值稳定的 softmax，-inf 位置输出 0。不要调用 torch.softmax / F.softmax。

    Returns:
        形状、dtype 与 x 相同的 tensor
    """
    raise NotImplementedError


def log_softmax(x, dim=-1):
    """x - logsumexp(x)，-inf 位置输出 -inf。不要调用 torch.log_softmax / F.log_softmax。

    Returns:
        形状、dtype 与 x 相同的 tensor
    """
    raise NotImplementedError
