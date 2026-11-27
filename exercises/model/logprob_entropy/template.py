import torch


def logprobs_from_logits(logits, labels):
    """每个位置 label 的 log 概率。不要调用 F.cross_entropy / F.nll_loss / torch.distributions.Categorical。

    Args:
        logits: (B, T, V) float tensor
        labels: (B, T) long tensor，取值在 [0, V)
    Returns:
        (B, T) tensor，dtype 与 logits 相同
    """
    raise NotImplementedError


def entropy_from_logits(logits):
    """H = -Σ_v p_v log p_v。不要调用 torch.distributions.Categorical(...).entropy()。

    Args:
        logits: (B, T, V) float tensor
    Returns:
        (B, T) tensor，dtype 与 logits 相同
    """
    raise NotImplementedError


def response_logprobs(logits, input_ids, response_length):
    """返回 response 部分每个 token 的 logprob。

    Args:
        logits: (B, T, V) float tensor，整条 prompt + response 序列前向的输出；
                logits[:, t] 预测的是 input_ids[:, t+1]
        input_ids: (B, T) long tensor，最后 R 个是 response
        response_length: python int R
    Returns:
        (B, R) tensor，dtype 与 logits 相同
    """
    raise NotImplementedError


def masked_mean(x, mask):
    """只在 mask=True 的位置上求均值。

    Args:
        x: 任意形状的 float tensor（可能带梯度）
        mask: 与 x 同形状的 bool tensor，True=参与平均，False=忽略（padding）
    Returns:
        0-d tensor，dtype 与 x 相同，保留梯度（之后要当 loss 反传）
    """
    raise NotImplementedError
