import torch


def top_k_filter(probs, k):
    """保留概率 >= 第 k 大概率的 token（并列全保留），重新归一化。k<=0 或 k>=V 不过滤。

    Args:
        probs: (V,) float tensor，概率
        k: int
    Returns:
        (V,) 过滤并归一化后的概率，dtype 与输入相同
    """
    raise NotImplementedError


def top_p_filter(probs, p):
    """降序（稳定排序）后保留累计概率第一次 >= p 的最短前缀，重新归一化。p>=1 不过滤。

    Args:
        probs: (V,) float tensor；p: python float
    Returns:
        (V,) 过滤并归一化后的概率
    """
    raise NotImplementedError


def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """temperature -> softmax -> top-k -> top-p，返回 (V,) 概率。temperature<1e-5 返回 argmax one-hot。"""
    raise NotImplementedError


def sample(logits, generator, temperature=1.0, top_k=0, top_p=1.0):
    """从 process_logits 的结果里采一个 token id，返回 python int。

    随机数只能来自传入的 generator（torch.Generator），例如
    torch.multinomial(probs, 1, generator=generator)。
    """
    raise NotImplementedError
