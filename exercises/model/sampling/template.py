import numpy as np


def top_k_filter(probs, k):
    """保留概率 >= 第 k 大概率的 token（并列全保留），重新归一化。k<=0 或 k>=V 不过滤。

    Args:
        probs: (V,) 概率
    Returns:
        (V,) 过滤并归一化后的概率
    """
    raise NotImplementedError


def top_p_filter(probs, p):
    """降序（稳定排序）后保留累计概率第一次 >= p 的最短前缀，重新归一化。p>=1 不过滤。"""
    raise NotImplementedError


def process_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    """temperature -> softmax -> top-k -> top-p，返回 (V,) 概率。temperature<1e-5 返回 argmax one-hot。"""
    raise NotImplementedError


def sample(logits, rng, temperature=1.0, top_k=0, top_p=1.0):
    """用 rng (np.random.Generator) 从 process_logits 的结果里采一个 token id，返回 int。"""
    raise NotImplementedError
