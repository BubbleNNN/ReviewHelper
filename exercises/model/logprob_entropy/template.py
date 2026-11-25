import numpy as np


def logprobs_from_logits(logits, labels):
    """logits: (B, T, V)，labels: (B, T) int -> (B, T)，每个位置 label 的 log 概率。"""
    raise NotImplementedError


def entropy_from_logits(logits):
    """(B, T, V) -> (B, T)，H = -Σ_v p_v log p_v。"""
    raise NotImplementedError


def response_logprobs(logits, input_ids, response_length):
    """整条序列前向的 logits (B, T, V) 与 input_ids (B, T)，返回 response 部分的 logprob (B, R)。

    logits[:, t] 预测的是 input_ids[:, t+1]。
    """
    raise NotImplementedError


def masked_mean(x, mask):
    """只在 mask=1 的位置上求均值，返回 float。"""
    raise NotImplementedError
