import numpy as np


def _logsumexp(z):
    mx = z.max(axis=-1, keepdims=True)
    return (mx + np.log(np.exp(z - mx).sum(axis=-1, keepdims=True)))[..., 0]


def logprobs_from_logits(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    picked = np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    return picked - _logsumexp(logits)


def entropy_from_logits(logits):
    logits = np.asarray(logits, dtype=np.float64)
    lse = _logsumexp(logits)
    p = np.exp(logits - lse[..., None])
    return lse - (p * logits).sum(axis=-1)


def response_logprobs(logits, input_ids, response_length):
    R = response_length
    return logprobs_from_logits(np.asarray(logits)[:, -R - 1:-1], np.asarray(input_ids)[:, -R:])


def masked_mean(x, mask):
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    return float((x * mask).sum() / mask.sum())
