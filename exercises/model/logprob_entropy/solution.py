import torch


def _logsumexp(z):
    mx = torch.amax(z, dim=-1, keepdim=True)
    return (mx + torch.log(torch.exp(z - mx).sum(dim=-1, keepdim=True))).squeeze(-1)


def logprobs_from_logits(logits, labels):
    # 只取被选中 token 的 logit，再减整行的 logsumexp；不必物化 (B, T, V) 的 log_softmax
    picked = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return picked - _logsumexp(logits)


def entropy_from_logits(logits):
    lse = _logsumexp(logits)
    p = torch.exp(logits - lse.unsqueeze(-1))
    return lse - (p * logits).sum(dim=-1)  # H = lse - Σ p z


def response_logprobs(logits, input_ids, response_length):
    R = response_length
    # logits[:, t] 预测 input_ids[:, t+1]：response 的 R 个 token 用它们前一位的 logits
    return logprobs_from_logits(logits[:, -R - 1:-1], input_ids[:, -R:])


def masked_mean(x, mask):
    mask = mask.to(x.dtype)  # bool -> float
    return (x * mask).sum() / mask.sum()
