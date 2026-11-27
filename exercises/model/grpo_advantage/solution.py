import torch


def _groups(index):
    if torch.is_tensor(index):
        index = index.tolist()  # tensor 元素按 id 哈希，转成 python 数再分组
    g = {}
    for i, key in enumerate(index):
        g.setdefault(key, []).append(i)
    return [torch.tensor(ids, dtype=torch.long) for ids in g.values()]


@torch.no_grad()  # 优势是常数，不参与反传
def grpo_advantage(token_level_rewards, response_mask, index, norm_adv_by_std=True, epsilon=1e-6):
    scores = token_level_rewards.sum(dim=-1)
    adv = torch.zeros_like(scores)
    for ids in _groups(index):
        s = scores[ids]
        if len(ids) == 1:
            mean, std = s.new_zeros(()), s.new_ones(())  # 单条时 torch.std 会给 nan
        else:
            mean, std = s.mean(), s.std()  # torch.std 默认无偏（除以 n-1）
        adv[ids] = (s - mean) / (std + epsilon) if norm_adv_by_std else s - mean
    return adv[:, None] * response_mask.to(adv.dtype)


@torch.no_grad()
def rloo_advantage(token_level_rewards, response_mask, index):
    scores = token_level_rewards.sum(dim=-1)
    adv = scores.clone()
    for ids in _groups(index):
        n = len(ids)
        if n > 1:
            s = scores[ids]
            adv[ids] = s - (s.sum() - s) / (n - 1)
    return adv[:, None] * response_mask.to(adv.dtype)
