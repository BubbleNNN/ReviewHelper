import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    # 参考模型是冻结的：detach 掉，梯度只流向 policy
    chosen_logratios = policy_chosen_logps - ref_chosen_logps.detach()
    rejected_logratios = policy_rejected_logps - ref_rejected_logps.detach()
    h = beta * (chosen_logratios - rejected_logratios)
    # F.logsigmoid 内部是稳定的 -softplus(-h)，h = ±1e4 也不会溢出
    losses = -(1 - label_smoothing) * F.logsigmoid(h) - label_smoothing * F.logsigmoid(-h)
    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    acc = (chosen_rewards > rejected_rewards).to(h.dtype).mean().item()
    return losses.mean(), chosen_rewards, rejected_rewards, acc
