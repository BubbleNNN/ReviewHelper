import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """DPO sigmoid 损失（cDPO label smoothing）。

    Args:
        四个输入都是 (B,) float tensor，序列级 log 概率。
        policy_* 带梯度；ref_* 视为常数（detach），不该收到梯度。
        beta, label_smoothing: python float
    Returns:
        loss: 0-d tensor（batch 平均），dtype 与输入相同，梯度只流向两个 policy 输入
        chosen_rewards:   (B,) tensor = beta * (policy_chosen - ref_chosen)，detach 过（日志用）
        rejected_rewards: (B,) tensor = beta * (policy_rejected - ref_rejected)，detach 过
        reward_accuracy:  python float，chosen_rewards > rejected_rewards 的比例
    """
    raise NotImplementedError
