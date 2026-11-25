import numpy as np


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """DPO sigmoid 损失（cDPO label smoothing）。

    Args:
        四个输入都是 (B,) 的序列级 log 概率
    Returns:
        loss: float（batch 平均）
        chosen_rewards:   (B,) = beta * (policy_chosen - ref_chosen)
        rejected_rewards: (B,) = beta * (policy_rejected - ref_rejected)
        reward_accuracy:  float，chosen_rewards > rejected_rewards 的比例
    """
    raise NotImplementedError
