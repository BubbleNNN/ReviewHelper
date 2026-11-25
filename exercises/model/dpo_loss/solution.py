import numpy as np


def _logsigmoid(x):
    return -np.logaddexp(0.0, -x)  # log σ(x) = -log(1 + e^{-x})


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    pc, pr, rc, rr = (np.asarray(a, dtype=np.float64) for a in
                      (policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps))
    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)
    h = chosen_rewards - rejected_rewards
    losses = -(1 - label_smoothing) * _logsigmoid(h) - label_smoothing * _logsigmoid(-h)
    acc = float(np.mean(chosen_rewards > rejected_rewards))
    return float(losses.mean()), chosen_rewards, rejected_rewards, acc
