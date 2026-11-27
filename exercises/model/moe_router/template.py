import math

import torch


def router_probs(logits):
    """对专家维 softmax。

    Args:
        logits: (T, E) float tensor
    Returns:
        (T, E) tensor，每行和为 1，dtype 与 logits 相同
    """
    raise NotImplementedError


def top_k_routing(logits, k):
    """全 softmax 后取 top-k（降序，并列取小下标），权重重新归一化。

    Args:
        logits: (T, E) float tensor；k: python int
    Returns:
        expert_idx: (T, k) long tensor
        weights:    (T, k) tensor，每行和为 1，dtype 与 logits 相同
    """
    raise NotImplementedError


def expert_capacity(num_tokens, num_experts, k, capacity_factor):
    """ceil(capacity_factor * num_tokens * k / num_experts)，返回 python int。"""
    raise NotImplementedError


def dispatch(expert_idx, num_experts, capacity):
    """按 slot 优先（先所有 token 的第 1 选择，再第 2 选择…）、slot 内按 token 顺序分配。

    Args:
        expert_idx: (T, k) long tensor；num_experts, capacity: python int
    Returns:
        assignments: 长度 num_experts 的 python list，assignments[e] 是 python int 的 token 下标列表（按加入顺序）
        dropped:     (T, k) bool tensor，True 表示这个 (token, slot) 被丢弃
    """
    raise NotImplementedError


def load_balancing_loss(logits, expert_idx, num_experts):
    """E * Σ_i f_i * P_i；f_i = expert_idx 中 i 的个数 / (T*k)，P_i = router 概率在 token 上的均值。

    Args:
        logits: (T, E) float tensor（可能带梯度）；expert_idx: (T, k) long tensor
    Returns:
        0-d tensor，dtype 与 logits 相同。它要加进训练 loss：梯度只经 P 回到 logits，f 是计数、不可导。
    """
    raise NotImplementedError


def moe_forward(x, logits, experts, k, capacity):
    """token-choice MoE 前向：top-k 路由 -> dispatch（超容量的 slot 丢弃）-> 按权重合并。

    Args:
        x:       (T, d) float tensor
        logits:  (T, E) float tensor，dtype 与 x 相同
        experts: 长度 E 的函数列表，f(rows: (n, d) tensor) -> (n, d) tensor
        k, capacity: python int
    Returns:
        (T, d) tensor，dtype 与 x 相同；被丢弃的 slot 贡献 0
    """
    raise NotImplementedError
