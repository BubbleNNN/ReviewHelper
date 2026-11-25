import numpy as np


def router_probs(logits):
    """(T, E) -> (T, E)，对专家维 softmax。"""
    raise NotImplementedError


def top_k_routing(logits, k):
    """全 softmax 后取 top-k（降序，并列取小下标），权重重新归一化。

    Returns:
        expert_idx: (T, k) int
        weights:    (T, k)，每行和为 1
    """
    raise NotImplementedError


def expert_capacity(num_tokens, num_experts, k, capacity_factor):
    """ceil(capacity_factor * num_tokens * k / num_experts)，返回 int。"""
    raise NotImplementedError


def dispatch(expert_idx, num_experts, capacity):
    """按 slot 优先（先所有 token 的第 1 选择，再第 2 选择…）、slot 内按 token 顺序分配。

    Returns:
        assignments: 长度 num_experts 的 list，assignments[e] 是 token 下标列表（按加入顺序）
        dropped:     (T, k) bool，被丢弃的 (token, slot)
    """
    raise NotImplementedError


def load_balancing_loss(logits, expert_idx, num_experts):
    """E * Σ_i f_i * P_i；f_i = expert_idx 中 i 的个数 / (T*k)，P_i = router 概率在 token 上的均值。返回 float。"""
    raise NotImplementedError


def moe_forward(x, logits, experts, k, capacity):
    """x: (T, d)，experts: 长度 E 的函数列表 f(rows) -> 同形状。被丢弃的 slot 贡献 0。返回 (T, d)。"""
    raise NotImplementedError
