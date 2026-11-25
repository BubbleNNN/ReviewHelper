def prefix_survival(conf):
    """a[r][j-1] = c[r][0] * c[r][1] * ... * c[r][j-1]（前 j 个 draft token 全部被接受的概率）。"""
    raise NotImplementedError


def greedy_schedule(conf, sps):
    """DSpark Hardware-Aware Prefix Scheduler（Algorithm 1）。

    Args:
        conf: 长度 R 的 list，conf[r] 是请求 r 各位置的条件置信度 c_{r,1..γ_r}
        sps:  函数 B -> 引擎在验证 batch 大小为 B（token 数）时每秒能跑的 step 数
    Returns:
        lengths: 每个请求验证的 draft 长度 l_r（0..γ_r）
        theta:   对应的 Θ = τ · SPS(B)
    """
    raise NotImplementedError


def brute_force_schedule(conf, sps):
    """枚举所有 (l_1..l_R) 组合，返回 Θ 最大的 (lengths, theta)。只用于小规模对拍。"""
    raise NotImplementedError
