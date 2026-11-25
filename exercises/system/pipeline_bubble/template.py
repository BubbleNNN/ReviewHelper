def stage_order(schedule, p, m, stage):
    """返回某个 stage 的操作顺序：list of ("F"|"B", micro_batch_id)。"""
    raise NotImplementedError


def simulate(schedule, p, m, tf=1.0, tb=2.0):
    """模拟流水线调度。

    Returns: {"timeline", "total_time", "bubble_ratio", "peak_inflight"}，含义见题面。
    """
    raise NotImplementedError
