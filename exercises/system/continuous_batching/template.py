def simulate_static(requests, max_batch, token_budget):
    """Static batching：一批全部结束后才能开下一批。

    requests: list of (arrival_step, prompt_len, output_len)
    Returns: {"start", "finish", "tokens_per_step", "running_per_step", "throughput"}
    """
    raise NotImplementedError


def simulate_continuous(requests, max_batch, token_budget):
    """Continuous (iteration-level) batching：每个 step 先保证 decode，再按 FCFS 接纳新请求。"""
    raise NotImplementedError
