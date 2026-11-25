def zero_model_states_bytes(num_params, dp, stage, k=12):
    psi = num_params
    if stage == 0:
        return (2 + 2 + k) * psi
    if stage == 1:
        return 2 * psi + 2 * psi + k * psi / dp
    if stage == 2:
        return 2 * psi + (2 + k) * psi / dp
    if stage == 3:
        return (2 + 2 + k) * psi / dp
    raise ValueError(f"unknown ZeRO stage {stage}")


def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch * dtype_bytes


def max_concurrent_seqs(free_bytes, num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2, block_size=None):
    per_token = kv_cache_bytes(num_layers, num_kv_heads, head_dim, 1, 1, dtype_bytes)
    if block_size is None:
        tokens_per_seq = seq_len
    else:
        n_blocks = -(-seq_len // block_size)  # ceil
        tokens_per_seq = n_blocks * block_size
    return int(free_bytes // (per_token * tokens_per_seq))


def mfu(num_params, tokens_per_sec, num_gpus, peak_flops_per_gpu):
    return 6 * num_params * tokens_per_sec / (num_gpus * peak_flops_per_gpu)
