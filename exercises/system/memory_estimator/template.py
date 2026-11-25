def zero_model_states_bytes(num_params, dp, stage, k=12):
    """混合精度 Adam 下每卡模型状态显存（字节）。stage ∈ {0,1,2,3}。"""
    raise NotImplementedError


def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    """KV cache 字节数（K、V 各一份）。"""
    raise NotImplementedError


def max_concurrent_seqs(free_bytes, num_layers, num_kv_heads, head_dim, seq_len, dtype_bytes=2, block_size=None):
    """free_bytes 能放下多少条长度为 seq_len 的序列（向下取整）。

    block_size 不为 None 时，按块分配：每条序列占 ceil(seq_len / block_size) 块。
    """
    raise NotImplementedError


def mfu(num_params, tokens_per_sec, num_gpus, peak_flops_per_gpu):
    """Model FLOPs Utilization，训练 FLOPs 按 6·N·tokens 计。"""
    raise NotImplementedError
