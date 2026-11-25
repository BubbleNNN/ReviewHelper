class BlockManager:
    """PagedAttention 风格的 KV cache 块管理器（引用计数 + 写时复制）。"""

    def __init__(self, num_blocks, block_size):
        raise NotImplementedError

    def allocate(self, seq_id, n_tokens):
        """为新序列分配 ceil(n_tokens / block_size) 个块。空间不足 MemoryError，重复 id ValueError。"""
        raise NotImplementedError

    def append_token(self, seq_id):
        """追加一个 token。需要写时复制时返回 (src_block, dst_block)，否则返回 None。"""
        raise NotImplementedError

    def fork(self, parent_id, child_id):
        """child 共享 parent 的全部块（引用计数 +1）。"""
        raise NotImplementedError

    def free(self, seq_id):
        """释放序列：引用计数 -1，降到 0 的块回到空闲池。"""
        raise NotImplementedError

    def block_table(self, seq_id):
        raise NotImplementedError

    def num_tokens(self, seq_id):
        raise NotImplementedError

    def num_free_blocks(self):
        raise NotImplementedError

    def ref_count(self, block_id):
        raise NotImplementedError
