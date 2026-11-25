class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_list = list(range(num_blocks - 1, -1, -1))  # 栈顶是 0
        self.ref = [0] * num_blocks
        self.tables = {}
        self.ntok = {}

    def _take(self):
        b = self.free_list.pop()
        self.ref[b] = 1
        return b

    def _release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free_list.append(b)

    def allocate(self, seq_id, n_tokens):
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already exists")
        need = -(-n_tokens // self.block_size)
        if need > len(self.free_list):
            raise MemoryError("not enough free blocks")
        self.tables[seq_id] = [self._take() for _ in range(need)]
        self.ntok[seq_id] = n_tokens

    def append_token(self, seq_id):
        table, n = self.tables[seq_id], self.ntok[seq_id]
        copy = None
        if n % self.block_size == 0:          # 最后一块满了（或空序列）
            if not self.free_list:
                raise MemoryError("not enough free blocks")
            table.append(self._take())
        else:
            last = table[-1]
            if self.ref[last] > 1:            # 共享的未满块 → 写时复制
                if not self.free_list:
                    raise MemoryError("not enough free blocks")
                new = self._take()
                self.ref[last] -= 1           # 仍被别人引用，不会归零
                table[-1] = new
                copy = (last, new)
        self.ntok[seq_id] = n + 1
        return copy

    def fork(self, parent_id, child_id):
        if child_id in self.tables:
            raise ValueError(f"sequence {child_id} already exists")
        table = self.tables[parent_id]
        for b in table:
            self.ref[b] += 1
        self.tables[child_id] = list(table)
        self.ntok[child_id] = self.ntok[parent_id]

    def free(self, seq_id):
        for b in self.tables.pop(seq_id):
            self._release(b)
        del self.ntok[seq_id]

    def block_table(self, seq_id):
        return list(self.tables[seq_id])

    def num_tokens(self, seq_id):
        return self.ntok[seq_id]

    def num_free_blocks(self):
        return len(self.free_list)

    def ref_count(self, block_id):
        return self.ref[block_id]
