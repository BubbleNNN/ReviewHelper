import random


def _expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"应当抛出 {exc.__name__}")


def _check_invariants(bm, num_blocks, bs, seqs):
    """seqs: 测试方自己记的 {seq_id: n_tokens}"""
    refs = [0] * num_blocks
    for s, n in seqs.items():
        table = bm.block_table(s)
        assert bm.num_tokens(s) == n
        assert len(table) == -(-n // bs), "块数应为 ceil(n_tokens / block_size)"
        for b in table:
            refs[b] += 1
    for b in range(num_blocks):
        assert bm.ref_count(b) == refs[b], f"块 {b} 的引用计数不对"
    assert bm.num_free_blocks() == sum(1 for r in refs if r == 0)


def test_allocate_and_append(m):
    bm = m.BlockManager(8, 4)
    bm.allocate("a", 6)
    assert len(bm.block_table("a")) == 2
    assert bm.num_free_blocks() == 6
    assert bm.append_token("a") is None  # 7 个 token，仍在第 2 块
    assert bm.append_token("a") is None  # 8 个，第 2 块满
    assert len(bm.block_table("a")) == 2
    assert bm.append_token("a") is None  # 9 个，需要第 3 块
    assert len(bm.block_table("a")) == 3
    assert len(set(bm.block_table("a"))) == 3
    _check_invariants(bm, 8, 4, {"a": 9})


def test_fork_and_copy_on_write(m):
    bm = m.BlockManager(8, 4)
    bm.allocate("a", 6)
    b0, b1 = bm.block_table("a")
    bm.fork("a", "b")
    assert bm.block_table("b") == [b0, b1]
    assert bm.ref_count(b0) == 2 and bm.ref_count(b1) == 2
    assert bm.num_free_blocks() == 6
    cp = bm.append_token("b")              # b1 未满且共享 → CoW
    assert cp is not None and cp[0] == b1
    new = cp[1]
    assert bm.block_table("b") == [b0, new]
    assert bm.ref_count(b1) == 1 and bm.ref_count(new) == 1 and bm.ref_count(b0) == 2
    assert bm.append_token("a") is None    # b1 现在只属于 a，不用复制
    _check_invariants(bm, 8, 4, {"a": 7, "b": 7})
    bm.free("a")
    assert bm.ref_count(b0) == 1 and bm.ref_count(b1) == 0
    bm.free("b")
    assert bm.num_free_blocks() == 8


def test_fork_full_last_block_no_copy(m):
    bm = m.BlockManager(6, 4)
    bm.allocate("p", 8)                    # 两块都满
    bm.fork("p", "c")
    assert bm.append_token("c") is None    # 满块不需要 CoW，直接开新块
    t = bm.block_table("c")
    assert t[:2] == bm.block_table("p") and len(t) == 3
    _check_invariants(bm, 6, 4, {"p": 8, "c": 9})


def test_out_of_memory_is_atomic(m):
    bm = m.BlockManager(3, 4)
    bm.allocate("a", 5)                    # 2 块
    _expect(MemoryError, bm.allocate, "b", 9)  # 需要 3 块，只剩 1
    assert bm.num_free_blocks() == 1
    _expect(KeyError, bm.block_table, "b")
    bm.allocate("b", 4)                    # 最后一块
    for _ in range(3):
        bm.append_token("a")               # 5→8，仍在第 2 块
    _expect(MemoryError, bm.append_token, "a")
    assert bm.num_tokens("a") == 8, "失败的 append 不能改 token 数"
    _check_invariants(bm, 3, 4, {"a": 8, "b": 4})


def test_duplicate_ids(m):
    bm = m.BlockManager(4, 2)
    bm.allocate("x", 1)
    _expect(ValueError, bm.allocate, "x", 1)
    bm.allocate("y", 1)
    _expect(ValueError, bm.fork, "x", "y")


def test_random_ops_invariants(m):
    rng = random.Random(0)
    N, BS = 40, 4
    bm = m.BlockManager(N, BS)
    seqs = {}
    nxt = 0
    for _ in range(600):
        op = rng.random()
        try:
            if op < 0.25 or not seqs:
                n = rng.randint(1, 10)
                bm.allocate(nxt, n)
                seqs[nxt] = n
                nxt += 1
            elif op < 0.65:
                s = rng.choice(list(seqs))
                bm.append_token(s)
                seqs[s] += 1
            elif op < 0.8:
                s = rng.choice(list(seqs))
                bm.fork(s, nxt)
                seqs[nxt] = seqs[s]
                nxt += 1
            else:
                s = rng.choice(list(seqs))
                bm.free(s)
                del seqs[s]
        except MemoryError:
            pass
        _check_invariants(bm, N, BS, seqs)
    for s in list(seqs):
        bm.free(s)
    assert bm.num_free_blocks() == N
