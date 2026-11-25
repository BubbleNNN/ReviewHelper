def test_zero_paper_numbers(m):
    # ZeRO 论文 Figure 1：Ψ=7.5B, Nd=64 → 120 / 31.4 / 16.6 / 1.9 GB
    psi, nd = 7.5e9, 64
    got = [m.zero_model_states_bytes(psi, nd, s) / 1e9 for s in range(4)]
    assert round(got[0], 1) == 120.0
    assert round(got[1], 1) == 31.4
    assert round(got[2], 1) == 16.6
    assert round(got[3], 1) == 1.9


def test_zero_stage_monotone_and_dp1(m):
    psi = 1e9
    vals = [m.zero_model_states_bytes(psi, 8, s) for s in range(4)]
    assert vals[0] > vals[1] > vals[2] > vals[3]
    # dp=1 时切分没有意义，四个 stage 一样
    same = {m.zero_model_states_bytes(psi, 1, s) for s in range(4)}
    assert same == {16e9}


def test_zero_custom_k(m):
    # 不存 fp32 master（例如纯 bf16 Adam）：K=8 → stage 0 为 12Ψ
    assert m.zero_model_states_bytes(1e9, 4, 0, k=8) == 12e9
    assert m.zero_model_states_bytes(1e9, 4, 1, k=8) == 4e9 + 2e9


def test_kv_cache_llama_examples(m):
    # Llama-2-7B：每 token 2*32*32*128*2 = 524288 B = 512 KiB
    assert m.kv_cache_bytes(32, 32, 128, 1) == 524288
    assert m.kv_cache_bytes(32, 32, 128, 4096) == 2 * 1024 ** 3
    # Llama-3-70B（GQA, 8 个 KV head, 80 层）：320 KiB / token
    assert m.kv_cache_bytes(80, 8, 128, 1) == 320 * 1024
    assert m.kv_cache_bytes(80, 8, 128, 1000, batch=3, dtype_bytes=1) == 3 * 1000 * 163840


def test_max_concurrent_seqs(m):
    per_seq = 2 * 1024 ** 3  # Llama-2-7B 4K 上下文
    assert m.max_concurrent_seqs(40 * 1024 ** 3, 32, 32, 128, 4096) == 20
    assert m.max_concurrent_seqs(per_seq - 1, 32, 32, 128, 4096) == 0


def test_max_concurrent_seqs_paged(m):
    per_token = 524288
    # seq_len=100, block 16 → 7 块 = 112 token 的空间
    free = per_token * 112 * 5 + per_token * 111
    assert m.max_concurrent_seqs(free, 32, 32, 128, 100, block_size=16) == 5
    assert m.max_concurrent_seqs(free, 32, 32, 128, 100) == 6  # 不按块：100 token/条


def test_mfu(m):
    v = m.mfu(7e9, 6e5, 64, 989e12)
    assert abs(v - 0.39813) < 1e-4  # 手算：2.52e16 / 6.3296e16
    assert abs(m.mfu(1e9, 1e3, 1, 6e12) - 1.0) < 1e-12
