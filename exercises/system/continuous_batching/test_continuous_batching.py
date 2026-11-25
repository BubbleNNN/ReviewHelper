import random


def _trace(seed, n=25):
    rng = random.Random(seed)
    return [(rng.randint(0, 20), rng.randint(1, 30), rng.randint(1, 12)) for _ in range(n)]


def test_hand_trace_batch_cap(m):
    R = [(0, 10, 3), (0, 10, 1), (0, 10, 2)]
    c = m.simulate_continuous(R, max_batch=2, token_budget=100)
    assert c["start"] == [0, 0, 1]
    assert c["finish"] == [2, 0, 2]
    assert c["tokens_per_step"] == [20, 11, 2]
    assert abs(c["throughput"] - 2.0) < 1e-12
    s = m.simulate_static(R, max_batch=2, token_budget=100)
    assert s["start"] == [0, 0, 3]
    assert s["finish"] == [2, 0, 4]
    assert s["tokens_per_step"] == [20, 1, 1, 10, 1]
    assert abs(s["throughput"] - 1.2) < 1e-12


def test_hand_trace_token_budget(m):
    R = [(0, 8, 2), (0, 8, 2), (1, 3, 1)]
    c = m.simulate_continuous(R, max_batch=4, token_budget=12)
    # t0 只放得下 A；t1 先留 1 个 decode，再放 B(8)、C(3) 正好 12
    assert c["start"] == [0, 1, 1]
    assert c["finish"] == [1, 2, 1]
    assert c["tokens_per_step"] == [8, 12, 1]
    s = m.simulate_static(R, max_batch=4, token_budget=12)
    assert s["start"] == [0, 2, 2]
    assert s["finish"] == [1, 3, 2]


def test_idle_until_arrival(m):
    c = m.simulate_continuous([(5, 4, 2)], max_batch=2, token_budget=8)
    assert c["finish"] == [6]
    assert c["tokens_per_step"] == [0, 0, 0, 0, 0, 4, 1]
    assert abs(c["throughput"] - 2 / 7) < 1e-12


def test_invariants_on_random_traces(m):
    for seed in range(6):
        R = _trace(seed)
        for fn in (m.simulate_continuous, m.simulate_static):
            r = fn(R, 4, 48)
            assert len(r["tokens_per_step"]) == max(r["finish"]) + 1
            assert max(r["tokens_per_step"]) <= 48, "超过 token budget"
            assert max(r["running_per_step"]) <= 4, "超过 max_batch"
            for i, (a, p, o) in enumerate(R):
                assert r["start"][i] >= a
                assert r["finish"][i] - r["start"][i] + 1 == o
            # FCFS：开始时间按 (arrival, idx) 单调不减
            order = sorted(range(len(R)), key=lambda i: (R[i][0], i))
            starts = [r["start"][i] for i in order]
            assert starts == sorted(starts)
            # 产出的 token 总数守恒
            assert sum(r["running_per_step"]) == sum(o for _, _, o in R)


def test_continuous_beats_static(m):
    for seed in range(6):
        R = _trace(seed)
        c = m.simulate_continuous(R, 4, 48)
        s = m.simulate_static(R, 4, 48)
        lat_c = sum(f - a for f, (a, _, _) in zip(c["finish"], R))
        lat_s = sum(f - a for f, (a, _, _) in zip(s["finish"], R))
        assert lat_c < lat_s
        assert c["throughput"] > s["throughput"]


def test_strict_fcfs_head_of_line(m):
    # A 在 decode 时每步占 1 个预算，只剩 11；B 需要 12，必须等 A 结束
    # C 虽然很小，也不能越过队头 B
    R = [(0, 6, 3), (1, 12, 1), (1, 1, 1)]
    c = m.simulate_continuous(R, max_batch=4, token_budget=12)
    assert c["start"][1] == 3, "B 需要整步预算，要等 A 在 t2 结束"
    assert c["start"][2] == 4, "C 不能越过队头 B"


def test_invalid_config(m):
    for args in ([(0, 20, 1)], 2, 10), ([(0, 1, 1)], 8, 4):
        try:
            m.simulate_continuous(*args)
        except ValueError:
            continue
        raise AssertionError("应抛 ValueError")
