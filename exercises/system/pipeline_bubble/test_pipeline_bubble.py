def _check_timeline(res, p, m, tf, tb):
    tl = res["timeline"]
    assert len(tl) == p
    when = {}
    for s in range(p):
        ops = sorted(tl[s], key=lambda e: e[2])
        assert sorted((o, i) for o, i, _, _ in ops) == sorted([("F", i) for i in range(m)] + [("B", i) for i in range(m)])
        for (o, i, a, b) in ops:
            assert abs((b - a) - (tf if o == "F" else tb)) < 1e-9
            when[(o, s, i)] = (a, b)
        for prev, nxt in zip(ops, ops[1:]):
            assert nxt[2] >= prev[3] - 1e-9, "同一个 stage 上的操作不能重叠"
    for s in range(p):
        for i in range(m):
            if s > 0:
                assert when[("F", s, i)][0] >= when[("F", s - 1, i)][1] - 1e-9
            assert when[("B", s, i)][0] >= when[("F", s, i)][1] - 1e-9
            if s < p - 1:
                assert when[("B", s, i)][0] >= when[("B", s + 1, i)][1] - 1e-9


def test_gpipe_bubble_formula(m):
    for p, mb in [(2, 2), (4, 4), (4, 8), (8, 3)]:
        r = m.simulate("gpipe", p, mb)
        assert abs(r["total_time"] - (mb + p - 1) * 3.0) < 1e-9
        assert abs(r["bubble_ratio"] - (p - 1) / (mb + p - 1)) < 1e-9


def test_1f1b_same_bubble_as_gpipe(m):
    for p, mb in [(2, 5), (4, 4), (4, 8), (3, 1), (8, 16)]:
        for tf, tb in [(1.0, 2.0), (1.0, 1.0)]:
            r = m.simulate("1f1b", p, mb, tf, tb)
            assert abs(r["total_time"] - (mb + p - 1) * (tf + tb)) < 1e-9
            assert abs(r["bubble_ratio"] - (p - 1) / (mb + p - 1)) < 1e-9


def test_timelines_respect_dependencies(m):
    for sch in ("gpipe", "1f1b"):
        for p, mb in [(3, 5), (4, 2)]:
            _check_timeline(m.simulate(sch, p, mb, 1.0, 2.0), p, mb, 1.0, 2.0)


def test_peak_activation_memory(m):
    assert m.simulate("gpipe", 4, 8)["peak_inflight"] == [8, 8, 8, 8]
    assert m.simulate("1f1b", 4, 8)["peak_inflight"] == [4, 3, 2, 1]
    assert m.simulate("1f1b", 4, 2)["peak_inflight"] == [2, 2, 2, 1]


def test_1f1b_stage0_order(m):
    r = m.simulate("1f1b", 3, 4)
    ops = [(o, i) for o, i, _, _ in sorted(r["timeline"][0], key=lambda e: e[2])]
    assert ops == [("F", 0), ("F", 1), ("F", 2), ("B", 0), ("F", 3), ("B", 1), ("B", 2), ("B", 3)]
    last = [(o, i) for o, i, _, _ in sorted(r["timeline"][2], key=lambda e: e[2])]
    assert last[:2] == [("F", 0), ("B", 0)], "最后一个 stage 没有 warmup"


def test_single_stage_no_bubble(m):
    for sch in ("gpipe", "1f1b"):
        r = m.simulate(sch, 1, 5, 1.0, 2.0)
        assert abs(r["total_time"] - 15.0) < 1e-9
        assert abs(r["bubble_ratio"]) < 1e-12


def test_more_microbatches_less_bubble(m):
    vals = [m.simulate("1f1b", 4, mb)["bubble_ratio"] for mb in (1, 2, 4, 8, 16, 32)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] < 0.09  # 3/35
