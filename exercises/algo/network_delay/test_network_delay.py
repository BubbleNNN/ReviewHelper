import random


def _bellman_ford(times, n, k):
    INF = float("inf")
    dist = [INF] * (n + 1)
    dist[k] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in times:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    ans = max(dist[1:])
    return -1 if ans == INF else ans


def test_examples(m):
    assert m.network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert m.network_delay_time([[1, 2, 1]], 2, 1) == 1
    assert m.network_delay_time([[1, 2, 1]], 2, 2) == -1


def test_single_node(m):
    assert m.network_delay_time([], 1, 1) == 0


def test_shortcut_beats_direct_edge(m):
    # 1->3 直连 10，但 1->2->3 只要 3
    times = [[1, 3, 10], [1, 2, 1], [2, 3, 2]]
    assert m.network_delay_time(times, 3, 1) == 3


def test_parallel_edges_self_loops_zero_weight(m):
    times = [[1, 2, 5], [1, 2, 2], [2, 2, 0], [2, 3, 0], [3, 1, 7]]
    assert m.network_delay_time(times, 3, 1) == 2


def test_random_against_bellman_ford(m):
    rng = random.Random(21)
    for _ in range(300):
        n = rng.randint(1, 9)
        times = [[rng.randint(1, n), rng.randint(1, n), rng.randint(0, 9)] for _ in range(rng.randint(0, 25))]
        k = rng.randint(1, n)
        assert m.network_delay_time(times, n, k) == _bellman_ford(times, n, k), (times, n, k)


def test_large_graph(m):
    rng = random.Random(5)
    n = 3000
    times = [[i, i + 1, rng.randint(1, 5)] for i in range(1, n)]  # 保证连通
    times += [[rng.randint(1, n), rng.randint(1, n), rng.randint(0, 50)] for _ in range(30000)]
    assert m.network_delay_time(times, n, 1) == _bellman_ford(times, n, 1)
