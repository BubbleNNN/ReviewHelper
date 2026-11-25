import itertools
import random


def _is_valid(order, n, edges):
    if sorted(order) != list(range(n)):
        return False
    pos = {c: i for i, c in enumerate(order)}
    return all(pos[b] < pos[a] for a, b in edges)


def _feasible_brute(n, edges):
    return any(_is_valid(list(p), n, edges) for p in itertools.permutations(range(n)))


def test_example(m):
    edges = [[1, 0], [2, 0], [3, 1], [3, 2]]
    order = m.find_order(4, edges)
    assert _is_valid(order, 4, edges)


def test_cycle_returns_empty(m):
    assert m.find_order(2, [[1, 0], [0, 1]]) == []
    assert m.find_order(3, [[0, 1], [1, 2], [2, 0]]) == []
    assert m.find_order(1, [[0, 0]]) == []


def test_no_edges_and_single(m):
    assert m.find_order(1, []) == [0]
    order = m.find_order(5, [])
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_duplicate_edges(m):
    edges = [[1, 0], [1, 0], [2, 1], [2, 1]]
    assert m.find_order(3, edges) == [0, 1, 2]


def test_cycle_hidden_behind_dag(m):
    # 0->1->2 是 DAG，但 3<->4 成环，整体无解
    assert m.find_order(5, [[1, 0], [2, 1], [4, 3], [3, 4]]) == []


def test_random_small_against_permutations(m):
    rng = random.Random(4)
    for _ in range(250):
        n = rng.randint(1, 6)
        edges = [[rng.randrange(n), rng.randrange(n)] for _ in range(rng.randint(0, 8))]
        order = m.find_order(n, edges)
        if _feasible_brute(n, edges):
            assert _is_valid(order, n, edges), (n, edges, order)
        else:
            assert order == [], (n, edges, order)


def test_large_random_dag(m):
    rng = random.Random(8)
    n = 3000
    perm = list(range(n))
    rng.shuffle(perm)
    edges = []
    for _ in range(15000):
        i, j = sorted(rng.sample(range(n), 2))
        edges.append([perm[j], perm[i]])  # perm[i] 必须在 perm[j] 之前
    order = m.find_order(n, edges)
    assert _is_valid(order, n, edges)
