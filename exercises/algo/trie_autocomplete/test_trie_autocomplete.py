import random


def _lcp_with_set(word, words):
    best = 0
    for w in words:
        n = 0
        while n < min(len(w), len(word)) and w[n] == word[n]:
            n += 1
        best = max(best, n)
    return best


def test_basic(m):
    t = m.Trie()
    t.insert("apple")
    assert t.search("apple")
    assert not t.search("app")
    assert t.starts_with("app")
    t.insert("app")
    assert t.search("app")
    assert not t.starts_with("b")


def test_longest_prefix(m):
    t = m.Trie()
    for w in ["apple", "apply", "banana"]:
        t.insert(w)
    assert t.longest_prefix("applet") == 5
    assert t.longest_prefix("apx") == 2
    assert t.longest_prefix("cat") == 0
    assert t.longest_prefix("") == 0
    assert t.longest_prefix("ban") == 3


def test_complete_order_and_limit(m):
    t = m.Trie()
    for w in ["car", "cart", "carbon", "cat", "dog", "ca"]:
        t.insert(w)
    assert t.complete("ca") == ["ca", "car", "carbon", "cart", "cat"]
    assert t.complete("car", limit=2) == ["car", "carbon"]
    assert t.complete("ca", limit=0) == []
    assert t.complete("z") == []
    assert t.complete("") == ["ca", "car", "carbon", "cart", "cat", "dog"]


def test_duplicates_and_empty_tree(m):
    t = m.Trie()
    assert not t.starts_with("")
    assert t.complete("") == []
    t.insert("x")
    t.insert("x")
    assert t.complete("") == ["x"]
    assert t.starts_with("")


def test_empty_string_word(m):
    t = m.Trie()
    t.insert("")
    assert t.search("")
    assert t.complete("") == [""]


def test_random_against_set(m):
    rng = random.Random(17)
    alphabet = "abc"
    t = m.Trie()
    words = set()
    for _ in range(400):
        w = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
        t.insert(w)
        words.add(w)
        q = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
        assert t.search(q) == (q in words)
        assert t.starts_with(q) == any(x.startswith(q) for x in words)
        assert t.longest_prefix(q) == _lcp_with_set(q, words)
        lim = rng.choice([None, 1, 3, 10])
        want = sorted(x for x in words if x.startswith(q))
        assert t.complete(q, lim) == (want if lim is None else want[:lim])
