import random
import statistics


def test_example(m):
    mf = m.MedianFinder()
    mf.add(1)
    mf.add(2)
    assert mf.median() == 1.5
    mf.add(3)
    assert mf.median() == 2.0
    assert len(mf) == 3


def test_empty_raises(m):
    mf = m.MedianFinder()
    assert len(mf) == 0
    try:
        mf.median()
    except ValueError:
        pass
    else:
        raise AssertionError("空的时候应该抛 ValueError")


def test_returns_float(m):
    mf = m.MedianFinder()
    mf.add(5)
    assert isinstance(mf.median(), float) and mf.median() == 5.0
    mf.add(6)
    assert mf.median() == 5.5


def test_negative_and_duplicates(m):
    mf = m.MedianFinder()
    for x in [-5, -5, -5, 10, 10]:
        mf.add(x)
    assert mf.median() == -5.0
    mf.add(10)
    assert mf.median() == 2.5


def test_descending_then_ascending(m):
    mf = m.MedianFinder()
    seen = []
    for x in list(range(20, 0, -1)) + list(range(21, 41)):
        mf.add(x)
        seen.append(x)
        assert mf.median() == statistics.median(seen)


def test_random_against_statistics(m):
    rng = random.Random(31)
    mf = m.MedianFinder()
    seen = []
    for _ in range(2000):
        x = rng.randint(-1000, 1000)
        mf.add(x)
        seen.append(x)
        if len(seen) % 7 == 0:
            assert mf.median() == statistics.median(seen)
    assert len(mf) == 2000


def test_float_values(m):
    rng = random.Random(37)
    mf = m.MedianFinder()
    seen = []
    for _ in range(301):
        x = rng.gauss(100.0, 15.0)
        mf.add(x)
        seen.append(x)
    assert abs(mf.median() - statistics.median(seen)) < 1e-12
