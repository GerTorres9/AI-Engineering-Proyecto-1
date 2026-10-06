from scripts.load_test import percentile95


def test_p95_linear_interpolation():
    assert percentile95([5, 1, 2, 4, 3]) == 4.8
