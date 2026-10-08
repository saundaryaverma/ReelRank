import numpy as np
import pytest

from reelrank import metrics


def test_rank_ignores_items_already_seen():
    scores = np.array([[0.9, 0.8, 0.1, 0.5]])
    seen = np.array([[1, 0, 0, 0]])
    # item 0 scores highest but was already watched, so item 1 ranks first
    assert metrics.ranks_of(scores, np.array([1]), seen)[0] == 0
    assert metrics.ranks_of(scores, np.array([3]), seen)[0] == 1


def test_ndcg_recall_mrr_values():
    ranks = np.array([0, 1, 9, 10])
    m = metrics.per_user(ranks, k=10)
    assert m["ndcg@10"] == pytest.approx([1.0, 1 / np.log2(3), 1 / np.log2(11), 0.0])
    assert list(m["recall@10"]) == [1, 1, 1, 0]
    assert m["mrr"] == pytest.approx([1, 1 / 2, 1 / 10, 1 / 11])


def test_coverage_counts_distinct_recommended_items():
    scores = np.array([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
    assert metrics.coverage(scores, np.zeros_like(scores), k=2) == 0.5
    seen = np.array([[0, 0, 0, 0], [1, 1, 0, 0]])
    assert metrics.coverage(scores, seen, k=2) == 1.0


def test_bootstrap_ci_brackets_the_mean():
    values = np.random.default_rng(1).normal(5, 1, 500)
    lo, hi = metrics.bootstrap_ci(values)
    assert lo < values.mean() < hi and hi - lo < 0.5


def test_paired_difference_detects_a_real_gap_and_not_noise():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 400)
    lo, hi = metrics.paired_difference(a + 0.5, a)
    assert lo > 0
    lo, hi = metrics.paired_difference(a + rng.normal(0, 0.01, 400), a)
    assert lo < 0 < hi or abs(lo) < 0.01


def test_coverage_when_k_covers_the_whole_catalog():
    scores = np.array([[1.0, 2.0, 3.0]])
    assert metrics.coverage(scores, np.array([[0, 0, 1]]), k=3) == 2 / 3
