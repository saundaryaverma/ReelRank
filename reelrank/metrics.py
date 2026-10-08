"""Ranking metrics for one held-out item per user, plus bootstrap confidence intervals."""
import numpy as np


def ranks_of(scores, targets, exclude):
    """0-based rank of each user's target item among all items, after removing items the
    user already interacted with in training (exclude is a 0/1 matrix)."""
    scores = np.where(exclude > 0, -np.inf, scores)
    rows = np.arange(len(targets))
    target_scores = scores[rows, targets]
    return (scores > target_scores[:, None]).sum(axis=1)


def per_user(ranks, k):
    hit = ranks < k
    return {
        f"ndcg@{k}": np.where(hit, 1.0 / np.log2(ranks + 2), 0.0),
        f"recall@{k}": hit.astype(float),
        "mrr": 1.0 / (ranks + 1),
    }


def coverage(scores, exclude, k):
    """Share of the catalog that appears in at least one user's top-k list."""
    scores = np.where(exclude > 0, -np.inf, scores)
    if k >= scores.shape[1]:
        return float(np.isfinite(scores).any(axis=0).mean())
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return len(np.unique(top)) / scores.shape[1]


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_difference(a, b, n=1000, seed=0):
    """Bootstrap CI for mean(a - b) over the same users. If it excludes 0, the
    difference is unlikely to be noise."""
    return bootstrap_ci(a - b, n, seed)
