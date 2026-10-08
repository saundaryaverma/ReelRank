"""Evaluate models on the held-out split, tune on validation only, report with confidence intervals."""
import itertools

import numpy as np

from reelrank import metrics
from reelrank.models import BPRMF, ItemKNN


def evaluate(model, split, k=10, on="test"):
    targets_all = split.test if on == "test" else split.val
    users = np.flatnonzero(targets_all >= 0)
    exclude = split.train_matrix()[users]
    if on == "test":  # the validation item is also history by test time
        exclude[np.arange(len(users)), split.val[users]] = 1.0
    scores = model.scores(users)
    ranks = metrics.ranks_of(scores, targets_all[users], exclude)
    result = metrics.per_user(ranks, k)
    result["coverage"] = metrics.coverage(scores, exclude, k)
    return result


def tune_bpr(split, grid=None, k=10, log=print):
    """Pick hyperparameters by validation NDCG. The test set is never touched here."""
    grid = grid or {"factors": [32, 64], "reg": [1e-4, 1e-3], "epochs": [20, 40]}
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        model = BPRMF(**params).fit(split)
        score = evaluate(model, split, k, on="val")[f"ndcg@{k}"].mean()
        log(f"  {params}  val NDCG@{k} = {score:.4f}")
        if best is None or score > best[0]:
            best = (score, params, model)
    return best


def tune_knn(split, shrinks=(0, 10, 50, 100), k=10, log=print):
    """Tune the baseline too, so the comparison with BPR is fair."""
    best = None
    for shrink in shrinks:
        model = ItemKNN(shrink=shrink).fit(split)
        score = evaluate(model, split, k, on="val")[f"ndcg@{k}"].mean()
        log(f"  shrink={shrink}  val NDCG@{k} = {score:.4f}")
        if best is None or score > best[0]:
            best = (score, shrink, model)
    return best


def summarize(name, result, k):
    key = f"ndcg@{k}"
    lo, hi = metrics.bootstrap_ci(result[key])
    return {
        "model": name,
        key: float(result[key].mean()),
        f"{key}_ci": [lo, hi],
        f"recall@{k}": float(result[f"recall@{k}"].mean()),
        "mrr": float(result["mrr"].mean()),
        "coverage": float(result["coverage"]),
    }
