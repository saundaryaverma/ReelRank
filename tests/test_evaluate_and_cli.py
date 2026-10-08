import json

import numpy as np

from reelrank.cli import main
from reelrank.data import leave_last_out
from reelrank.evaluate import evaluate, tune_bpr
from tests.conftest import write_ml100k


class Fixed:
    """A fake model that always scores items in the same order."""
    def __init__(self, n_items):
        self.order = np.arange(n_items, 0, -1, dtype=float)

    def scores(self, users):
        return np.tile(self.order, (len(users), 1))


def test_test_eval_also_excludes_the_validation_item(data):
    split = leave_last_out(data)
    # Put the validation item at the very top for every user. It must be skipped.
    model = Fixed(split.n_items)
    users = np.flatnonzero(split.test >= 0)
    r = evaluate(model, split, k=split.n_items, on="test")
    assert len(r["mrr"]) == len(users)
    scores = model.scores(users)
    for row, u in enumerate(users):
        better = (scores[row] > scores[row, split.test[u]]).sum()
        seen = set(split.train_items[split.train_users == u]) | {split.val[u]}
        expected = better - sum(scores[row, i] > scores[row, split.test[u]] for i in seen)
        assert r["mrr"][row] == 1 / (expected + 1)


def test_tuning_only_reads_validation(data, monkeypatch):
    split = leave_last_out(data)
    split.test[:] = -1  # if tuning touched test, evaluate would fail on empty users
    score, params, _ = tune_bpr(split, grid={"factors": [4, 8], "epochs": [2]}, log=lambda *a: None)
    assert params["factors"] in (4, 8) and score >= 0


def test_cli_run_end_to_end(tmp_path, data, capsys):
    folder = write_ml100k(tmp_path / "ml-100k", data)
    out = tmp_path / "results"
    assert main(["run", "--data-path", str(folder), "--out", str(out), "--k", "5"]) == 0
    printed = capsys.readouterr().out
    assert "BPR-MF embeddings" in printed and "Item-kNN" in printed and "95% CI" in printed
    saved = json.loads((out / "results.json").read_text())
    assert {m["model"] for m in saved["models"]} == {"Popularity", "Item-kNN", "BPR-MF embeddings"}


def test_cli_similar(tmp_path, data, capsys):
    folder = write_ml100k(tmp_path / "ml-100k", data)
    assert main(["similar", "Movie 3", "--data-path", str(folder)]) == 0
    assert "Closest to Movie 3" in capsys.readouterr().out
    assert main(["similar", "No Such Film", "--data-path", str(folder)]) == 2
