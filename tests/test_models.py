import numpy as np

from reelrank.data import leave_last_out
from reelrank.evaluate import evaluate
from reelrank.models import BPRMF, ItemKNN, Popularity


def same_cluster(item, user):
    return (item < 20) == (user % 2 == 0)


def test_popularity_ranks_by_count(data):
    split = leave_last_out(data)
    model = Popularity().fit(split)
    top = np.argmax(model.scores(np.array([0]))[0])
    assert top == np.argmax(np.bincount(split.train_items, minlength=split.n_items))


def test_item_knn_recommends_within_taste_cluster(data):
    split = leave_last_out(data)
    scores = ItemKNN().fit(split).scores(np.arange(10))
    assert all(same_cluster(np.argmax(s), u) for u, s in enumerate(scores))


def test_bpr_learns_clusters_and_beats_popularity(data):
    split = leave_last_out(data)
    bpr = BPRMF(factors=8, epochs=30, lr=0.1, batch_size=64).fit(split)
    pop = Popularity().fit(split)
    assert evaluate(bpr, split, k=5)["recall@5"].mean() > evaluate(pop, split, k=5)["recall@5"].mean()
    top = np.argmax(bpr.scores(np.arange(10)), axis=1)
    assert sum(same_cluster(i, u) for u, i in enumerate(top)) >= 9


def test_bpr_similar_items_share_a_cluster(data):
    split = leave_last_out(data)
    bpr = BPRMF(factors=8, epochs=30, lr=0.1, batch_size=64).fit(split)
    neighbors = bpr.similar_items(0, n=5)
    assert 0 not in neighbors and all(n < 20 for n in neighbors)


def test_bpr_is_reproducible_with_a_seed(data):
    split = leave_last_out(data)
    a = BPRMF(factors=4, epochs=2, seed=7).fit(split).scores(np.array([0]))
    b = BPRMF(factors=4, epochs=2, seed=7).fit(split).scores(np.array([0]))
    assert np.allclose(a, b)
