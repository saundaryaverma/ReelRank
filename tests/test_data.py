import numpy as np

from reelrank.data import Interactions, load, leave_last_out
from tests.conftest import write_ml100k


def test_holds_out_latest_for_test_and_previous_for_validation():
    d = Interactions(np.array([0, 0, 0, 0]), np.array([5, 6, 7, 8]), np.array([1, 4, 2, 3]),
                     1, 10, [], None)
    split = leave_last_out(d)
    assert split.test[0] == 6 and split.val[0] == 8
    assert set(split.train_items) == {5, 7}


def test_no_future_leakage(data):
    split = leave_last_out(data)
    for u in range(data.n_users):
        mine = data.users == u
        last_time = data.times[mine].max()
        test_time = data.times[mine & (data.items == split.test[u])][0]
        assert test_time == last_time
        train_items = split.train_items[split.train_users == u]
        assert split.test[u] not in train_items and split.val[u] not in train_items


def test_users_with_too_little_history_are_not_evaluated():
    d = Interactions(np.array([0, 0, 1, 1, 1]), np.array([1, 2, 1, 2, 3]),
                     np.array([1, 2, 1, 2, 3]), 2, 5, [], None)
    split = leave_last_out(d)
    assert split.test[0] == -1 and split.test[1] == 3


def test_load_reads_movielens_files(tmp_path, data):
    folder = write_ml100k(tmp_path / "ml-100k", data)
    loaded = load(folder)
    assert loaded.n_users == data.n_users and loaded.n_items == data.n_items
    assert np.array_equal(np.sort(loaded.items), np.sort(data.items))
    assert loaded.titles[0] == "Movie 0"
