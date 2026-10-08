import numpy as np
import pytest

from reelrank.data import GENRES, Interactions


def clustered(n_users=60, n_items=40, per_user=12, seed=0):
    """Two taste groups: even users watch items 0-19, odd users watch items 20-39.
    A model that learns anything should recommend within the user's cluster."""
    rng = np.random.default_rng(seed)
    users, items, times = [], [], []
    for u in range(n_users):
        pool = np.arange(0, 20) if u % 2 == 0 else np.arange(20, 40)
        chosen = rng.choice(pool, per_user, replace=False)
        users += [u] * per_user
        items += list(chosen)
        times += list(1000 * u + np.arange(per_user))
    return Interactions(np.array(users), np.array(items), np.array(times), n_users, n_items,
                        [f"Movie {i}" for i in range(n_items)],
                        np.zeros((n_items, len(GENRES)), dtype=np.int8))


@pytest.fixture
def data():
    return clustered()


def write_ml100k(folder, data):
    """Write a dataset in MovieLens file format for end-to-end tests."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "u.data", "w") as f:
        for u, i, t in zip(data.users, data.items, data.times):
            f.write(f"{u + 1}\t{i + 1}\t4\t{t}\n")
    with open(folder / "u.item", "w", encoding="latin-1") as f:
        for i, title in enumerate(data.titles):
            genres = "|".join(["0"] * len(GENRES))
            f.write(f"{i + 1}|{title}|01-Jan-1995||http://x|{genres}\n")
    return folder
