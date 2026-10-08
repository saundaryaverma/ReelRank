"""Download MovieLens 100K and split it the way a live system would see it."""
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def download(data_dir="data"):
    """Fetch the dataset once. Returns the folder holding u.data and u.item."""
    target = Path(data_dir) / "ml-100k"
    if (target / "u.data").exists():
        return target
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as res:
        zipfile.ZipFile(io.BytesIO(res.read())).extractall(data_dir)
    return target


@dataclass
class Interactions:
    """Ratings as arrays, with users and items re-indexed from 0."""
    users: np.ndarray
    items: np.ndarray
    times: np.ndarray
    n_users: int
    n_items: int
    titles: list
    genres: np.ndarray  # n_items x len(GENRES), 0/1


def load(folder):
    folder = Path(folder)
    raw = np.loadtxt(folder / "u.data", dtype=np.int64)
    users, items, times = raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 3]
    titles, genres = [], []
    with open(folder / "u.item", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            titles.append(parts[1])
            genres.append([int(g) for g in parts[5:5 + len(GENRES)]])
    return Interactions(users, items, times, int(users.max()) + 1, len(titles), titles,
                        np.array(genres, dtype=np.int8))


@dataclass
class Split:
    train_users: np.ndarray
    train_items: np.ndarray
    val: np.ndarray   # held-out item per user (-1 if none)
    test: np.ndarray  # held-out item per user (-1 if none)
    n_users: int
    n_items: int

    def train_matrix(self):
        m = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        m[self.train_users, self.train_items] = 1.0
        return m


def leave_last_out(data):
    """For each user, hold out their most recent interaction for test and the one
    before it for validation. Training only sees the past, so nothing from the future
    leaks into the model, which is the mistake random splits make."""
    order = np.lexsort((data.items, data.times, data.users))
    users, items = data.users[order], data.items[order]
    val = np.full(data.n_users, -1)
    test = np.full(data.n_users, -1)
    keep = np.ones(len(users), dtype=bool)
    ends = np.flatnonzero(np.r_[users[1:] != users[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    for s, e in zip(starts, ends):
        if e - s + 1 < 3:  # need at least one training interaction
            continue
        test[users[e]], val[users[e]] = items[e], items[e - 1]
        keep[e - 1:e + 1] = False
    return Split(users[keep], items[keep], val, test, data.n_users, data.n_items)
