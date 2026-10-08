"""Three recommenders, from simplest to learned embeddings. Each has fit(split) and
scores(users) -> (len(users) x n_items) array where higher means recommend first."""
import numpy as np


class Popularity:
    name = "Popularity"

    def fit(self, split):
        self.counts = np.bincount(split.train_items, minlength=split.n_items).astype(np.float32)
        return self

    def scores(self, users):
        return np.tile(self.counts, (len(users), 1))


class ItemKNN:
    """Recommend items similar to what the user already watched. Similarity is cosine
    over co-watch counts, with shrinkage so rare pairs don't look too similar."""
    name = "Item-kNN"

    def __init__(self, shrink=10.0):
        self.shrink = shrink

    def fit(self, split):
        self.x = split.train_matrix()
        co = self.x.T @ self.x
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        return self

    def scores(self, users):
        return self.x[users] @ self.sim


class BPRMF:
    """Matrix factorization trained with Bayesian Personalized Ranking (BPR).
    Each user and movie gets an embedding. Training pushes a user's embedding closer to
    movies they watched than to random movies they didn't, which optimizes ranking
    directly instead of predicting star ratings."""
    name = "BPR-MF embeddings"

    def __init__(self, factors=64, lr=0.05, reg=1e-4, epochs=40, batch_size=1024, seed=0):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed

    def fit(self, split, on_epoch=None):
        rng = np.random.default_rng(self.seed)
        self.u = rng.normal(0, 0.1, (split.n_users, self.factors))
        self.v = rng.normal(0, 0.1, (split.n_items, self.factors))
        self.b = np.zeros(split.n_items)
        seen = split.train_matrix() > 0
        users, items = split.train_users, split.train_items
        for epoch in range(self.epochs):
            order = rng.permutation(len(users))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                u, i = users[idx], items[idx]
                j = rng.integers(0, split.n_items, len(idx))
                bad = seen[u, j]  # resample negatives the user actually watched
                while bad.any():
                    j[bad] = rng.integers(0, split.n_items, bad.sum())
                    bad = seen[u, j]
                self._step(u, i, j)
            if on_epoch:
                on_epoch(epoch + 1, self)
        return self

    def _step(self, u, i, j):
        pu, qi, qj = self.u[u], self.v[i], self.v[j]
        x = (pu * (qi - qj)).sum(axis=1) + self.b[i] - self.b[j]
        g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))  # gradient of log-sigmoid
        lr, reg = self.lr, self.reg
        np.add.at(self.u, u, lr * (g[:, None] * (qi - qj) - reg * pu))
        np.add.at(self.v, i, lr * (g[:, None] * pu - reg * qi))
        np.add.at(self.v, j, lr * (-g[:, None] * pu - reg * qj))
        np.add.at(self.b, i, lr * (g - reg * self.b[i]))
        np.add.at(self.b, j, lr * (-g - reg * self.b[j]))

    def scores(self, users):
        return self.u[users] @ self.v.T + self.b

    def similar_items(self, item, n=5):
        """Movies whose embeddings point the same way, a quick check that they learned meaning."""
        v = self.v / (np.linalg.norm(self.v, axis=1, keepdims=True) + 1e-9)
        sims = v @ v[item]
        sims[item] = -np.inf
        return np.argsort(-sims)[:n]
