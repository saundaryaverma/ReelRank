# ReelRank

Movie recommendations with learned embeddings, evaluated the way a production ranking team would evaluate them.

![CI](https://github.com/saundaryaverma/reelrank/actions/workflows/ci.yml/badge.svg)

ReelRank trains three recommenders on MovieLens 100K (943 users, 1,682 movies, 100,000 ratings) and compares them on ranking quality. The models range from a popularity baseline to BPR matrix factorization, which learns an embedding for every user and movie. The focus is on getting the evaluation right: a time-based split with no future data leakage, tuning on validation only, full-catalog ranking, and confidence intervals so that small differences aren't mistaken for real ones.

## Results

Each user's most recent rating is the test item. Every model ranks all 1,682 movies for every user, minus movies the user has already watched, and we check where the held-out movie lands.

| Model | NDCG@10 (95% CI) | Recall@10 | MRR | Catalog coverage |
| --- | --- | --- | --- | --- |
| Popularity | 0.0255 (0.0176-0.0335) | 0.0498 | 0.0259 | 5.4% |
| Item-kNN | 0.0275 (0.0203-0.0357) | 0.0594 | 0.0282 | 14.0% |
| **BPR-MF embeddings** | **0.0327** (0.0247-0.0412) | **0.0710** | **0.0349** | **45.2%** |

BPR-MF minus Item-kNN, NDCG@10: +0.0052 (95% CI -0.0029 to +0.0135, not significant).

What this shows:

- **The embedding model is the best on every metric.** Compared with popularity it improves NDCG@10 by 28% and Recall@10 by 43%. Compared with Item-kNN it improves NDCG@10 by 19% and Recall@10 by 20%.
- **Its biggest advantage is catalog coverage.** Its top-10 lists draw on 45% of the catalog, about 3x more than Item-kNN and 8x more than popularity. Popularity-style models keep recommending the same few hits. The embedding model recommends across the long tail while still ranking better.
- **The NDCG gain over Item-kNN is not statistically significant.** A paired bootstrap over the same 943 users gives a 95% interval of -0.003 to +0.014 for the difference. With one held-out movie per user, the test set is small, so I report the gain as promising, not proven. Shipping it would call for an online A/B test.
- **Validation scores are higher than test scores** (BPR: 0.057 vs 0.033). Part of that is selection: the best of eight configurations looks a bit better on the data used to pick it. Part is time: the test movie is later than the validation movie, and tastes drift. Using validation only for tuning keeps the test numbers honest.

Your exact numbers may differ slightly from these, depending on your NumPy version.

### Do the embeddings mean anything?

Movies close together in embedding space should be similar. They are:

```
$ reelrank similar "Star Wars"
Closest to Star Wars (1977) in embedding space:
  Return of the Jedi (1983)
  Raiders of the Lost Ark (1981)
  Empire Strikes Back, The (1980)
  Monty Python and the Holy Grail (1974)
  Terminator, The (1984)
```

The model was never told about sequels, genres, or directors. It learned this structure from who watched what.

## Run it

```bash
git clone https://github.com/saundaryaverma/reelrank.git
cd reelrank
pip install -e .
reelrank run                    # downloads MovieLens 100K, tunes, trains, evaluates (about a minute)
reelrank similar "Toy Story"    # nearest movies in embedding space
```

Results are saved to `results/results.md` and `results/results.json`. The only dependency is NumPy.

## How it works

**Split (`data.py`).** For each user, the latest rating is the test item and the one before it is validation. Training sees only earlier ratings. A random split would let the model train on a user's future to predict their past, which inflates scores and doesn't match how a live system works.

**Models (`models.py`).**
- *Popularity:* recommends the most-watched movies to everyone. A baseline every model should beat.
- *Item-kNN:* scores movies by cosine similarity to what the user already watched, using co-watch counts.
- *BPR-MF:* learns a 64-dimensional embedding per user and per movie. Training samples a movie the user watched and one they didn't, and nudges the user's embedding toward the first and away from the second (Bayesian Personalized Ranking). That optimizes the ranking directly instead of predicting star ratings. Training is vectorized in mini-batches, and negatives the user has actually watched are resampled.

**Tuning (`evaluate.py`).** Both Item-kNN and BPR-MF are tuned on validation NDCG@10, so the comparison is fair. The test set is used once, at the end.

**Metrics (`metrics.py`).**
- *NDCG@10:* rewards ranking the held-out movie high, with more credit near the top.
- *Recall@10:* how often the held-out movie appears in the top 10.
- *MRR:* average of 1 / rank across the full catalog.
- *Catalog coverage:* share of all movies that appear in at least one user's top 10.
- *Bootstrap confidence intervals:* resample users 1,000 times. The paired version compares two models on the same users.

## Tests

```bash
pip install -e ".[dev]"
pytest --cov=reelrank
```

19 tests (97% coverage) run on small synthetic data with two clear taste groups, so they're fast and need no download. They check metric formulas against hand-computed values, that the split never leaks future ratings, that already-watched movies are never recommended, that tuning never reads the test set, and that BPR learns the taste groups and beats popularity. CI also runs the full MovieLens benchmark and posts the results table to the run summary.

## Next: a two-tower model with movie features

BPR-MF only knows movie IDs, so it can't recommend a brand-new movie that nobody has watched yet. The next step is a two-tower model in PyTorch: one tower turns a user's history into an embedding, the other turns a movie's ID plus its genres into an embedding, and the score is their dot product. That handles cold-start movies and is how large recommendation systems usually do candidate retrieval. It can be added to `models.py` with the same `fit` and `scores` interface and compared in the same table.

## License

MIT
