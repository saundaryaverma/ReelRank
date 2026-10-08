"""reelrank run      train and compare all models, write results/
reelrank similar  show movies closest to a title in embedding space"""
import argparse
import json
import sys
from pathlib import Path

from reelrank import data as dataset
from reelrank import metrics
from reelrank.evaluate import evaluate, summarize, tune_bpr, tune_knn
from reelrank.models import BPRMF, Popularity


def load_split(args):
    folder = Path(args.data_path) if args.data_path else dataset.download(args.data_dir)
    data = dataset.load(folder)
    return data, dataset.leave_last_out(data)


def run(args):
    data, split = load_split(args)
    k = args.k
    print(f"MovieLens 100K: {data.n_users} users, {data.n_items} movies, {len(data.users)} ratings")
    print("Split: each user's latest rating is the test item, the one before is validation\n")

    print("Tuning Item-kNN on validation:")
    _, shrink, knn = tune_knn(split, k=k)
    print(f"Best: shrink={shrink}\n")
    print("Tuning BPR-MF on validation:")
    _, params, bpr = tune_bpr(split, k=k)
    print(f"Best: {params}\n")

    models = [Popularity().fit(split), knn, bpr]
    results = {m.name: evaluate(m, split, k) for m in models}
    rows = [summarize(name, r, k) for name, r in results.items()]

    key = f"ndcg@{k}"
    lines = [f"| Model | NDCG@{k} (95% CI) | Recall@{k} | MRR | Catalog coverage |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lo, hi = r[f"{key}_ci"]
        lines.append(f"| {r['model']} | {r[key]:.4f} ({lo:.4f}-{hi:.4f}) | {r[f'recall@{k}']:.4f} | "
                     f"{r['mrr']:.4f} | {r['coverage']:.1%} |")
    a, b = results[bpr.name][key], results["Item-kNN"][key]
    lo, hi = metrics.paired_difference(a, b)
    verdict = "significant" if lo > 0 or hi < 0 else "not significant"
    lines += ["", f"BPR-MF minus Item-kNN, NDCG@{k}: {(a - b).mean():+.4f} "
                  f"(95% CI {lo:+.4f} to {hi:+.4f}, {verdict})"]
    table = "\n".join(lines)
    print(table)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps({"k": k, "bpr_params": params, "knn_shrink": shrink, "models": rows}, indent=2))
    print(f"\nSaved {out / 'results.md'}")
    return 0


def similar(args):
    data, split = load_split(args)
    matches = [i for i, t in enumerate(data.titles) if args.title.lower() in t.lower()]
    if not matches:
        print(f"No movie matching {args.title!r}", file=sys.stderr)
        return 2
    model = BPRMF(factors=64, epochs=40).fit(split)
    item = matches[0]
    print(f"Closest to {data.titles[item]} in embedding space:")
    for i in model.similar_items(item, args.n):
        print(f"  {data.titles[i]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="reelrank")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "similar"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default="data", help="where to download MovieLens")
        p.add_argument("--data-path", help="use an existing ml-100k folder instead of downloading")
        if name == "run":
            p.add_argument("--k", type=int, default=10)
            p.add_argument("--out", default="results")
        else:
            p.add_argument("title")
            p.add_argument("--n", type=int, default=5)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else similar(args)


if __name__ == "__main__":
    sys.exit(main())
