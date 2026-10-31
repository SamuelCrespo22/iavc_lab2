"""Part 1 experiments: BoVW + linear SVM over every option of the assignment.

Runs, for each category (positives vs. background) on the protocol splits:

  grid      Stages A/C/D/E/F: vocabulary size K x spatial tiling x feature map.
            Reports train/val/test AP; settings are chosen on val AP.
  fractions Stage G: AP vs. fraction of training data, several seeds,
            linear (L2) vs. Hellinger feature map.

Results are written to <out>/part1_grid.csv, <out>/part1_fractions.csv and
<out>/part1_fractions_<category>.png.

Usage (from the directory containing README.md):
    python -m runs.run_part1
    python -m runs.run_part1 --categories motorbike --ks 512 --skip-fractions
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from src import utils
from src.splits import loadProtocolSubset

CATEGORIES = ["motorbike", "aeroplane", "person", "horse", "car"]
KS = [256, 512, 1024]
NUM_TILES = 4


def removeSpatialInformation(histograms: np.ndarray) -> np.ndarray:
    """Average four spatially tiled histograms into one histogram per image.

    Args:
        histograms: Array of shape ``(N, num_words * 4)``.

    Returns:
        Array of shape ``(N, num_words)`` with the tile dimension averaged.
    """
    N = histograms.shape[0]
    return np.mean(np.reshape(histograms, (N, -1, 4)), axis=2)


def l1(x):
    return x / np.maximum(np.abs(x).sum(axis=1, keepdims=True), 1e-12)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# Feature maps applied to the raw count histograms. "sqrt_l2" is the Hellinger
# map: sqrt(h) / ||sqrt(h)||_2 == sqrt(h / ||h||_1).
FEATURE_MAPS = {
    "none": lambda h: h,
    "l1": l1,
    "l2": l2,
    "sqrt": np.sqrt,
    "sqrt_l1": lambda h: l1(np.sqrt(h)),
    "sqrt_l2": lambda h: l2(np.sqrt(h)),
}


def loadData(category, encoding):
    return {subset: loadProtocolSubset(category, encoding, subset=subset)
            for subset in ["train", "val", "test"]}


def prepare(hist, spatial, feature_map):
    hist = hist.astype(np.float64)
    if not spatial:
        hist = removeSpatialInformation(hist)
    return FEATURE_MAPS[feature_map](hist)


def trainAndEvaluate(data, transform, C, train_subset=None, return_model=False):
    """Train on ``transform(x_train)`` and report AP on train/val/test.

    With ``return_model``, returns ``(result, (W, b))``.
    """
    x_train, y_train = train_subset if train_subset is not None else data["train"][:2]
    x_train = transform(x_train)

    start = time.perf_counter()
    W, b = utils.train_svm(x_train, y_train, C=C)
    train_time = time.perf_counter() - start

    result = {"dim": x_train.shape[1], "train_time_s": train_time,
              "ap_train": average_precision_score(y_train, x_train @ W + b)}
    for subset in ["val", "test"]:
        x, y, _ = data[subset]
        start = time.perf_counter()
        scores = transform(x) @ W + b
        if subset == "test":
            result["infer_ms_per_image"] = 1000 * (time.perf_counter() - start) / len(y)
            top = np.argsort(scores)[::-1][:36]
            result["test_top36_pos"] = int((y[top] > 0).sum())
            pred = np.where(scores > 0, 1, -1)
            result["acc_test"] = accuracy_score(y, pred)
            result["f1_test"] = f1_score(y, pred, average="macro")
        result[f"ap_{subset}"] = average_precision_score(y, scores)
    return (result, (W, b)) if return_model else result


def runGrid(categories, ks, feature_maps, C):
    rows = []
    for category in categories:
        for K in ks:
            data = loadData(category, f"bovw_K{K}_C0")
            for spatial in [True, False]:
                for feature_map in feature_maps:
                    res = trainAndEvaluate(data, lambda h: prepare(h, spatial, feature_map), C)
                    rows.append({"category": category, "K": K, "spatial": spatial,
                                 "feature_map": feature_map, **res})
                    print(f"{category:10s} K={K:5d} spatial={spatial!s:5s} {feature_map:8s} "
                          f"AP train={res['ap_train']:.3f} val={res['ap_val']:.3f} "
                          f"test={res['ap_test']:.3f}")
    return pd.DataFrame(rows)


def stratifiedSubset(labels, fraction, seed):
    """Indices of a random subset keeping ``fraction`` of each class (at least one)."""
    rng = np.random.default_rng(seed)
    sel = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        num = len(idx) if fraction >= 1 else max(1, int(np.ceil(fraction * len(idx))))
        sel.append(rng.choice(idx, size=num, replace=False))
    return np.concatenate(sel)


def runFractions(categories, K, spatial, fractions, seeds, C):
    rows = []
    for category in categories:
        data = loadData(category, f"bovw_K{K}_C0")
        x_all, y_all, _ = data["train"]
        for fraction in fractions:
            for seed in seeds:
                sel = stratifiedSubset(y_all, fraction, seed)
                x, y = x_all[sel], y_all[sel]
                for feature_map in ["l2", "sqrt_l2"]:
                    res = trainAndEvaluate(data, lambda h: prepare(h, spatial, feature_map), C,
                                           train_subset=(x, y))
                    rows.append({"category": category, "K": K, "spatial": spatial,
                                 "fraction": fraction, "seed": seed,
                                 "num_train": len(y), "num_pos": int((y > 0).sum()),
                                 "feature_map": feature_map, **res})
            print(f"{category:10s} fraction={fraction:.2f} done")
    return pd.DataFrame(rows)


def plotFractions(df, out_dir):
    labels = {"l2": "linear (L2)", "sqrt_l2": "Hellinger"}
    for category, cat_df in df.groupby("category"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for feature_map, fm_df in cat_df.groupby("feature_map"):
            stats = fm_df.groupby("num_train")["ap_val"].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats["num_train"], stats["mean"], yerr=stats["std"].fillna(0),
                        marker="o", capsize=3, label=labels[feature_map])
        ax.set_xlabel("Number of training images")
        ax.set_ylabel("Validation AP")
        ax.set_title(f"{category}: AP vs. training set size")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"part1_fractions_{category}.png", dpi=150)
        plt.close(fig)


def summarize(grid, fractions):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== Test AP: feature map x spatial (rows) / category (cols), per K ===")
    for K, k_df in grid.groupby("K"):
        print(f"\nK = {K}")
        print(k_df.pivot_table(index=["spatial", "feature_map"], columns="category",
                               values="ap_test").round(3))

    print("\n=== Best setting per category, selected on val AP ===")
    best = grid.loc[grid.groupby("category")["ap_val"].idxmax()]
    print(best[["category", "K", "spatial", "feature_map", "dim",
                "ap_train", "ap_val", "ap_test"]].round(3).to_string(index=False))
    print(f"mAP (test) of selected settings: {best['ap_test'].mean():.3f}")

    if fractions is not None and len(fractions):
        print("\n=== Val AP vs. training fraction (mean +- std over seeds) ===")
        stats = fractions.groupby(["category", "feature_map", "fraction"])["ap_val"].agg(["mean", "std"])
        print(stats.round(3).unstack("fraction"))


def parseArgs():
    parser = argparse.ArgumentParser(description="Run all Part 1 (BoVW + SVM) experiments.")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--ks", nargs="+", type=int, default=KS)
    parser.add_argument("--feature-maps", nargs="+", default=list(FEATURE_MAPS),
                        choices=list(FEATURE_MAPS))
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (fixed, as in the assignment)")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fraction-k", type=int, default=512, help="K used for Stage G")
    parser.add_argument("--fraction-global", action="store_true",
                        help="Stage G without spatial tiling (default: tiled)")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-fractions", action="store_true")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = fractions = None
    if not args.skip_grid:
        grid = runGrid(args.categories, args.ks, args.feature_maps, args.C)
        grid.to_csv(out_dir / "part1_grid.csv", index=False)
    if not args.skip_fractions:
        fractions = runFractions(args.categories, args.fraction_k, not args.fraction_global,
                                 args.fractions, args.seeds, args.C)
        fractions.to_csv(out_dir / "part1_fractions.csv", index=False)
        plotFractions(fractions, out_dir)

    if grid is not None:
        summarize(grid, fractions)
    print(f"\nResults written to {out_dir.resolve()}")
