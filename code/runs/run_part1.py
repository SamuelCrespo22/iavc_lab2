"""Part 1 experiments: BoVW + linear SVM.

Runs, for each category (positives vs. background) on the protocol splits:

  grid      Stages A/C/D/E/F: vocabulary size K x spatial tiling x feature map.
            Reports train/val/test AP; settings are chosen on val AP.
  fractions Stage G: AP vs. fraction of training data, several seeds,
            linear (L2) vs. Hellinger feature map.

Results are written to <out>/part1_grid.csv, <out>/part1_fractions.csv and
<out>/part1_fractions.png. Example images, ranked lists, PR curves and visual
words (Stages A-D) are drawn by runs.run_part1_inspect.

Usage:
    python -m runs.run_part1
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src import utils
from src.splits import loadProtocolSubset

CATEGORIES = ["motorbike", "aeroplane", "person", "horse", "car"]
KS = [256, 512, 1024]
NUM_TILES = 4


def l1(x):
    return x / np.maximum(np.abs(x).sum(axis=1, keepdims=True), 1e-12)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# map: sqrt(h) / ||sqrt(h)||_2 == sqrt(h / ||h||_1).
FEATURE_MAPS = {
    "none": lambda h: h,
    "l1": l1,
    "l2": l2,
    "sqrt": np.sqrt,
    "sqrt_l1": lambda h: l1(np.sqrt(h)),
    "sqrt_l2": lambda h: l2(np.sqrt(h)), # Hellinger
}


def removeSpatialInformation(histograms: np.ndarray) -> np.ndarray:
    """Sum four spatially tiled histograms into one global histogram per image.

    Every keypoint falls in exactly one tile, so the sum equals the histogram of
    the whole image (the mean would be that histogram / 4).

    Args:
        histograms: Array of shape ``(N, num_words * 4)``.

    Returns:
        Array of shape ``(N, num_words)`` with the tile dimension summed.
    """
    # Tiles are stored one after the other: [tile 0 | tile 1 | tile 2 | tile 3].
    N = histograms.shape[0]
    return np.sum(np.reshape(histograms, (N, NUM_TILES, -1)), axis=1)


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
            # Classification at the SVM's own operating threshold, score > 0.
            pred = np.where(scores > 0, 1, -1)
            result["acc_test"] = accuracy_score(y, pred)
            result["f1_test"] = f1_score(y, pred, average="macro")
            result["precision_test"] = precision_score(y, pred, zero_division=0)
            result["recall_test"] = recall_score(y, pred)
            (tn, fp), (fn, tp) = confusion_matrix(y, pred, labels=[-1, 1])
            result.update(tp_test=int(tp), fp_test=int(fp), fn_test=int(fn), tn_test=int(tn))
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
    """Indices of a random subset keeping ``fraction`` of each class (at least one).

    Each class is permuted once per seed and the first images are kept, so for
    a fixed seed the subsets are nested: 10 % ⊂ 25 % ⊂ 50 % ⊂ 100 %.
    """
    rng = np.random.default_rng(seed)
    sel = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        num = len(idx) if fraction >= 1 else max(1, int(np.ceil(fraction * len(idx))))
        sel.append(rng.permutation(idx)[:num])
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
    """One panel per category: validation AP vs. number of training images."""
    labels = {"l2": "linear (L2)", "sqrt_l2": "Hellinger"}
    categories = list(dict.fromkeys(df["category"]))
    fig, axes = plt.subplots(1, len(categories), figsize=(3.6 * len(categories), 3.6), sharey=True, squeeze=False)
    for ax, category in zip(axes[0], categories):
        for feature_map, fm_df in df[df["category"] == category].groupby("feature_map"):
            stats = fm_df.groupby("num_train")["ap_val"].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats["num_train"], stats["mean"], yerr=stats["std"].fillna(0),
                        marker="o", capsize=3, label=labels[feature_map])
        ax.set_xlabel("Number of training images")
        ax.set_title(category)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Validation AP")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "part1_fractions.png", dpi=150)
    plt.close(fig)


def summarize(grid, fractions):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # Settings are compared on validation AP only; test AP is printed for the chosen one.
    print("\n=== Val AP: feature map x spatial (rows) / category (cols), per K ===")
    for K, k_df in grid.groupby("K"):
        print(f"\nK = {K}")
        print(k_df.pivot_table(index=["spatial", "feature_map"], columns="category",
                               values="ap_val").round(3))

    # One setting for every category, as in runs.report: the best mean val AP.
    K, spatial, feature_map = grid.groupby(["K", "spatial", "feature_map"])["ap_val"].mean().idxmax()
    best = grid[(grid["K"] == K) & (grid["spatial"] == spatial) & (grid["feature_map"] == feature_map)]
    print(f"\n=== Selected setting (best val mAP): K={K}, spatial={spatial}, {feature_map} ===")
    print(best[["category", "dim", "ap_train", "ap_val", "ap_test"]].round(3).to_string(index=False))
    print(f"mAP: val {best['ap_val'].mean():.3f}, test {best['ap_test'].mean():.3f}")

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
