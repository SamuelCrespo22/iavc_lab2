"""Part 3 experiments: large visual vocabulary + PCA vs. a directly smaller vocabulary.

Question: is it better to build a rich representation with a large codebook and
compress it with PCA, or to use a small codebook directly?

Representations (same protocol and linear SVM as Part 1: one classifier per
category, positives vs. background, C fixed):

  BoVW  global histogram (2x2 tiles summed), Hellinger map      dim = K
  VLAD  2x2 tiles, signed sqrt + L2                             dim = 4 * K * C
  FV    2x2 tiles, signed sqrt + L2 ("improved" FV)             dim = 4 * (2*K*C + K)

VLAD/FV stay tiled: their stored tiles are encoded (and VLAD normalized) per
tile, so, unlike BoVW counts, they cannot be merged into an exact global vector.

Each representation is evaluated without PCA and with PCA to every d in
--pca-dims such that d < dim and d <= n_train - 1. PCA is fitted on the training
images only and its output is L2 re-normalized.

Conditions of the assignment table (BoVW):
  A  small K, no PCA          (--small-k, default 256)
  B  large K, no PCA          (--large-k, default 1024)
  C  large K + PCA to d
  D  K = d, no PCA            (same final dimension as C)
VLAD/FV have no condition D: their smallest raw dimension is far above any d.
For them, the same-dimension comparison is a larger K + PCA vs. a smaller K +
PCA to the same d (part3_matched.csv).

Stability: each seed trains on a different stratified subset (--subsample) of
the training set and refits PCA on it; validation and test sets are always
complete. The codebooks themselves are fixed (one k-means/GMM per K).

Outputs (in --out):
  part3_grid.csv         one row per (encoding, d, category, seed)
  part3_summary.csv      per (encoding, d): mAP mean/std over seeds + costs   (Q2, Q4, Q5)
  part3_conditions.csv   the A/B/C/D rows of the summary                      (Q3)
  part3_matched.csv      test mAP per (type, final dimension) x codebook K    (Q3, Q5)
  part3_per_class.csv    test AP per category, no PCA                         (Q1)
  part3_map_vs_dim.png   test mAP vs. final dimension, one line per K         (Q1-Q3, Q5)

Usage:
    python -m runs.run_part3
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from runs.run_part1 import CATEGORIES, NUM_TILES, l2, loadData, stratifiedSubset, trainAndEvaluate
from runs.run_part2 import ENCODINGS_DIR, measureEncodings, parseEncoding, signedSqrt

DEFAULT_ENCODINGS = [
    "bovw_K32_C0", "bovw_K64_C0", "bovw_K128_C0", "bovw_K256_C0", "bovw_K512_C0", "bovw_K1024_C0",
    "vlad_K16_C80", "vlad_K32_C80", "vlad_K64_C80",
    "fv_K16_C80", "fv_K32_C80", "fv_K64_C80",
]


def globalHistogram(x):
    """(N, 4*K) tiled BoVW counts -> (N, K) whole-image counts.

    The stored vector is [tile 0 | tile 1 | tile 2 | tile 3]; every keypoint lies
    in exactly one tile, so the sum over tiles is the global histogram.
    """
    return x.reshape(len(x), NUM_TILES, -1).sum(axis=1)


# Representation fed to the SVM (and to PCA), per encoding type.
FEATURES = {
    "bovw": lambda x: l2(np.sqrt(globalHistogram(x))),  # Hellinger
    "vlad": lambda x: l2(signedSqrt(x)),
    "fv": lambda x: l2(signedSqrt(x)),
}


def fitPCA(x_train, d, seed):
    """Fit PCA to d dimensions; returns (transform, fit time in seconds)."""
    start = time.perf_counter()
    pca = PCA(n_components=d, random_state=seed).fit(x_train)
    fit_time = time.perf_counter() - start
    return (lambda x: l2(pca.transform(x))), fit_time


def runGrid(encodings, categories, pca_dims, seeds, subsample, C):
    rows = []
    for encoding in encodings:
        enc_type, K, _ = parseEncoding(encoding)
        for category in categories:
            # Compute the (non-PCA) representation once for train/val/test.
            data = {subset: (FEATURES[enc_type](x.astype(np.float64)), y, names)
                    for subset, (x, y, names) in loadData(category, encoding).items()}
            x_all, y_all, _ = data["train"]
            orig_dim = x_all.shape[1]
            for seed in seeds:
                sel = stratifiedSubset(y_all, subsample, seed)
                x, y = x_all[sel], y_all[sel]
                dims = [0] + [d for d in pca_dims if d < orig_dim and d <= len(y) - 1]
                for d in dims:
                    if d == 0:
                        transform, pca_time = (lambda v: v), 0.0
                    else:
                        transform, pca_time = fitPCA(x, d, seed)
                    res = trainAndEvaluate(data, transform, C, train_subset=(x, y))
                    rows.append({"encoding": encoding, "type": enc_type, "K": K,
                                 "category": category, "seed": seed, "num_train": len(y),
                                 "orig_dim": orig_dim, "pca_dim": d, "pca_fit_s": pca_time, **res})
                    print(f"{encoding:14s} {category:10s} seed={seed} "
                          f"dim={orig_dim:6d}->{res['dim']:6d} AP val={res['ap_val']:.3f} "
                          f"test={res['ap_test']:.3f}")
    return pd.DataFrame(rows)


def summarize(grid, costs):
    """One row per (encoding, pca_dim): mAP over categories, then mean/std over seeds."""
    keys = ["encoding", "type", "K", "pca_dim"]
    per_seed = grid.groupby(keys + ["seed"], sort=False).agg(
        mAP_val=("ap_val", "mean"), mAP_test=("ap_test", "mean"),
        acc_test=("acc_test", "mean"), f1_test=("f1_test", "mean"),
        orig_dim=("orig_dim", "first"), dim=("dim", "first"), pca_fit_s=("pca_fit_s", "mean"),
        svm_train_s=("train_time_s", "mean"), infer_us=("infer_ms_per_image", lambda s: 1000 * s.mean()))
    summary = per_seed.groupby(keys, sort=False).agg(
        orig_dim=("orig_dim", "first"), dim=("dim", "first"),
        mAP_val=("mAP_val", "mean"), mAP_test=("mAP_test", "mean"), mAP_test_std=("mAP_test", "std"),
        acc_test=("acc_test", "mean"), f1_test=("f1_test", "mean"), pca_fit_s=("pca_fit_s", "mean"),
        svm_train_s=("svm_train_s", "mean"), infer_us=("infer_us", "mean")).reset_index()
    summary["removed_%"] = 100 * (1 - summary["dim"] / summary["orig_dim"])
    summary["kb_per_image"] = summary["dim"] * 4 / 1024  # float32
    return summary.merge(costs, on="encoding", how="left")


def conditionsTable(summary, small_k, large_k):
    """Rows A/B/C/D of the assignment table, taken from the BoVW summary."""
    bovw = summary[summary["type"] == "bovw"]
    no_pca = bovw[bovw["pca_dim"] == 0]
    parts = [no_pca[no_pca["K"] == small_k].assign(condition="A"),
             no_pca[no_pca["K"] == large_k].assign(condition="B")]
    c_rows = bovw[(bovw["K"] == large_k) & (bovw["pca_dim"] > 0)]
    parts.append(c_rows.assign(condition="C"))
    parts.append(no_pca[no_pca["K"].isin(c_rows["pca_dim"])].assign(condition="D"))
    table = pd.concat(parts)
    return table[["condition", "encoding", "pca_dim", "dim", "mAP_val", "mAP_test", "mAP_test_std",
                  "acc_test", "f1_test", "codebook_fit_s", "encode_ms_per_image",
                  "pca_fit_s", "svm_train_s", "infer_us", "kb_per_image"]]


def matchedTable(summary):
    """Test mAP: rows = (type, final dimension), columns = codebook K.

    Each row compares codebooks at the same final dimension. For BoVW the
    diagonal (K = d) is the codebook used directly and the cells to its right
    are larger codebooks compressed with PCA to d.
    """
    return summary.pivot_table(index=["type", "dim"], columns="K", values="mAP_test")


def plotMapVsDim(summary, path):
    """Test mAP vs. final dimension: one line per codebook, from its smallest PCA d to no PCA."""
    types = [t for t in ["bovw", "vlad", "fv"] if t in set(summary["type"])]
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 4), sharey=True, squeeze=False)
    for ax, enc_type in zip(axes[0], types):
        df = summary[summary["type"] == enc_type]
        for K, g in df.groupby("K"):
            if (g["pca_dim"] > 0).any():
                g = g.sort_values("dim")
                ax.errorbar(g["dim"], g["mAP_test"], yerr=g["mAP_test_std"], marker="o", capsize=2,
                            label=f"K={K} + PCA")
        if enc_type == "bovw":
            direct = df[df["pca_dim"] == 0].sort_values("dim")
            ax.plot(direct["dim"], direct["mAP_test"], "k--s", label="K = d, no PCA")
        ax.set_xscale("log", base=2)
        ax.set_title(enc_type.upper())
        ax.set_xlabel("Final feature dimension")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Test mAP (mean over categories)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser(description="Run Part 3 (large vocabulary + PCA) experiments.")
    parser.add_argument("--encodings", nargs="+", default=DEFAULT_ENCODINGS,
                        help="Encodings to evaluate; missing ones are skipped")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--small-k", type=int, default=256, help="Condition A codebook")
    parser.add_argument("--large-k", type=int, default=1024, help="Conditions B/C codebook")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--subsample", type=float, default=0.9,
                        help="Stratified fraction of the training set used per seed (1.0 = all)")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (fixed, as in Part 1)")
    parser.add_argument("--timing-images", type=int, default=5,
                        help="Images used to time SIFT + encoding; 0 disables")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    encodings = [e for e in args.encodings if (ENCODINGS_DIR / e).is_dir()]
    missing = [e for e in args.encodings if e not in encodings]
    print("Encodings:", ", ".join(encodings))
    if missing:
        print("Missing (skipped):", ", ".join(missing))

    # Per-encoder costs: codebook fitting (from the encoder .npz), SIFT and encoding per image.
    costs = measureEncodings(encodings, args.timing_images).reindex(
        columns=["encoding", "codebook_fit_s", "sift_ms_per_image", "encode_ms_per_image"])

    grid = runGrid(encodings, args.categories, args.pca_dims, args.seeds, args.subsample, args.C)
    summary = summarize(grid, costs)
    conditions = conditionsTable(summary, args.small_k, args.large_k)
    matched = matchedTable(summary)
    per_class = grid[grid["pca_dim"] == 0].pivot_table(
        index="encoding", columns="category", values="ap_test", sort=False)

    grid.to_csv(out / "part3_grid.csv", index=False)
    summary.to_csv(out / "part3_summary.csv", index=False)
    conditions.to_csv(out / "part3_conditions.csv", index=False)
    matched.to_csv(out / "part3_matched.csv")
    per_class.to_csv(out / "part3_per_class.csv")
    plotMapVsDim(summary, out / "part3_map_vs_dim.png")

    print("\n=== Q1: test AP per category, no PCA ===")
    print(per_class.round(3).to_string())
    print("\n=== Q2/Q4/Q5: every representation and PCA dimension (0 = no PCA) ===")
    print(summary.drop(columns=["type", "K"]).round(3).to_string(index=False))
    print(f"\n=== Q3: assignment conditions (A: K={args.small_k}, B/C: K={args.large_k}) ===")
    print(conditions.round(3).to_string(index=False))
    print("\n=== Q3/Q5: test mAP at the same final dimension (rows) per codebook K (cols) ===")
    print(matched.round(3).to_string())
    print(f"\nResults written to {out.resolve()}")
