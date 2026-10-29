"""Part 3 experiments: large vocabulary + PCA vs. a directly smaller vocabulary.

For every representation (BoVW at each available K, plus VLAD / FV for the
"does the answer change" question) and every PCA output dimension d, trains the
Part 1 linear SVM per category and reports test AP, accuracy and macro-F1.

  BoVW       global histogram (dimension = K, as in the assignment), Hellinger map
  VLAD / FV  2x2 tiles, signed-sqrt + L2 (the setting selected in Part 2)
  PCA        fit on the training features only, then L2 re-normalization;
             d < original dimension and d <= num_train - 1

Conditions of the assignment table map to rows as follows:
  A  small K, no PCA          B  large K, no PCA
  C  large K + PCA to d       D  bovw K = d, no PCA (same final dimension as C)

Stability: each seed trains on a different stratified subset (--subsample of the
training set) and fits PCA on it; val/test are always the full sets.

Codebook fitting time (PCA + k-means/GMM) is read from the encoder .npz
("fit_time_s", saved by src.train); encoders trained before that was added
show NaN until retrained.

Results: <out>/part3_grid.csv and <out>/part3_encodings.csv.

Usage (from the directory containing README.md):
    python -m runs.run_part3
    python -m runs.run_part3 --pca-dims 64 128 256 --seeds 0 1 2 --categories motorbike
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from runs.run_part1 import CATEGORIES, l2, loadData, stratifiedSubset, trainAndEvaluate
from runs.run_part2 import ENCODINGS_DIR, measureEncodings, parseEncoding, prepare

# (encoding, spatial, feature map); missing encodings are skipped.
DEFAULT_CONFIGS = [
    ("bovw_K64_C0", False, "sqrt_l2"),
    ("bovw_K128_C0", False, "sqrt_l2"),
    ("bovw_K256_C0", False, "sqrt_l2"),
    ("bovw_K512_C0", False, "sqrt_l2"),
    ("bovw_K1024_C0", False, "sqrt_l2"),
    ("vlad_K32_C80", True, "ssr_l2"),
    ("fv_K32_C80", True, "ssr_l2"),
]


def pcaTransform(base, x_train, d, seed):
    """Fit PCA on ``base(x_train)``; returns (transform, fit time in seconds)."""
    start = time.perf_counter()
    pca = PCA(n_components=d, random_state=seed).fit(base(x_train))
    fit_time = time.perf_counter() - start
    return (lambda x: l2(pca.transform(base(x)))), fit_time


def runGrid(configs, categories, pca_dims, seeds, subsample, C):
    rows = []
    for encoding, spatial, feature_map in configs:
        enc_type, K, pca_desc = parseEncoding(encoding)
        D = pca_desc or 128
        base = lambda x: prepare(x, spatial, enc_type, feature_map, D)
        for category in categories:
            data = loadData(category, encoding)
            x_all, y_all, _ = data["train"]
            for seed in seeds:
                sel = stratifiedSubset(y_all, subsample, seed)
                x, y = x_all[sel], y_all[sel]
                orig_dim = base(x[:1]).shape[1]
                dims = [None] + [d for d in pca_dims if d < orig_dim and d <= len(y) - 1]
                for d in dims:
                    if d is None:
                        transform, pca_time = base, 0.0
                    else:
                        transform, pca_time = pcaTransform(base, x, d, seed)
                    res = trainAndEvaluate(data, transform, C, train_subset=(x, y))
                    rows.append({"encoding": encoding, "type": enc_type, "K": K,
                                 "spatial": spatial, "feature_map": feature_map,
                                 "category": category, "seed": seed,
                                 "orig_dim": orig_dim, "pca_dim": d or 0,
                                 "pca_fit_s": pca_time, **res})
                    print(f"{encoding:14s} {category:10s} seed={seed} "
                          f"dim={orig_dim:6d}->{res['dim']:6d} AP test={res['ap_test']:.3f}")
    return pd.DataFrame(rows)


def summarize(grid, encodings_info):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    # Average over categories first (mAP per seed), then mean/std over seeds.
    per_seed = (grid.groupby(["encoding", "pca_dim", "seed"], sort=False)
                .agg(mAP=("ap_test", "mean"), acc=("acc_test", "mean"), f1=("f1_test", "mean"),
                     orig_dim=("orig_dim", "first"), dim=("dim", "first"),
                     pca_fit_s=("pca_fit_s", "mean"), train_s=("train_time_s", "mean"),
                     infer_ms=("infer_ms_per_image", "mean")))
    table = per_seed.groupby(level=["encoding", "pca_dim"], sort=False).agg(
        orig_dim=("orig_dim", "first"), dim=("dim", "first"),
        mAP=("mAP", "mean"), mAP_std=("mAP", "std"), acc=("acc", "mean"), f1=("f1", "mean"),
        pca_fit_s=("pca_fit_s", "mean"), train_s=("train_s", "mean"), infer_ms=("infer_ms", "mean"))
    table["removed_%"] = 100 * (1 - table["dim"] / table["orig_dim"])
    table["kb_per_image_f32"] = table["dim"] * 4 / 1024

    print("\n=== Per representation and PCA dimension (0 = no PCA); mean over categories ===")
    print(table.round(3).to_string())

    print("\n=== Same final dimension: BoVW K=d (D) vs. larger K + PCA to d (C), test mAP ===")
    bovw = table.reset_index()
    bovw = bovw[bovw["encoding"].str.startswith("bovw")]
    matched = []
    for d in sorted(bovw.loc[bovw["pca_dim"] > 0, "pca_dim"].unique()):
        row = {"final_dim": d}
        direct = bovw[(bovw["pca_dim"] == 0) & (bovw["orig_dim"] == d)]
        if len(direct):
            row["direct K=d (no PCA)"] = direct["mAP"].iloc[0]
        for _, r in bovw[bovw["pca_dim"] == d].iterrows():
            row[f"K={r['orig_dim']} + PCA"] = r["mAP"]
        matched.append(row)
    print(pd.DataFrame(matched).set_index("final_dim").round(3).to_string())

    print("\n=== Per category test AP (mean over seeds) ===")
    print(grid.pivot_table(index=["encoding", "pca_dim"], columns="category",
                           values="ap_test", sort=False).round(3).to_string())

    print("\n=== Encoding cost (per image) and storage ===")
    print(encodings_info.round(3).to_string(index=False))


def parseArgs():
    parser = argparse.ArgumentParser(description="Run Part 3 (large vocabulary + PCA) experiments.")
    parser.add_argument("--encodings", nargs="+", default=None,
                        help="Restrict to these encodings (default: all of DEFAULT_CONFIGS that exist)")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=[32, 64, 128, 256, 512])
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [c for c in DEFAULT_CONFIGS if (ENCODINGS_DIR / c[0]).is_dir()]
    if args.encodings:
        configs = [c for c in configs if c[0] in args.encodings]
    missing = [c[0] for c in DEFAULT_CONFIGS if not (ENCODINGS_DIR / c[0]).is_dir()]
    print("Encodings:", ", ".join(c[0] for c in configs))
    if missing:
        print("Missing (skipped):", ", ".join(missing))

    encodings_info = measureEncodings([c[0] for c in configs], args.timing_images)
    encodings_info.to_csv(out_dir / "part3_encodings.csv", index=False)

    grid = runGrid(configs, args.categories, args.pca_dims, args.seeds, args.subsample, args.C)
    grid.to_csv(out_dir / "part3_grid.csv", index=False)

    summarize(grid, encodings_info)
    print(f"\nResults written to {out_dir.resolve()}")
