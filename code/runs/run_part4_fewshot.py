"""Part 4, Stages C and D: few-shot learning, flip augmentation and the SVM C.

Uses one frozen representation (default: DINOv2 CLS token, L2-normalized) with
the Part 1 protocol: one linear SVM per category, positives vs. background.

  Stage C  Train with 1, 5, 10, 20, 50 and all training positives. Negatives are
           always every background training image. Each size is drawn with
           several seeds (all positives: a single draw). Two conditions:
             none  the sampled training images only
             flip  plus their horizontally flipped copies (training data only;
                   validation and test images are never flipped)
           Reports validation AP, mean +- std over seeds.

           Test-time augmentation, as a separate condition with all positives:
             score_avg    mean of the SVM scores of the image and of its flip
             feature_avg  SVM score of the feature map of the mean raw feature
           With a linear SVM, averaging the vectors the SVM sees equals averaging
           scores exactly; feature_avg differs only through the (non-linear) L2.

  Stage D  Sweeps C over 0.1 ... 1000 with 5 and with all positives (no
           augmentation) and reports train and validation AP. C is selected per
           regime on mean validation AP over the categories; only the selected C
           is evaluated on the test set.

Extract the features with flips first: python -m src.extract_deep -m dinov2_vits14 --flip

Results (in --out): part4_fewshot.csv / .png, part4_tta.csv,
part4_svm_c.csv / .png, part4_svm_c_test.csv.

Usage:
    python -m runs.run_part4_fewshot
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import utils
from src.splits import loadProtocolSubset
from runs.run_part1 import CATEGORIES, l2
from runs.run_part2 import ENCODINGS_DIR

FEATURE_MAPS = {
    "none": lambda x: x,
    "l2": l2,
    "sqrt_l2": lambda x: l2(np.sqrt(x)),  # non-negative features only (ResNet)
}


def loadSubsets(category, encoding):
    """``{subset: (x, x_flip, y)}``: raw features of the images and of their flips."""
    subsets = {}
    for subset in ["train", "val", "test"]:
        x, y, _ = loadProtocolSubset(category, encoding, subset)
        x_flip, _, _ = loadProtocolSubset(category, encoding, subset, key="histograms_flip")
        subsets[subset] = (x.astype(np.float64), x_flip.astype(np.float64), y)
    return subsets


def sampleTrain(y, num_pos, seed):
    """Indices of ``num_pos`` random positives (all when 0) and of every negative."""
    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
    if 0 < num_pos < len(pos):
        pos = np.random.default_rng(seed).choice(pos, size=num_pos, replace=False)
    return np.concatenate([pos, neg])


def fit(x, y, C):
    """Train the Part 1 linear SVM; returns the scoring function."""
    W, b = utils.train_svm(x, y, C=C)
    return lambda v: v @ W + b


def drawsFor(num_pos, seeds):
    # With all positives every seed selects the same images.
    return seeds if num_pos else seeds[:1]


def regimeName(num_pos):
    return f"{num_pos} positives" if num_pos else "all positives"


def runFewShot(data, fm, num_pos_list, seeds, C):
    rows = []
    for category, subsets in data.items():
        x, x_flip, y = subsets["train"]
        x_val, _, y_val = subsets["val"]
        for num_pos in num_pos_list:
            for seed in drawsFor(num_pos, seeds):
                sel = sampleTrain(y, num_pos, seed)
                for aug in ["none", "flip"]:
                    x_tr, y_tr = fm(x[sel]), y[sel]
                    if aug == "flip":
                        x_tr, y_tr = np.vstack([x_tr, fm(x_flip[sel])]), np.concatenate([y_tr, y_tr])
                    score = fit(x_tr, y_tr, C)
                    rows.append({"category": category, "num_pos": int((y[sel] > 0).sum()),
                                 "all_pos": num_pos == 0, "seed": seed, "aug": aug,
                                 "ap_val": average_precision_score(y_val, score(fm(x_val)))})
        print(f"Stage C: {category} done")
    return pd.DataFrame(rows)


def runTTA(data, fm, C):
    """Validation AP with all positives: train-time flip x test-time flip."""
    rows = []
    for category, subsets in data.items():
        x, x_flip, y = subsets["train"]
        x_val, x_val_flip, y_val = subsets["val"]
        for aug in ["none", "flip"]:
            if aug == "flip":
                score = fit(np.vstack([fm(x), fm(x_flip)]), np.concatenate([y, y]), C)
            else:
                score = fit(fm(x), y, C)
            tta = {"no_tta": score(fm(x_val)),
                   "score_avg": (score(fm(x_val)) + score(fm(x_val_flip))) / 2,
                   "feature_avg": score(fm((x_val + x_val_flip) / 2))}
            rows.append({"category": category, "train_aug": aug,
                         **{k: average_precision_score(y_val, s) for k, s in tta.items()}})
    return pd.DataFrame(rows)


def runCSweep(data, fm, Cs, regimes, seeds):
    rows = []
    for category, subsets in data.items():
        x, _, y = subsets["train"]
        x_val, _, y_val = subsets["val"]
        for num_pos in regimes:
            for seed in drawsFor(num_pos, seeds):
                sel = sampleTrain(y, num_pos, seed)
                for C in Cs:
                    score = fit(fm(x[sel]), y[sel], C)
                    rows.append({"category": category, "regime": regimeName(num_pos), "seed": seed, "C": C,
                                 "ap_train": average_precision_score(y[sel], score(fm(x[sel]))),
                                 "ap_val": average_precision_score(y_val, score(fm(x_val)))})
        print(f"Stage D: {category} done")
    return pd.DataFrame(rows)


def testSelectedC(data, fm, sweep, regimes, seeds):
    """Pick C per regime on mean val AP, then evaluate it once on the test set."""
    mean_val = sweep.groupby(["regime", "C"])["ap_val"].mean()
    rows = []
    for num_pos in regimes:
        regime = regimeName(num_pos)
        best_C = mean_val[regime].idxmax()
        for category, subsets in data.items():
            x, _, y = subsets["train"]
            x_test, _, y_test = subsets["test"]
            for seed in drawsFor(num_pos, seeds):
                sel = sampleTrain(y, num_pos, seed)
                score = fit(fm(x[sel]), y[sel], best_C)
                rows.append({"regime": regime, "C": best_C, "category": category, "seed": seed,
                             "ap_test": average_precision_score(y_test, score(fm(x_test)))})
    return pd.DataFrame(rows)


def plotFewShot(fewshot, path):
    labels = {"none": "no augmentation", "flip": "+ horizontal flip"}
    n = fewshot["category"].nunique()
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (category, df) in zip(axes, fewshot.groupby("category", sort=False)):
        for aug, g in df.groupby("aug", sort=False):
            stats = g.groupby("num_pos")["ap_val"].agg(["mean", "std"]).reset_index()
            ax.errorbar(stats["num_pos"], stats["mean"], yerr=stats["std"].fillna(0),
                        marker="o", ms=4, capsize=2, label=labels[aug])
        ax.set_xscale("log")
        ax.set_title(category)
        ax.set_xlabel("Positive training images")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Validation AP")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plotCSweep(sweep, path):
    # mAP over categories per seed, then mean +- std over seeds.
    per_seed = sweep.groupby(["regime", "C", "seed"])[["ap_train", "ap_val"]].mean()
    stats = per_seed.groupby(level=["regime", "C"]).agg(["mean", "std"])
    regimes = list(dict.fromkeys(sweep["regime"]))
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 3.5), sharey=True, squeeze=False)
    for ax, regime in zip(axes[0], regimes):
        s = stats.loc[regime]
        for col, label in [("ap_train", "train"), ("ap_val", "validation")]:
            ax.errorbar(s.index, s[col]["mean"], yerr=s[col]["std"].fillna(0),
                        marker="o", ms=4, capsize=2, label=label)
        ax.set_xscale("log")
        ax.set_title(regime)
        ax.set_xlabel("SVM C")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mAP (mean over categories)")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(fewshot, tta, sweep, test):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== Stage C: validation AP, mean +- std over seeds ===")
    stats = fewshot.groupby(["category", "aug", "num_pos"])["ap_val"].agg(["mean", "std"])
    print(stats.round(3).to_string())

    print("\n=== Stage C: test-time flip, validation AP with all positives ===")
    print(tta.round(4).to_string(index=False))
    print("mAP:")
    print(tta.groupby("train_aug")[["no_tta", "score_avg", "feature_avg"]].mean().round(4).to_string())

    print("\n=== Stage D: mAP per regime and C (mean over categories and seeds) ===")
    print(sweep.groupby(["regime", "C"])[["ap_train", "ap_val"]].mean().round(3).to_string())

    print("\n=== Stage D: selected C, test AP (mean over seeds) ===")
    table = test.pivot_table(index=["regime", "C"], columns="category", values="ap_test")
    table["mAP"] = table.mean(axis=1)
    print(table.round(3).to_string())


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4 Stages C/D: few-shot, flip augmentation and SVM C.")
    parser.add_argument("--encoding", default="dinov2_vits14_cls")
    parser.add_argument("--feature-map", default="l2", choices=list(FEATURE_MAPS))
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--num-pos", nargs="+", type=int, default=[1, 5, 10, 20, 50, 0],
                        help="Positive training images per draw; 0 = all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--C", type=float, default=10.0, help="SVM C for Stage C (as in Part 1)")
    parser.add_argument("--Cs", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000], help="Stage D sweep")
    parser.add_argument("--regimes", nargs="+", type=int, default=[5, 0],
                        help="Positives used in the Stage D sweep; 0 = all")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(ENCODINGS_DIR / args.encoding / "background_train.npz") as f:
        if "histograms_flip" not in f.files:
            raise SystemExit(f"{args.encoding} has no flipped features: re-extract it with --flip")
    print(f"Representation: {args.encoding} ({args.feature_map}). Negatives: all background training images.")

    fm = FEATURE_MAPS[args.feature_map]
    data = {category: loadSubsets(category, args.encoding) for category in args.categories}

    fewshot = runFewShot(data, fm, args.num_pos, args.seeds, args.C)
    tta = runTTA(data, fm, args.C)
    sweep = runCSweep(data, fm, args.Cs, args.regimes, args.seeds)
    test = testSelectedC(data, fm, sweep, args.regimes, args.seeds)

    fewshot.to_csv(out_dir / "part4_fewshot.csv", index=False)
    tta.to_csv(out_dir / "part4_tta.csv", index=False)
    sweep.to_csv(out_dir / "part4_svm_c.csv", index=False)
    test.to_csv(out_dir / "part4_svm_c_test.csv", index=False)
    plotFewShot(fewshot, out_dir / "part4_fewshot.png")
    plotCSweep(sweep, out_dir / "part4_svm_c.png")

    summarize(fewshot, tta, sweep, test)
    print(f"\nResults written to {out_dir.resolve()}")
