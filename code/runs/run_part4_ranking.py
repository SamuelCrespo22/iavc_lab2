"""Part 4: ranking metrics and precision-recall curves of the final methods.

For every method of the required comparison table (the representation selected
on validation, see runs.report.finalSelection) and every category, the test
images (category positives vs. background) are ranked by the method's score:
the Part 1 linear SVM (C = 10) or, for CLIP zero-shot, the cosine similarity to
the class prompt (prompt set selected on validation by run_part4_clip).

Reported per method and category:
  AP                              scikit-learn average_precision_score
  P@K, R@K                        precision / recall of the top K images (--ks)
  precision_thr0, recall_thr0     classification at the SVM threshold 0
                                  (none for zero-shot, which has no threshold)

Run after the other Part 4 scripts. Results (in --out): part4_ranking.csv and
part4_pr_curves.png (one panel per category, one curve per method).

Usage (from the directory containing README.md):
    python -m runs.run_part4_ranking
"""

import os

# torch must be imported before numpy in this env (see src.extract_deep).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

from src import utils
from runs.report import finalSelection
from runs.run_part1 import CATEGORIES, l2, loadData
from runs.run_part2 import parseEncoding, prepare
from runs.run_part4_clip import ENCODING as CLIP_ENCODING, loadClip, textEmbeddings
from runs.run_part4_resnet import BASELINES, FEATURE_MAPS
from runs.run_part4_visual import selectedPromptSet


def transformFor(encoding, feature_map):
    """The feature map the comparison table used for this encoding."""
    if encoding.startswith(("bovw_", "vlad_", "fv_")):
        enc_type, _, pca_dim = parseEncoding(encoding)
        spatial = {name: spatial for name, spatial, _ in BASELINES}[encoding]
        return lambda x: prepare(x, spatial, enc_type, feature_map, pca_dim or 128)
    return lambda x: FEATURE_MAPS[feature_map](x.astype(np.float64))


def svmScores(encoding, feature_map, category, C):
    data = loadData(category, encoding)
    transform = transformFor(encoding, feature_map)
    (x_train, y_train, _), (x_test, y_test, _) = data["train"], data["test"]
    W, b = utils.train_svm(transform(x_train), y_train, C=C)
    return y_test, transform(x_test) @ W + b


def zeroShotScores(texts, category):
    x_test, y_test, _ = loadData(category, CLIP_ENCODING)["test"]
    return y_test, l2(x_test.astype(np.float64)) @ texts[category]


def rankingMetrics(y, scores, ks, threshold):
    relevant = y[np.argsort(-scores)] > 0
    row = {"AP": average_precision_score(y, scores)}
    for k in ks:
        row[f"P@{k}"] = relevant[:k].mean()
    for k in ks:
        row[f"R@{k}"] = relevant[:k].sum() / relevant.sum()
    if threshold is not None:
        row["precision_thr0"] = precision_score(y > 0, scores > threshold)
        row["recall_thr0"] = recall_score(y > 0, scores > threshold)
    return row


def plotCurves(curves, categories, path):
    fig, axes = plt.subplots(1, len(categories), figsize=(3.6 * len(categories), 3.6), sharey=True, squeeze=False)
    for ax, category in zip(axes[0], categories):
        for (method, cat), (precision, recall) in curves.items():
            if cat == category:
                ax.plot(recall, precision, lw=1.5, label=method)
        ax.set_title(category)
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Precision")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4: ranking metrics of the final methods.")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100], help="K of P@K and R@K")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (as in the comparison table)")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    selection = finalSelection(out_dir)
    if selection is None:
        raise SystemExit("Missing Part 4 grids: run run_part4_resnet and run_part4_dino first")
    texts = textEmbeddings(loadClip(), selectedPromptSet(out_dir))

    rows, curves = [], {}
    for category in args.categories:
        scored = {method: (svmScores(r["encoding"], r["feature_map"], category, args.C), 0)
                  for method, r in selection.iterrows()}
        scored["CLIP zero-shot"] = (zeroShotScores(texts, category), None)
        for method, ((y, scores), threshold) in scored.items():
            rows.append({"method": method, "category": category,
                         **rankingMetrics(y, scores, args.ks, threshold)})
            precision, recall, _ = precision_recall_curve(y, scores)
            curves[method, category] = (precision, recall)
        print(f"{category} done")

    ranking = pd.DataFrame(rows)
    ranking.to_csv(out_dir / "part4_ranking.csv", index=False)
    plotCurves(curves, args.categories, out_dir / "part4_pr_curves.png")

    pd.set_option("display.width", 200)
    print(ranking.drop(columns="category").groupby("method", sort=False).mean().round(3).to_string())
    print(f"\nResults written to {out_dir.resolve()}")
