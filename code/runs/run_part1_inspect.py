"""Part 1, Stages A-D: example images, ranked lists, precision-recall curves and visual words.

  Stage A  a few training images per category and of the background
           (part1_samples.png)
  Stage B  the motorbike SVM ranks its own training images: top and bottom of
           the list (part1_ranked_train.png); the visual words that contribute
           most to the score of the top-ranked test image and of the
           highest-ranked background test image (part1_words.png)
  Stage C  the same SVM ranks the test images (part1_ranked_test.png)
  C, D     precision-recall curves on train and test, every category
           (part1_pr_curves.png)

Setting: the K = 512 vocabulary trained on our protocol (bovw_K512_C0), 2x2 tiles and the Hellinger
map, the setting with the best mean validation AP in run_part1. Linear SVM,
C = 10, positives vs. background, protocol splits.

Usage:
    python -m runs.run_part1_inspect
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve

from src import utils
from src.encoders import ImageEncoder
from src.features import standarizeImage
from src.visualization import RelevantVisualWordsDisplay
from runs.run_part1 import CATEGORIES, loadData, prepare

ENCODING = "bovw_K512_C0"
IMG_DIR = Path("data/images")
COLOURS = ["red", "lime", "cyan", "yellow"]


def trainCategory(category, C):
    """Data, SVM weights and the train / test scores of one category (tiles, Hellinger)."""
    data = loadData(category, ENCODING)
    transform = lambda h: prepare(h, True, "sqrt_l2")
    x_train, y_train, _ = data["train"]
    W, b = utils.train_svm(transform(x_train), y_train, C=C)
    scores = {s: transform(data[s][0]) @ W + b for s in ["train", "test"]}
    return data, W, scores


def plotSamples(num, path, seed=0):
    rng = np.random.default_rng(seed)
    _, y, names = loadData(CATEGORIES[0], ENCODING)["train"]
    classes = {"background": names[y < 0]}
    for category in CATEGORIES:
        _, y, names = loadData(category, ENCODING)["train"]
        classes[category] = names[y > 0]
    fig, axes = plt.subplots(len(classes), num, figsize=(1.8 * num, 1.8 * len(classes)), squeeze=False)
    for row, (label, names) in zip(axes, classes.items()):
        for ax, name in zip(row, rng.choice(names, size=num, replace=False)):
            ax.imshow(Image.open(IMG_DIR / f"{name}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label)
    fig.suptitle("Random training images per class")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plotRanked(scores, labels, names, num, title, path):
    """Top and bottom ``num`` images of a ranking; green title = positive, red = background."""
    order = np.argsort(-scores)
    ranks = {"top": np.arange(num), "bottom": np.arange(len(order) - num, len(order))}
    fig, axes = plt.subplots(2, num, figsize=(1.8 * num, 4.4), squeeze=False)
    for row, (label, positions) in zip(axes, ranks.items()):
        for ax, rank in zip(row, positions):
            i = order[rank]
            ax.imshow(Image.open(IMG_DIR / f"{names[i]}.jpg"))
            ax.set_title(f"#{rank + 1} {scores[i]:+.2f}", fontsize=8,
                         color="tab:green" if labels[i] > 0 else "tab:red")
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(label)
    fig.suptitle(f"{title} (green = positive, red = background)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plotWords(images, W, num_words, path):
    """For each image: the keypoints of its top words, then one patch mosaic per word."""
    encoder = ImageEncoder(Path("data/encoders") / f"{ENCODING}.npz")
    fig, axes = plt.subplots(len(images), num_words + 1, figsize=(3.2 * (num_words + 1), 3.2 * len(images)),
                             squeeze=False)
    for row, (label, name) in zip(axes, images.items()):
        img = standarizeImage(str(IMG_DIR / f"{name}.jpg"))
        words = RelevantVisualWordsDisplay(img, W, encoder)  # words sorted by w * Hellinger histogram
        row[0].imshow(img)
        row[0].set_title(label, fontsize=9)
        for rank in range(num_words):
            word = words.perm[rank]
            idx = np.flatnonzero(words.words == word)
            kpts = words.kpts[idx]  # (y, x, ...)
            row[0].scatter(kpts[:, 1], kpts[:, 0], s=8, color=COLOURS[rank])
            row[rank + 1].imshow(words.build_mosaic(kpts, words.distances[idx], img.shape[:2]))
            row[rank + 1].set_title(f"word {word}: {words.scores[rank]:+.3f}", fontsize=9, color=COLOURS[rank])
        for ax in row:
            ax.axis("off")
    fig.suptitle("Visual words with the largest contribution to the motorbike score (patches of each word)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plotCurves(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.6), sharey=True, squeeze=False)
    for ax, (category, (data, _, scores)) in zip(axes[0], results.items()):
        for subset, style in [("train", "--"), ("test", "-")]:
            y = data[subset][1]
            precision, recall, _ = precision_recall_curve(y, scores[subset])
            ax.plot(recall, precision, style, label=f"{subset} AP = {average_precision_score(y, scores[subset]):.3f}")
        ax.set_title(category)
        ax.set_xlabel("Recall")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 1 Stages A-D: images, rankings, PR curves, visual words.")
    parser.add_argument("--category", default="motorbike", help="Category of Stages B and C")
    parser.add_argument("--num", type=int, default=8, help="Images per row of the ranked lists")
    parser.add_argument("--words", type=int, default=3, help="Visual words shown per image")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (as in run_part1)")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotSamples(args.num, out_dir / "part1_samples.png")

    results = {category: trainCategory(category, args.C) for category in CATEGORIES}
    plotCurves(results, out_dir / "part1_pr_curves.png")

    data, W, scores = results[args.category]
    for subset in ["train", "test"]:
        _, y, names = data[subset]
        plotRanked(scores[subset], y, names, args.num, f"{args.category} SVM, {subset} images ranked by score",
                   out_dir / f"part1_ranked_{subset}.png")

    _, y, names = data["test"]
    order = np.argsort(-scores["test"])
    top_background = next(i for i in order if y[i] < 0)
    images = {f"top-ranked test image (#1, {'positive' if y[order[0]] > 0 else 'background'})": names[order[0]],
              f"top-ranked background test image (#{np.flatnonzero(order == top_background)[0] + 1})":
                  names[top_background]}
    plotWords(images, W, args.words, out_dir / "part1_words.png")

    for category, (data, _, s) in results.items():
        y = data["test"][1]
        print(f"{category:10s} test AP = {average_precision_score(y, s['test']):.3f}, "
              f"positives in the top 36 = {(y[np.argsort(-s['test'])[:36]] > 0).sum()}")
    print(f"\nResults written to {out_dir.resolve()}")
