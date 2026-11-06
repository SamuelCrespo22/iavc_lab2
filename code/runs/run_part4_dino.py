"""Part 4, Stage B: frozen DINOv2 ViT-S/14 features + linear SVM.

Evaluates the encodings written by `python -m src.extract_deep -m dinov2_vits14`:

  dinov2_vits14_cls        global image embedding (CLS token, 384-d)
  dinov2_vits14_patchmean  mean of the patch tokens (384-d; optional extension)

plus, for comparison when available, the torchvision ViT-B/32 class token and
the CLIP image embedding. Features are L2-normalized, as in Stage A. Trains the
Part 1 linear SVM per category and reports per-class AP and mAP.

Confusion matrix (DINOv2 encodings only): each test image is assigned to the category whose SVM gives
the highest score, or to "background" when every score is negative. Images of
several categories (e.g. horse + person) have no single true class and are left
out; their number is reported. The one-vs-background SVM scores are not
calibrated across categories, which this argmax ignores.

Results: <out>/part4_dino_grid.csv, <out>/part4_dino_encodings.csv,
<out>/part4_dino_confusion_<encoding>.csv / .png.

Usage (from the directory containing README.md):
    python -m runs.run_part4_dino
    python -m runs.run_part4_dino --compare
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from runs.run_part1 import CATEGORIES, loadData, trainAndEvaluate
from runs.run_part2 import ENCODINGS_DIR
from runs.run_part4_resnet import deepEncodingInfo, deepTransform, summarize

PROTOCOL_DIR = Path("data/splits_protocol")
DINO_ENCODINGS = ["dinov2_vits14_cls", "dinov2_vits14_patchmean"]

def runGrid(encodings, categories, C):
    rows, models = [], {}
    for encoding in encodings:
        for category in categories:
            res, models[encoding, category] = trainAndEvaluate(
                loadData(category, encoding), deepTransform, C, return_model=True)
            rows.append({"encoding": encoding, "feature_map": "l2", "category": category, **res})
            print(f"{encoding:24s} {category:10s} dim={res['dim']:4d} "
                  f"AP train={res['ap_train']:.3f} val={res['ap_val']:.3f} test={res['ap_test']:.3f}")
    return pd.DataFrame(rows), models


def loadMultiClassTest(encoding, categories):
    """Features and label sets of every test image of the categories and background."""
    labels = {}
    for cat in categories + ["background"]:
        with open(PROTOCOL_DIR / f"{cat}_test.txt", "r") as f:
            for name in (l.strip() for l in f if l.strip()):
                labels.setdefault(name, set()).add(cat)
    feats = {}
    for cat in categories + ["background"]:
        # Protocol test images come from the supplied *_val encodings.
        with np.load(ENCODINGS_DIR / encoding / f"{cat}_val.npz") as data:
            for name, h in zip(data["file_names"], data["histograms"]):
                if name in labels:
                    feats[name] = h
    names = sorted(labels)
    return names, np.stack([feats[n] for n in names]), [labels[n] for n in names]


def singleLabels(label_sets):
    """``(keep, true)``: images with at most one category and their class name."""
    true = [next(iter(s - {"background"} or s)) if len(s - {"background"}) <= 1 else None
            for s in label_sets]
    keep = np.array([t is not None for t in true])
    return keep, np.array([t for t in true if t is not None])


def saveConfusion(cm, classes, title, path):
    """Write the confusion matrix as <path>.csv and <path>.png."""
    pd.DataFrame(cm, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="pred")).to_csv(
        path.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def confusionMatrix(encoding, models, categories, out_dir):
    _, x, label_sets = loadMultiClassTest(encoding, categories)
    x = deepTransform(x)
    scores = np.stack([x @ models[encoding, c][0] + models[encoding, c][1] for c in categories], axis=1)

    keep, true = singleLabels(label_sets)
    scores = scores[keep]
    best = np.array(categories)[np.argmax(scores, axis=1)]
    pred = np.where(scores.max(axis=1) > 0, best, "background")

    classes = categories + ["background"]
    cm = confusion_matrix(true, pred, labels=classes)
    saveConfusion(cm, classes, f"{encoding}: test confusion matrix",
                  out_dir / f"part4_dino_confusion_{encoding}")

    objects = true != "background"
    print(f"\n=== {encoding}: confusion matrix on {keep.sum()} test images "
          f"({(~keep).sum()} multi-category images left out) ===")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print(classification_report(true, pred, labels=classes, zero_division=0))
    print(f"Accuracy over the 5 categories only (argmax, no background): "
          f"{accuracy_score(true[objects], best[objects]):.3f}")


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4 Stage B: frozen DINOv2 features + linear SVM.")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--compare", nargs="*", default=["vit_b_32", "clip_vit_b_32"],
                        help="Other src.extract_deep encodings to evaluate alongside DINOv2")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (fixed, as in Part 1)")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [e for e in DINO_ENCODINGS if not (ENCODINGS_DIR / e / "meta.json").exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}: run `python -m src.extract_deep -m dinov2_vits14 --flip` first")
    encodings = DINO_ENCODINGS + [e for e in args.compare if (ENCODINGS_DIR / e / "meta.json").exists()]

    info = pd.DataFrame([deepEncodingInfo(e) for e in encodings])
    info.to_csv(out_dir / "part4_dino_encodings.csv", index=False)

    grid, models = runGrid(encodings, args.categories, args.C)
    grid.to_csv(out_dir / "part4_dino_grid.csv", index=False)

    summarize(grid, info)
    for encoding in DINO_ENCODINGS:
        confusionMatrix(encoding, models, args.categories, out_dir)
    print(f"\nResults written to {out_dir.resolve()}")
