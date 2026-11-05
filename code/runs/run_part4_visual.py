"""Part 4, Stage F: correct / incorrect examples and class-relevant regions.

For each method and category, the test images (category positives vs.
background) are split by the method's decision and three groups are shown:

  TP  correctly detected positives, highest scores first
  FP  background images wrongly detected, highest scores first
  FN  missed positives, lowest scores first

Decision: SVM score above 0; for CLIP, the category's prompt being the most
similar of the 6 class prompts (the Stage E classification). Only real errors
are shown, so a category can have fewer than --num FP or FN.

The SVMs are the Part 1 linear SVM (L2 feature map, C = 10), the settings
selected on validation in Stages A / B. The region maps
are exact decompositions of the SVM score, so they show what the classifier
used, not an approximation of it:

  resnet18  SVM on L2(GAP(layer4)). GAP is linear, so the score minus the bias
            is the mean over the 7x7 positions of  w . f(x, y) / ||GAP(f)||:
            a class activation map built from the SVM weights.
  dinov2    SVM on L2(mean of the patch tokens) (`dinov2_vits14_patchmean`).
            The score minus the bias is the mean over the 16x16 patches of
            w . t_p / ||mean(t)||. The CLS-token SVM of Stage B has no such
            decomposition, so it is not used here.
  clip      zero-shot cosine similarity to the class text embedding (prompt set
            selected on validation by run_part4_clip). Examples only, no map.

Only positive contributions are drawn
(evidence for the category), on the 224x224 centre crop the models see, with
one colour scale per category. The script checks that every map averages back
to its SVM score.

Results (in --out): part4_examples_<method>.png and part4_examples.csv.

Usage (from the directory containing README.md):
    python -m runs.run_part4_visual
    python -m runs.run_part4_visual --methods resnet18 --categories horse --num 4
"""

import os

# torch must be imported before numpy in this env (see src.extract_deep).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src import utils
from src.deep_models import IntermediateFeatureExtractor, getModel
from runs.run_part1 import CATEGORIES, l2, loadData
from runs.run_part4_clip import ENCODING as CLIP_ENCODING, loadClip, textEmbeddings

IMG_DIR = Path("data/images")
# Both models resize to 256 and centre-crop 224: the region the maps cover.
CROP = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
GROUPS = ["TP", "FP", "FN"]


@lru_cache(maxsize=None)
def loadModel(model_name):
    model, preprocess = getModel(model_name)
    return (IntermediateFeatureExtractor(model) if model_name == "resnet18" else model), preprocess


def resnetMaps(names, W):
    extractor, preprocess = loadModel("resnet18")
    with torch.inference_mode():
        _, blocks = extractor(loadBatch(names, preprocess))
    f = blocks["layer4.1"].double().numpy()                     # (B, 512, 7, 7)
    norm = np.linalg.norm(f.mean(axis=(2, 3)), axis=1)
    return np.einsum("c,bchw->bhw", W, f) / norm[:, None, None]


def dinoMaps(names, W):
    model, preprocess = loadModel("dinov2_vits14")
    with torch.inference_mode():
        t = model.forward_features(loadBatch(names, preprocess))["x_norm_patchtokens"].double().numpy()
    norm = np.linalg.norm(t.mean(axis=1), axis=1)               # t: (B, 256, 384)
    side = int(np.sqrt(t.shape[1]))
    return ((t @ W) / norm[:, None]).reshape(-1, side, side)


# method -> (cached encoding, map function or None)
METHODS = {
    "resnet18": ("resnet18_layer4", resnetMaps),
    "dinov2": ("dinov2_vits14_patchmean", dinoMaps),
    "clip": (CLIP_ENCODING, None),
}


def loadBatch(names, preprocess):
    return torch.stack([preprocess(Image.open(IMG_DIR / f"{n}.jpg").convert("RGB")) for n in names])


def testScores(method, category, clip_texts, C):
    """Test labels, names, scores and predicted-positive flags of one method; plus (W, b) for the SVMs.

    SVM: predicted positive when the score is above 0. CLIP: when the category's prompt is the
    most similar of all class prompts (background included), as in the Stage E classification.
    """
    data = loadData(category, METHODS[method][0])
    x_test, y_test, names = data["test"]
    x_test = l2(x_test.astype(np.float64))
    if method == "clip":
        classes = list(clip_texts)
        sims = x_test @ np.stack([clip_texts[c] for c in classes], axis=1)
        pred = np.array(classes)[sims.argmax(axis=1)] == category
        return y_test, names, sims[:, classes.index(category)], pred, None
    x_train, y_train, _ = data["train"]
    W, b = utils.train_svm(l2(x_train.astype(np.float64)), y_train, C=C)
    scores = x_test @ W + b
    return y_test, names, scores, scores > 0, (W, b)


def pickExamples(y, scores, pred, num):
    """``{group: indices}``: real errors only, the most confident first."""
    order = np.argsort(-scores)
    pos, neg = y[order] > 0, y[order] < 0
    return {"TP": order[pos & pred[order]][:num],
            "FP": order[neg & pred[order]][:num],
            "FN": order[pos & ~pred[order]][::-1][:num]}


def collect(method, categories, clip_texts, num, C):
    """Rows (category, group, rank, name, score) and the region maps, if any."""
    rows, maps = [], {}
    for category in categories:
        y, names, scores, pred, model = testScores(method, category, clip_texts, C)
        picked = pickExamples(y, scores, pred, num)
        for group in GROUPS:
            for rank, i in enumerate(picked[group]):
                rows.append({"method": method, "category": category, "group": group, "rank": rank,
                             "name": names[i], "score": scores[i]})
        map_fn = METHODS[method][1]
        if map_fn is not None:
            idx = np.concatenate([picked[g] for g in GROUPS]).astype(int)
            W, b = model
            m = map_fn(names[idx], W)
            # Sanity check: each map averages back to the SVM score.
            gap = np.abs(m.mean(axis=(1, 2)) + b - scores[idx]).max()
            assert gap < 1e-3, f"{method}/{category}: map and SVM score differ by {gap:.2g}"
            maps.update({(category, names[i]): mi for i, mi in zip(idx, m)})
        print(f"{method}: {category} done")
    return pd.DataFrame(rows), maps


def plotExamples(method, examples, maps, path):
    categories = list(dict.fromkeys(examples["category"]))
    num = examples["rank"].max() + 1
    fig, axes = plt.subplots(len(categories), 3 * num, figsize=(1.7 * 3 * num, 1.9 * len(categories)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")  # a category can have fewer than `num` errors of a kind
    for r, category in enumerate(categories):
        rows = examples[examples["category"] == category]
        if maps:
            # One colour scale per category: the SVMs of different categories have different score ranges.
            vmax = np.percentile(np.concatenate([np.maximum(maps[category, n], 0).ravel()
                                                 for n in rows["name"]]), 99)
        for row in rows.itertuples():
            ax = axes[r][GROUPS.index(row.group) * num + row.rank]
            ax.axis("on")
            ax.imshow(CROP(Image.open(IMG_DIR / f"{row.name}.jpg").convert("RGB")))
            if maps:
                ax.imshow(np.maximum(maps[category, row.name], 0), cmap="jet", alpha=0.45, vmin=0, vmax=vmax,
                          extent=(0, 224, 224, 0), interpolation="bilinear")
            ax.set_title(f"{row.group} {row.score:+.2f}", fontsize=8,
                         color="tab:green" if row.group == "TP" else "tab:red")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(category, fontsize=10)
    score = "cosine similarity" if method == "clip" else "SVM score"
    fig.suptitle(f"{method} ({score}): TP correct detections | FP background wrongly detected | "
                 f"FN missed positives")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def selectedPromptSet(out_dir):
    prompts = out_dir / "part4_clip_prompts.csv"
    if not prompts.exists():
        raise SystemExit("Missing part4_clip_prompts.csv: run `python -m runs.run_part4_clip` first")
    table = pd.read_csv(prompts)
    return table.loc[table["mAP_val"].idxmax(), "prompt_set"]


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4 Stage F: examples and class-relevant regions.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--num", type=int, default=3, help="Images per group (TP / FP / FN)")
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (as in Part 1)")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_texts = None
    if "clip" in args.methods:
        prompt_set = selectedPromptSet(out_dir)
        print(f"CLIP prompt set (selected on validation): {prompt_set}")
        clip_texts = textEmbeddings(loadClip(), prompt_set)

    tables = []
    for method in args.methods:
        examples, maps = collect(method, args.categories, clip_texts, args.num, args.C)
        plotExamples(method, examples, maps, out_dir / f"part4_examples_{method}.png")
        tables.append(examples)
    examples = pd.concat(tables, ignore_index=True)
    examples.to_csv(out_dir / "part4_examples.csv", index=False)

    pd.set_option("display.width", 200)
    print(examples.round(3).to_string(index=False))
    print(f"\nResults written to {out_dir.resolve()}")
