"""Part 4, Stage E: CLIP zero-shot classification and retrieval.

Image embeddings are the cached `clip_vit_b_32` encodings (src.extract_deep,
official CLIP image preprocessing). Each prompt goes through clip.tokenize and
the CLIP text encoder; when a class has several prompts, their unit-norm
embeddings are averaged (prompt ensembling). Similarity is the cosine between
the L2-normalized image and class embeddings.

  Prompts         every set in PROMPT_SETS is scored on the validation split
                  only; the best one (mean validation AP) is the only one
                  evaluated on the test set.
  Retrieval       per category, the protocol subset (category positives vs.
                  background, the same images as every SVM experiment) is ranked
                  by similarity to the category's text embedding -> AP / mAP.
  Classification  each test image goes to the most similar of the 5 category
                  embeddings and a "background scene" embedding: confusion
                  matrix and per-class precision / recall / F1. Multi-category
                  images are left out, as in run_part4_dino.
  Comparison      the linear SVM of Part 1 on the same CLIP image embeddings
                  (L2, C = 10), trained with task labels.

Results (in --out): part4_clip_prompts.csv, part4_clip_test.csv,
part4_clip_classification.csv, part4_clip_confusion.csv / .png, part4_clip_costs.csv.

Usage:
    python -m runs.run_part4_clip
"""

import os

# torch must be imported before numpy in this env (see src.extract_deep).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import clip

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix

from runs.run_part1 import CATEGORIES, l2, loadData, trainAndEvaluate
from runs.run_part2 import ENCODINGS_DIR
from runs.run_part4_dino import loadMultiClassTest, saveConfusion, singleLabels

ENCODING = "clip_vit_b_32"
BACKGROUND = "background scene"

NAMES = {"aeroplane": "aeroplane", "motorbike": "motorbike", "person": "person", "horse": "horse",
         "car": "car", "background": BACKGROUND}
SYNONYMS = {"aeroplane": ["aeroplane", "airplane", "aircraft"], "motorbike": ["motorbike", "motorcycle"],
            "person": ["person", "man", "woman", "child"], "horse": ["horse", "pony"],
            "car": ["car", "automobile"], "background": [BACKGROUND]}
# {a}: noun with its article ("an aeroplane"), {n}: bare noun.
TEMPLATES = ["a photo of {a}.", "a blurry photo of {a}.", "a cropped photo of {a}.", "a close-up photo of {a}.",
             "a low resolution photo of {a}.", "a photo of the {n}.", "a photo of a small {n}.",
             "a photo of a large {n}."]


def withArticle(noun):
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


# Prompt set -> {class: [prompts]}. Chosen from prior knowledge, never from test labels.
PROMPT_SETS = {
    "name": {c: [n] for c, n in NAMES.items()},
    "photo": {c: [f"a photo of {withArticle(n)}."] for c, n in NAMES.items()},
    "templates": {c: [t.format(a=withArticle(n), n=n) for t in TEMPLATES] for c, n in NAMES.items()},
    "synonyms": {c: [f"a photo of {withArticle(s)}." for s in syns] for c, syns in SYNONYMS.items()},
}


def loadClip(device="cpu"):
    model, _ = clip.load("ViT-B/32", device=device)
    return model.eval()


def textEmbeddings(model, prompt_set):
    """``{class: unit-norm embedding}``; several prompts per class are averaged."""
    out = {}
    with torch.inference_mode():
        for cls, prompts in PROMPT_SETS[prompt_set].items():
            e = model.encode_text(clip.tokenize(prompts)).float()
            e = (e / e.norm(dim=-1, keepdim=True)).mean(dim=0)
            out[cls] = (e / e.norm()).numpy().astype(np.float64)
    return out


def zeroShotAP(data, texts, subset):
    """``{category: AP}`` ranking the subset by cosine similarity to the class embedding."""
    aps = {}
    for category, subsets in data.items():
        x, y, _ = subsets[subset]
        aps[category] = average_precision_score(y, l2(x.astype(np.float64)) @ texts[category])
    return aps


def comparePrompts(model, data):
    rows, texts = [], {}
    for prompt_set in PROMPT_SETS:
        texts[prompt_set] = textEmbeddings(model, prompt_set)
        ap = zeroShotAP(data, texts[prompt_set], "val")
        rows.append({"prompt_set": prompt_set, "prompts_per_class": len(PROMPT_SETS[prompt_set]["car"]),
                     **ap, "mAP_val": np.mean(list(ap.values()))})
    return pd.DataFrame(rows), texts


def classify(texts, categories, out_dir):
    """6-way argmax (5 categories + background) on single-label test images."""
    _, x, label_sets = loadMultiClassTest(ENCODING, categories)
    keep, true = singleLabels(label_sets)
    classes = categories + ["background"]
    sims = l2(x[keep].astype(np.float64)) @ np.stack([texts[c] for c in classes], axis=1)
    pred = np.array(classes)[sims.argmax(axis=1)]

    cm = confusion_matrix(true, pred, labels=classes)
    saveConfusion(cm, classes, "CLIP zero-shot: test confusion matrix", out_dir / "part4_clip_confusion")
    report = classification_report(true, pred, labels=classes, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out_dir / "part4_clip_classification.csv")

    objects = true != "background"
    five_way = np.array(categories)[sims[objects, :len(categories)].argmax(axis=1)]
    print(f"\n=== Zero-shot classification on {keep.sum()} test images "
          f"({(~keep).sum()} multi-category images left out) ===")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print(classification_report(true, pred, labels=classes, zero_division=0))
    print(f"Accuracy, 6-way (with background): {accuracy_score(true, pred):.3f}")
    print(f"Accuracy, 5-way on object images (no background prompt): "
          f"{accuracy_score(true[objects], five_way):.3f}")


def measureCosts(model, prompt_set, data):
    meta = json.loads((ENCODINGS_DIR / ENCODING / "meta.json").read_text())
    start = time.perf_counter()
    texts = textEmbeddings(model, prompt_set)
    text_time = time.perf_counter() - start

    x = np.concatenate([d["test"][0] for d in data.values()]).astype(np.float64)
    T = np.stack(list(texts.values()), axis=1)
    start = time.perf_counter()
    l2(x) @ T
    infer_us = 1e6 * (time.perf_counter() - start) / len(x)
    return pd.DataFrame([{"encoding": ENCODING, "dim": meta["dim"], "kb_per_image": meta["dim"] * 4 / 1024,
                          "extract_ms_per_image": meta["extract_ms_per_image"], "device": meta["device"],
                          "prompt_set": prompt_set, "num_prompts": sum(map(len, PROMPT_SETS[prompt_set].values())),
                          "text_encode_s": text_time, "infer_us_per_image": infer_us}])


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4 Stage E: CLIP zero-shot classification and retrieval.")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--C", type=float, default=10.0, help="C of the SVM on CLIP embeddings")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    if not (ENCODINGS_DIR / ENCODING / "meta.json").exists():
        raise SystemExit(f"Missing {ENCODING}: run `python -m src.extract_deep -m clip_vit_b_32 --flip` first")
    model = loadClip()
    data = {category: loadData(category, ENCODING) for category in args.categories}

    # Prompt selection on validation only.
    prompts, texts = comparePrompts(model, data)
    prompts.to_csv(out_dir / "part4_clip_prompts.csv", index=False)
    best = prompts.loc[prompts["mAP_val"].idxmax(), "prompt_set"]
    print("=== Validation AP per prompt set ===")
    print(prompts.round(3).to_string(index=False))
    print(f"Selected on validation: {best!r} -> {PROMPT_SETS[best]['car']}")

    # Test: selected prompts vs. a linear SVM on the same embeddings.
    test = pd.DataFrame({
        f"zero_shot ({best})": zeroShotAP(data, texts[best], "test"),
        "svm_on_clip": {c: trainAndEvaluate(d, l2, args.C)["ap_test"] for c, d in data.items()},
    })
    test.loc["mAP"] = test.mean()
    test.index.name = "category"
    test.to_csv(out_dir / "part4_clip_test.csv")
    print("\n=== Test AP: zero-shot (selected prompts) vs. linear SVM on CLIP embeddings ===")
    print(test.round(3).to_string())

    classify(texts[best], args.categories, out_dir)

    costs = measureCosts(model, best, data)
    costs.to_csv(out_dir / "part4_clip_costs.csv", index=False)
    print("\n=== Costs ===")
    print(costs.round(4).to_string(index=False))
    print(f"\nResults written to {out_dir.resolve()}")
