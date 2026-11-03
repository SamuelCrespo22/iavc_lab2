"""Build one Markdown report per part from the CSVs and figures in results/.

Reads what the runs/ scripts wrote and writes results/report_part<N>.md with the
summary tables, the figures and the assignment questions of that part. Parts or
stages whose results are missing are listed as pending, with the command that
produces them. Nothing is recomputed: rerun a run_part script to update a table.

Usage (from the directory containing README.md):
    python -m runs.report
    python -m runs.report --parts 1 3
"""

import argparse
from pathlib import Path

import pandas as pd

QUESTIONS = {
    1: {
        "A": ["How do you choose the number of visual words for BoVW?",
              "How does spatial tiling encode useful layout information? When might it reduce generalization?",
              "How does vocabulary size affect representation capacity, quantization error, computation and storage?"],
        "B": ["What do you anticipate the results to be? What do you predict will occur?",
              "Do the highest-ranked images share object features, contextual cues or background patterns?",
              "Why can a classifier achieve high training performance but fail on new images?"],
        "C": ["Why is the bias term irrelevant to the ordering of images scored by one fixed binary linear classifier?",
              "How does the ordering of positive and negative examples affect the precision-recall curve and AP?"],
        "D": ["How do intra-class variation, sample count, object scale and background diversity help explain differences in AP?"],
        "E": ["How does L2 normalization affect the self-similarity of a nonzero feature vector under a linear kernel?",
              "Regarding the L2 normalized BoVW histogram h, what can be inferred about its self-similarity K(h,h)?",
              "How does K(h,h) of an L2 normalized histogram compare to K(h,h') of two distinct L2 normalized histograms?",
              "Would the same hold true for unnormalized or L1 normalized histograms?",
              "Is there a correlation between classification performance and the use of L2 normalization?",
              "Why can removing spatial information help some classes but hurt others?"],
        "F": ["How should the BoVW histograms be normalized for the Hellinger kernel classifier? Before or after the square root?",
              "Why does the square-root mapping change the relative contribution of frequent and rare visual words?"],
        "G": ["What performance do you get with the linear kernel? And with the Hellinger kernel?",
              "Does performance saturate? Would collecting more examples or improving their diversity likely help more?"],
    },
    2: {
        "H": ["For K visual words and D-dimensional local descriptors, compare the nominal dimensions of BoVW (K) "
              "and VLAD (K x D). What changes in representation capacity and storage?"],
        "I": ["What additional information does a Fisher Vector encode compared with VLAD, and what computational "
              "costs does it introduce?"],
    },
    3: {
        "": ["Does a larger codebook improve recognition before PCA? For which classes or visual patterns?",
             "Does PCA retain the gain from the larger codebook, and how much dimensionality reduction is possible "
             "before performance drops?",
             "At a similar final feature dimension, does a large codebook plus PCA outperform a directly small codebook?",
             "What is the trade-off between accuracy/mAP, memory, feature extraction time, and classifier inference time?",
             "Does the answer change for BoVW versus VLAD/Fisher Vector, and why might their dimensionality behave differently?"],
    },
    4: {
        "A": ["How do learned features differ from handcrafted local descriptors? Does a higher-dimensional feature "
              "necessarily give better AP?"],
        "B": ["Why can DINOv2 features be useful with few labeled examples even though DINOv2 is not trained as a "
              "classifier for these target categories?",
              "What information might be lost when patch tokens are replaced by a single global descriptor?"],
        "C": ["How much does pretrained representation quality reduce dependence on labeled examples?",
              "When could horizontal flipping harm recognition? Is averaging feature vectors equivalent to averaging "
              "classifier scores?"],
        "D": ["How does regularization affect the bias-variance trade-off when only a few positive examples are available?"],
        "E": ["How does CLIP zero-shot recognition differ from a linear SVM trained on frozen CLIP image embeddings?",
              "How sensitive are predictions to prompt wording? What are the risks of selecting prompts based on the test set?"],
        "F": ["Do the highlighted regions correspond to the object? What can and cannot be concluded from a saliency "
              "visualization?",
              "Which categories and visual conditions remain difficult, and what evidence supports your explanation?"],
    },
}


def md(df, floatfmt=".3f"):
    # Named index levels become ordinary columns, so each gets its own header.
    if any(df.index.names):
        return df.reset_index().to_markdown(index=False, floatfmt=floatfmt)
    return df.to_markdown(floatfmt=floatfmt)


def labelSpatial(grid):
    """'spatial' True/False -> '2x2 tiles' / 'global' (single histogram, tiles summed)."""
    return grid.assign(spatial=grid["spatial"].map({True: "2x2 tiles", False: "global"}))


def meanStd(series):
    return f"{series.mean():.3f} ± {series.std():.3f}" if len(series) > 1 else f"{series.mean():.3f}"


def withMAP(table):
    table = table.copy()
    table["mAP"] = table.mean(axis=1)
    return table


def selectOnVal(grid, setting):
    """Rows of the setting with the best mean val AP over categories, per encoding."""
    mean_val = grid.groupby(["encoding", *setting], sort=False)["ap_val"].mean()
    best = mean_val.groupby(level="encoding", sort=False).idxmax().tolist()
    return grid.set_index(["encoding", *setting]).loc[best].reset_index()


def questions(part):
    lines = ["## Questions", ""]
    for stage, qs in QUESTIONS[part].items():
        if stage:
            lines.append(f"**Stage {stage}**\n")
        lines += [f"- {q}\n  - *Answer:*" for q in qs] + [""]
    return lines


def pending(what, command):
    return [f"> **Pending:** {what} — run `{command}`.", ""]


def read(results, name):
    path = results / name
    return pd.read_csv(path) if path.exists() else None


################################################################################

def reportPart1(results):
    lines = ["# Part 1 — Handcrafted features: BoVW and linear SVM", ""]
    grid = read(results, "part1_grid.csv")
    if grid is None:
        return lines + pending("Part 1 grid", "python -m runs.run_part1") + questions(1)
    grid = labelSpatial(grid)

    lines += ["Linear SVM (C fixed) per category, positives vs. background, on the protocol splits. "
              "Values are test AP; mAP is the mean over categories.", ""]

    lines += ["## Stages A, E, F — vocabulary size, spatial tiling and feature map", "",
              "Test mAP. Rows: vocabulary size K and spatial layout (`2x2 tiles`: 4 concatenated "
              "histograms, dimension 4K; `global`: one histogram, dimension K). Columns: feature map "
              "(normalization); `sqrt_l2` is the Hellinger map.", ""]
    lines += [md(grid.pivot_table(index=["K", "spatial"], columns="feature_map", values="ap_test")), ""]

    k = 512 if 512 in grid["K"].values else grid["K"].max()
    sub = grid[grid["K"] == k]
    lines += [f"## Stages D, E — per-category test AP (K = {k})", ""]
    lines += [md(withMAP(sub.pivot_table(index=["spatial", "feature_map"], columns="category",
                                         values="ap_test"))), ""]

    lines += [f"## Stage E — effect of spatial tiling per category (K = {k}, Hellinger)", ""]
    spatial = sub[sub["feature_map"] == "sqrt_l2"].pivot_table(index="category", columns="spatial",
                                                               values="ap_test")
    spatial["tiles - global"] = spatial["2x2 tiles"] - spatial["global"]
    lines += [md(spatial), ""]

    lines += ["## Stages B, C, D — setting selected on validation AP, per category", "",
              "A large train–test gap indicates overfitting.", ""]
    best = grid.loc[grid.groupby("category")["ap_val"].idxmax()]
    best = best[["category", "K", "spatial", "feature_map", "dim", "ap_train", "ap_val", "ap_test"]]
    best["train - test"] = best["ap_train"] - best["ap_test"]
    lines += [md(best.set_index("category")), "", f"mAP (test) of the selected settings: "
              f"**{best['ap_test'].mean():.3f}**", ""]

    lines += ["## Stage G — training set size", ""]
    fractions = read(results, "part1_fractions.csv")
    if fractions is None:
        lines += pending("Stage G", "python -m runs.run_part1 --skip-grid")
    else:
        lines += ["Validation AP, mean ± std over seeds (stratified subsets). "
                  "`l2` = linear kernel, `sqrt_l2` = Hellinger.", ""]
        table = (fractions.groupby(["category", "feature_map", "fraction"])["ap_val"]
                 .apply(meanStd).unstack("fraction"))
        lines += [md(table), ""]
        sizes = fractions.groupby("fraction")[["num_train", "num_pos"]].mean().round(0).astype(int)
        lines += ["Mean training set size per fraction (positives in `num_pos`):", "", md(sizes), ""]
        for png in sorted(results.glob("part1_fractions_*.png")):
            lines += [f"![{png.stem}]({png.name})", ""]

    lines += ["## Stages B, C — visual inspection", "",
              "Ranked image lists, precision–recall curves and relevant visual words are in "
              "`1_handcraft.ipynb`.", ""]
    return lines + questions(1)


def reportPart2(results):
    lines = ["# Part 2 — Advanced encodings: VLAD and Fisher Vectors", ""]
    grid = read(results, "part2_grid.csv")
    if grid is None:
        return lines + pending("Part 2 grid", "python -m runs.run_part2") + questions(2)
    grid = labelSpatial(grid)

    lines += ["Same linear SVM and splits as Part 1. Values are test AP / mAP.", ""]
    lines += ["## Normalization and spatial tiling per encoding", "",
              "`ssr` = signed square root, `intra` = per-word L2 (VLAD).", ""]
    lines += [md(grid.pivot_table(index=["encoding", "spatial", "feature_map"], values="ap_test",
                                  sort=False).rename(columns={"ap_test": "mAP"})), ""]

    selected = selectOnVal(grid, ["spatial", "feature_map"])
    lines += ["## Selected setting per encoding (on mean validation AP): per-class test AP", ""]
    lines += [md(withMAP(selected.pivot_table(index=["encoding", "spatial", "feature_map"],
                                              columns="category", values="ap_test", sort=False))), ""]

    info = read(results, "part2_encodings.csv")
    if info is not None:
        costs = selected.groupby("encoding", sort=False).agg(
            train_time_s=("train_time_s", "mean"), infer_ms_per_image=("infer_ms_per_image", "mean"),
            mAP=("ap_test", "mean"))
        lines += ["## Dimension, storage and cost", "",
                  "`codebook_fit_s`: PCA + k-means/GMM (NaN for encoders trained before it was recorded). "
                  "SIFT and encoding times are per image, on a few sample images.", ""]
        lines += [md(info.set_index("encoding").join(costs)), ""]
    return lines + questions(2)


def reportPart3(results):
    lines = ["# Part 3 — Large visual vocabularies with PCA vs. smaller vocabularies", ""]
    grid = read(results, "part3_grid.csv")
    if grid is None:
        return lines + pending("Part 3 grid", "python -m runs.run_part3") + questions(3)

    lines += ["BoVW uses global histograms (dimension = K) with the Hellinger map; VLAD / FV use the "
              "Part 2 setting. PCA is fit on the training features only and followed by L2. "
              "Each seed trains on a different stratified subset of the training set.", ""]

    per_seed = (grid.groupby(["encoding", "pca_dim", "seed"], sort=False)
                .agg(mAP=("ap_test", "mean"), acc=("acc_test", "mean"), f1=("f1_test", "mean"),
                     orig_dim=("orig_dim", "first"), dim=("dim", "first"), pca_fit_s=("pca_fit_s", "mean"),
                     train_s=("train_time_s", "mean"), infer_ms=("infer_ms_per_image", "mean")))
    table = per_seed.groupby(level=["encoding", "pca_dim"], sort=False).agg(
        orig_dim=("orig_dim", "first"), dim=("dim", "first"), mAP=("mAP", meanStd),
        acc=("acc", "mean"), macro_f1=("f1", "mean"), pca_fit_s=("pca_fit_s", "mean"),
        train_s=("train_s", "mean"), infer_ms=("infer_ms", "mean"))
    table["removed_%"] = 100 * (1 - table["dim"] / table["orig_dim"])
    lines += ["## Recognition, compactness and efficiency", "",
              "`pca_dim` 0 = no PCA. mAP is mean ± std over seeds; accuracy and macro-F1 use the SVM "
              "threshold 0.", ""]
    lines += [md(table), ""]

    mAP = per_seed.groupby(level=["encoding", "pca_dim"], sort=False)["mAP"].mean().reset_index()
    mAP = mAP.merge(table.reset_index()[["encoding", "pca_dim", "orig_dim"]])
    bovw = mAP[mAP["encoding"].str.startswith("bovw")]
    matched = []
    for d in sorted(bovw.loc[bovw["pca_dim"] > 0, "pca_dim"].unique()):
        row = {"final_dim": d}
        direct = bovw[(bovw["pca_dim"] == 0) & (bovw["orig_dim"] == d)]
        row["D: K = d, no PCA"] = direct["mAP"].iloc[0] if len(direct) else float("nan")
        for _, r in bovw[bovw["pca_dim"] == d].iterrows():
            row[f"C: K = {r['orig_dim']} + PCA"] = r["mAP"]
        matched.append(row)
    lines += ["## Conditions C vs. D — same final dimension (BoVW, test mAP)", ""]
    lines += [md(pd.DataFrame(matched).set_index("final_dim")), ""]

    lines += ["## Per-category test AP", ""]
    lines += [md(grid.pivot_table(index=["encoding", "pca_dim"], columns="category",
                                  values="ap_test", sort=False)), ""]

    info = read(results, "part3_encodings.csv")
    if info is not None:
        lines += ["## Encoding cost and storage", "", md(info.set_index("encoding")), ""]

    lines += ["## Quantization error", ""]
    quant = read(results, "quantization.csv")
    if quant is None:
        lines += pending("quantization error", "python -m runs.quantization")
    else:
        lines += ["`rel_qerr_sift`: mean squared distance to the nearest word in 128-d SIFT space / total "
                  "SIFT variance (comparable across C0 and C80). Test images were never seen by any codebook.", ""]
        cols = ["rel_qerr_sift", "qerr_sift", "unused_words_%", "pca_var_%"]
        lines += [md(quant.pivot_table(index=["type", "pca_dim", "K"], columns="subset", values=cols)), ""]
        if (results / "quantization.png").exists():
            lines += ["![quantization](quantization.png)", ""]
    return lines + questions(3)


def deepSection(results, grid_name, info_name, title, command):
    lines = [f"## {title}", ""]
    grid = read(results, grid_name)
    if grid is None:
        return lines + pending(title, command), None
    lines += ["Test AP per encoding and feature map:", ""]
    lines += [md(withMAP(grid.pivot_table(index=["encoding", "feature_map"], columns="category",
                                          values="ap_test", sort=False))), ""]
    selected = selectOnVal(grid, ["feature_map"])
    costs = selected.groupby("encoding", sort=False).agg(
        feature_map=("feature_map", "first"), train_time_s=("train_time_s", "mean"),
        infer_ms_per_image=("infer_ms_per_image", "mean"), mAP=("ap_test", "mean"))
    info = read(results, info_name)
    summary = info.set_index("encoding").join(costs) if info is not None else costs
    lines += ["Selected feature map per encoding (on mean validation AP), with dimension, storage and cost "
              "(`extract_ms_per_image`: forward pass for the deep models, SIFT + encoding for the baselines):", "",
              md(summary), ""]
    return lines, grid


def reportPart4(results):
    lines = ["# Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP", ""]

    section, _ = deepSection(results, "part4_resnet_grid.csv", "part4_resnet_encodings.csv",
                             "Stage A — Frozen ResNet-18 features",
                             "python -m src.extract_deep -m resnet18 && python -m runs.run_part4_resnet")
    lines += section

    section, grid = deepSection(results, "part4_dino_grid.csv", "part4_dino_encodings.csv",
                                "Stage B — DINOv2 ViT-S/14",
                                "python -m src.extract_deep -m dinov2_vits14 --flip && python -m runs.run_part4_dino")
    lines += section
    if grid is not None:
        for csv in sorted(results.glob("part4_dino_confusion_*.csv")):
            cm = pd.read_csv(csv, index_col=0)
            lines += [f"Confusion matrix (test) — `{csv.stem.removeprefix('part4_dino_confusion_')}` "
                      "(rows: true, columns: predicted):", "", md(cm, floatfmt=".0f"), ""]

    lines += fewShotSection(results)
    lines += clipSection(results)

    lines += ["## Stage F — visual analysis and failure cases", ""]
    figures = sorted(results.glob("part4_examples_*.png"))
    if not figures:
        lines += pending("Stage F", "python -m runs.run_part4_visual")
    else:
        lines += ["Per category: highest-scoring positives (TP), highest-scoring background images (FP) and "
                  "lowest-scoring positives (FN) on the test set. ResNet-18 and DINOv2 maps are exact "
                  "decompositions of the SVM score over image positions (positive part shown); CLIP has no map. "
                  "Image ids and scores are in `part4_examples.csv`.", ""]
        lines += [f"![{png.stem}]({png.name})\n" for png in figures]

    lines += comparisonSection(results)
    return lines + questions(4)


def fewShotSection(results):
    lines = ["## Stage C — few-shot learning and flip augmentation", ""]
    fewshot = read(results, "part4_fewshot.csv")
    if fewshot is None:
        return lines + pending("Stages C and D", "python -m runs.run_part4_fewshot")

    sizes = sorted(fewshot.loc[~fewshot["all_pos"], "num_pos"].unique())
    label = fewshot["num_pos"].astype(str).where(~fewshot["all_pos"], "all")
    fewshot = fewshot.assign(positives=pd.Categorical(label, [str(s) for s in sizes] + ["all"], ordered=True))
    table = (fewshot.groupby(["category", "aug", "positives"], observed=True)["ap_val"]
             .apply(meanStd).unstack("positives"))
    per_seed = fewshot.groupby(["aug", "positives", "seed"], observed=True)["ap_val"].mean()
    mean = per_seed.groupby(level=["aug", "positives"], observed=True).apply(meanStd).unstack("positives")
    table = pd.concat([table, pd.concat({"mAP": mean}, names=["category"])])
    lines += ["Validation AP (mean ± std over random draws of the positives) against the number of positive "
              "training images. Negatives: every background training image. `flip` adds the flipped training "
              "images; validation images are never flipped. `all` is a single draw.", "", md(table), "",
              "![part4_fewshot](part4_fewshot.png)", ""]

    tta = read(results, "part4_tta.csv")
    if tta is not None:
        cols = ["no_tta", "score_avg", "feature_avg"]
        lines += ["Test-time flip as a separate condition (all positives, validation AP): `score_avg` averages "
                  "the SVM scores of an image and its flip, `feature_avg` scores the feature map of the mean raw "
                  "feature.", "", md(tta.groupby("train_aug")[cols].mean().rename_axis("train_aug (mAP)"), ".4f"), ""]

    lines += ["## Stage D — choice of the SVM C", ""]
    sweep, test = read(results, "part4_svm_c.csv"), read(results, "part4_svm_c_test.csv")
    if sweep is not None:
        per_seed = sweep.groupby(["regime", "C", "seed"], sort=False)[["ap_train", "ap_val"]].mean()
        table = per_seed.groupby(level=["regime", "C"], sort=False).agg(meanStd)
        lines += ["Train and validation mAP (mean ± std over draws) per C, with 5 and with all positives.", "",
                  md(table), "", "![part4_svm_c](part4_svm_c.png)", ""]
    if test is not None:
        lines += ["C selected per regime on validation mAP, evaluated once on the test set (test AP, mean over "
                  "draws):", "", md(withMAP(test.pivot_table(index=["regime", "C"], columns="category",
                                                             values="ap_test"))), ""]
    return lines


def clipSection(results):
    lines = ["## Stage E — CLIP zero-shot classification", ""]
    prompts = read(results, "part4_clip_prompts.csv")
    if prompts is None:
        return lines + pending("Stage E", "python -m runs.run_part4_clip")
    lines += ["Validation AP per prompt set (ranking each category's positives vs. background by cosine "
              "similarity to the class text embedding). Only the best set is evaluated on test.", "",
              md(prompts.set_index("prompt_set")), ""]

    test = read(results, "part4_clip_test.csv")
    if test is not None:
        lines += ["Test AP: zero-shot with the selected prompts vs. the Part 1 linear SVM on the same CLIP image "
                  "embeddings.", "", md(test.set_index("category")), ""]
    if (results / "part4_clip_confusion.csv").exists():
        cm = pd.read_csv(results / "part4_clip_confusion.csv", index_col=0)
        lines += ["Zero-shot classification of single-category test images (argmax over the 5 class prompts and "
                  "a background prompt; rows: true, columns: predicted):", "", md(cm, floatfmt=".0f"), ""]
    report = read(results, "part4_clip_classification.csv")
    if report is not None:
        lines += [md(report.set_index(report.columns[0]).rename_axis("class")), ""]
    costs = read(results, "part4_clip_costs.csv")
    if costs is not None:
        lines += ["Costs:", "", md(costs.set_index("encoding"), ".4f"), ""]
    return lines


def comparisonSection(results):
    """The required comparison table, from the Part 4 results."""
    lines = ["## Required comparison table", ""]
    supervision = {"bovw": "Task labels", "vlad": "Task labels", "fv": "Task labels",
                   "resnet18": "Pretrained (ImageNet) + task labels", "dinov2": "Self-supervised + task labels"}
    parts = []
    for grid_name, info_name in [("part4_resnet_grid.csv", "part4_resnet_encodings.csv"),
                                 ("part4_dino_grid.csv", "part4_dino_encodings.csv")]:
        grid, info = read(results, grid_name), read(results, info_name)
        if grid is None or info is None:
            continue
        selected = selectOnVal(grid, ["feature_map"])
        selected = selected[selected["encoding"].str.split("_").str[0].isin(supervision)]
        part = (selected.groupby("encoding", sort=False)
                .agg(mAP=("ap_test", "mean"), inference_ms_per_image=("infer_ms_per_image", "mean"))
                .join(info.set_index("encoding")[["dim", "extract_ms_per_image"]]))
        part.insert(0, "supervision", [supervision[e.split("_")[0]] for e in part.index])
        parts.append(part)

    test, costs = read(results, "part4_clip_test.csv"), read(results, "part4_clip_costs.csv")
    if test is not None and costs is not None:
        zero_shot = [c for c in test.columns if c.startswith("zero_shot")][0]
        parts.append(pd.DataFrame([{
            "supervision": "Image-text pretraining, no task labels",
            "mAP": test.set_index("category").loc["mAP", zero_shot],
            "inference_ms_per_image": costs["infer_us_per_image"].iloc[0] / 1000,
            "dim": costs["dim"].iloc[0], "extract_ms_per_image": costs["extract_ms_per_image"].iloc[0],
        }], index=[f"clip_vit_b_32 {zero_shot}"]))
    if not parts:
        return lines + pending("comparison table", "the Part 4 scripts")
    table = pd.concat(parts)
    table = table[~table.index.duplicated()]  # ResNet layer4 can appear in both grids
    lines += ["Test mAP over the 5 categories with the setting selected on validation. Extraction: SIFT + "
              "encoding (handcrafted, a few images) or one forward pass on CPU (deep). Inference: feature map + "
              "classifier (or text similarity) per image, excluding extraction.", "",
              md(table.rename_axis("method"), ".4f"), ""]
    return lines


REPORTS = {1: reportPart1, 2: reportPart2, 3: reportPart3, 4: reportPart4}


def parseArgs():
    parser = argparse.ArgumentParser(description="Build one Markdown report per part from results/.")
    parser.add_argument("--parts", nargs="+", type=int, default=list(REPORTS), choices=list(REPORTS))
    parser.add_argument("--results", default="results",
                        help="Directory with the run_part outputs; reports are written there too, "
                             "so the figure links resolve")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    results = Path(args.results)

    for part in args.parts:
        out_file = results / f"report_part{part}.md"
        out_file.write_text("\n".join(REPORTS[part](results)) + "\n", encoding="utf-8")
        print(f"Wrote {out_file}")
