"""Build one Markdown report per part from the CSVs and figures in results/.

Reads what the runs/ scripts wrote and writes results/report_part<N>.md with the
summary tables and the figures of that part (the answers to the assignment
questions are kept in answers/). Parts or
stages whose results are missing are listed as pending, with the command that
produces them. Nothing is recomputed: rerun a run_part script to update a table.

Usage:
    python -m runs.report
"""

import argparse
import platform
from importlib import metadata
from pathlib import Path

import pandas as pd


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


def note(text):
    """What the assignment asks in this section."""
    return [f"> {text}", ""]


def fmt(df, floatfmt=".3f"):
    """Numbers as text, missing cells as '–' (for tables with gaps)."""
    return df.apply(lambda col: col.map(lambda v: "–" if pd.isna(v) else
                                        v if isinstance(v, str) else f"{v:{floatfmt}}"))


def environment():
    """CPU, Python and package versions of the environment the runs use."""
    packages = ["numpy", "scikit-learn", "torch", "torchvision", "cyvlfeat"]
    versions = []
    for name in packages:
        try:
            versions.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            pass
    return f"{platform.processor()}, no GPU used; Python {platform.python_version()}, " + ", ".join(versions)


def pending(what, command):
    return [f"> **Pending:** {what} — run `{command}`.", ""]


def read(results, name):
    path = results / name
    return pd.read_csv(path) if path.exists() else None


def figure(results, name):
    """Markdown link to a figure in results/, or nothing if it was not produced."""
    return [f"![{Path(name).stem}]({name})", ""] if (results / name).exists() else []


def splitSizes():
    """Number of images per class (rows) and protocol subset (columns)."""
    counts = {}
    for path in Path("data/splits_protocol").glob("*_*.txt"):
        cls, subset = path.stem.rsplit("_", 1)
        with open(path, "r") as f:
            counts[cls, subset] = sum(1 for line in f if line.strip())
    return pd.Series(counts).unstack()[["train", "val", "test"]]


def labelOverlap(categories, subset="test"):
    """Images of each class (rows) that are also labelled with another class (columns)."""
    names = {}
    for cls in categories + ["background"]:
        with open(Path("data/splits_protocol") / f"{cls}_{subset}.txt", "r") as f:
            names[cls] = {line.strip() for line in f if line.strip()}
    return pd.DataFrame({other: {cls: len(names[cls] & names[other]) for cls in categories}
                         for other in categories + ["background"]})


################################################################################

def vocabularyTable(results, grid):
    """Per K, Hellinger map: dimension, validation mAP with / without tiles, quantization error and costs."""
    hellinger = grid[grid["feature_map"] == "sqrt_l2"]
    val = hellinger.pivot_table(index="K", columns="spatial", values="ap_val")
    tiles = hellinger[hellinger["spatial"] == "2x2 tiles"].groupby("K")
    table = pd.DataFrame({"dim (tiles)": tiles["dim"].first(), "val mAP (tiles)": val["2x2 tiles"],
                          "val mAP (global)": val["global"]})
    quant = read(results, "quantization.csv")
    if quant is not None:
        q = quant[quant["type"] == "bovw"].pivot_table(index="K", columns="subset", values="rel_qerr_sift")
        table["qerr (train)"], table["qerr (test)"] = q["train"], q["test"]
    costs = read(results, "part3_summary.csv")
    if costs is not None:
        c = costs[(costs["type"] == "bovw") & (costs["pca_dim"] == 0)].set_index("K")
        table["codebook_fit_s"] = c["codebook_fit_s"]
        table["extract_ms"] = c["sift_ms_per_image"] + c["encode_ms_per_image"]
    table["kb_per_image"] = table["dim (tiles)"] * 8 / 1024  # stored as int64
    table["svm_train_s"] = tiles["train_time_s"].mean()
    table["infer_us"] = 1000 * tiles["infer_ms_per_image"].mean()
    # A text column keeps the integers of K and dim from being printed as floats.
    table["dim (tiles)"] = table["dim (tiles)"].astype(str)
    return table


def reportPart1(results):
    lines = ["# Part 1 — Handcrafted features: BoVW and linear SVM", ""]
    grid = read(results, "part1_grid.csv")
    if grid is None:
        return lines + pending("Part 1", "python -m runs.run_part1")
    grid = labelSpatial(grid)
    categories = list(dict.fromkeys(grid["category"]))
    sizes = splitSizes().reindex(categories + ["background"])

    # One setting for the whole part: the best mean validation AP of the grid.
    K, spatial, fm = grid.groupby(["K", "spatial", "feature_map"])["ap_val"].mean().idxmax()
    chosen = grid[(grid["K"] == K) & (grid["spatial"] == spatial) & (grid["feature_map"] == fm)]
    chosen = chosen.set_index("category").reindex(categories)

    lines += ["Setup, the same for every stage:", "",
              "- Splits: `data/splits_protocol` (train / validation = 80 / 20 of the supplied train files, test = "
              "the supplied val files). Each category is a binary task: its images are positives, all background "
              "images of the same subset are negatives.",
              "- Features: dense SIFT (`vl_phow`, step 4); k-means vocabulary learned on training descriptors "
              "(supplied K = 512, plus K = 256 and 1024); histograms of the 2x2 tiles concatenated (dimension 4K). "
              "`global` = the 4 tile histograms summed into one, i.e. the histogram of the whole image "
              "(dimension K).",
              "- Feature maps: `none`, `l1`, `l2` (Stage E); `sqrt` = square root only, `sqrt_l1` = L1 after the "
              "root, `sqrt_l2` = L2 after the root = Hellinger map (Stage F).",
              "- Classifier: linear SVM (`LinearSVC`, hinge loss), C = 10 fixed, as the assignment suggests.",
              "- Selection: settings are compared on validation mAP (mean AP over the categories); the test set "
              "only reports the chosen setting. AP = scikit-learn `average_precision_score`, positive class = "
              "the category.",
              f"- Chosen setting (highest validation mAP of the whole grid): K = {K}, {spatial}, `{fm}`. It is "
              "used in Stages B–D and is the BoVW baseline of Parts 2 and 4.",
              "- Figures of Stages A–D: `python -m runs.run_part1_inspect`.",
              f"- Environment: {environment()}.", ""]

    lines += ["## Stage A — data, vocabulary size and spatial tiling", ""]
    lines += note("Inspect the images; learn the vocabulary on training data only; compare one global histogram "
                  "with spatial tiles; evaluate at least two vocabulary sizes besides the supplied K = 512.")
    lines += ["Images per class and subset:", "", md(sizes.rename_axis("class"), ".0f"), ""]
    lines += figure(results, "part1_samples.png")
    lines += ["Per vocabulary size, Hellinger map. `qerr`: mean squared distance of a SIFT descriptor to its "
              "nearest word / total SIFT variance (`runs.quantization`, 100 images per subset). Codebook fit and "
              "SIFT + encoding time per image (5 images, CPU) measured by `run_part3`. Storage: tiled histogram "
              "as stored (int64). SVM training and inference (per test image): mean over categories.", "",
              md(vocabularyTable(results, grid)), ""]

    lines += ["## Stage B — motorbike classifier", ""]
    lines += note("Motorbike images as positives, background as negatives, linear SVM on BoVW; rank the training "
                  "images by score; visualize the visual words associated with high classifier weights.")
    motorbike = grid[(grid["category"] == "motorbike") & (grid["K"] == K) & (grid["spatial"] == spatial)]
    lines += [f"Motorbike AP on the training images and on held-out images (K = {K}, {spatial}), per feature "
              "map:", "", md(motorbike.set_index("feature_map")[["ap_train", "ap_val", "ap_test"]]), ""]
    lines += figure(results, "part1_ranked_train.png")
    lines += ["Visual words with the largest contribution to the score (SVM weight, averaged over the tiles, × "
              "Hellinger histogram), for the top-ranked test image and the highest-ranked background test image. "
              "Each mosaic shows patches assigned to one word; dots of the same colour mark them in the image.", ""]
    lines += figure(results, "part1_words.png")

    lines += ["## Stage C — test set: ranking, precision–recall curve and AP", ""]
    lines += note("Apply the motorbike classifier to the test set, rank the images by the SVM score, plot the "
                  "precision–recall curve and report AP.")
    m = chosen.loc["motorbike"]
    lines += [f"Motorbike test AP = **{m['ap_test']:.3f}**; {m['test_top36_pos']} of the 36 top-ranked test images "
              f"are motorbikes (test set: {sizes.loc['motorbike', 'test']} motorbikes, "
              f"{sizes.loc['background', 'test']} background images).", "",
              "Classification at the SVM's own threshold (score > 0): "
              f"precision = {m['precision_test']:.3f}, recall = {m['recall_test']:.3f} "
              f"({m['tp_test'] + m['fp_test']} images predicted positive). Confusion matrix on test:", "",
              md(pd.DataFrame([[m["tp_test"], m["fn_test"]], [m["fp_test"], m["tn_test"]]],
                              index=pd.Index(["motorbike", "background"], name="true \\ predicted"),
                              columns=["motorbike", "background"]), ".0f"), "",
              "AP and the precision–recall curve depend only on the order of the scores; the bias b only sets "
              "where the score > 0 threshold falls on that curve.", ""]
    lines += figure(results, "part1_ranked_test.png")
    lines += ["Precision–recall curves on the training images (dashed) and on the test set (solid):", ""]
    lines += figure(results, "part1_pr_curves.png")

    lines += ["## Stage D — other categories", ""]
    lines += note("The same experiment for aeroplane, person, horse and car, same splits and protocol; per-class AP.")
    table = chosen[["ap_train", "ap_val", "ap_test", "test_top36_pos", "precision_test", "recall_test",
                    "tp_test", "fp_test", "fn_test", "tn_test"]].copy()
    table = table.rename(columns={"precision_test": "precision (s>0)", "recall_test": "recall (s>0)",
                                  "tp_test": "TP", "fp_test": "FP", "fn_test": "FN", "tn_test": "TN"})
    table.insert(0, "train positives", sizes["train"])
    table.insert(1, "test positives", sizes["test"])
    # AP of a random ranking = share of positives among the ranked test images.
    table.insert(2, "random AP", sizes["test"] / (sizes["test"] + sizes.loc["background", "test"]))
    # Share of the gap between a random and a perfect ranking that the classifier closes.
    table.insert(6, "normalized AP", (table["ap_test"] - table["random AP"]) / (1 - table["random AP"]))
    overlap = labelOverlap(categories)
    overlap = overlap.astype(str).where(overlap.values > 0, "–")
    for cls in categories:
        overlap.loc[cls, cls] = "·"
    lines += [f"K = {K}, {spatial}, `{fm}`. Test mAP = **{table['ap_test'].mean():.3f}**. `normalized AP` = "
              "(AP − random AP) / (1 − random AP): raw AP is not comparable across classes with different "
              "shares of positives.", "",
              md(table.rename_axis("category")), "",
              "Test images of each class (rows) also labelled with another class (columns). The negatives of "
              "every task are background images only, so images of the other classes are never scored:", "",
              md(overlap.rename_axis("class")), ""]

    lines += ["## Stage E — normalization and spatial information", ""]
    lines += note("No normalization vs. L1 vs. L2, selected on validation; 2x2 tiles vs. one global histogram.")
    val = grid.pivot_table(index=["K", "spatial"], columns="feature_map", values="ap_val")
    lines += ["Validation mAP:", "", md(val[["none", "l1", "l2"]]), ""]
    sub = grid[(grid["K"] == K) & (grid["feature_map"] == fm)]
    test = sub.pivot_table(index="category", columns="spatial", values="ap_test")
    val = sub.pivot_table(index="category", columns="spatial", values="ap_val")
    effect = pd.DataFrame({"test: 2x2 tiles": test["2x2 tiles"], "test: global": test["global"],
                           "test: tiles − global": test["2x2 tiles"] - test["global"],
                           "val: tiles − global": val["2x2 tiles"] - val["global"]}).reindex(categories)
    lines += [f"Effect of the tiles per category (K = {K}, `{fm}`):", "", md(effect.rename_axis("category")), ""]

    lines += ["## Stage F — Hellinger feature map", ""]
    lines += note("Square root of the histogram + normalization (Hellinger kernel as an explicit feature map), "
                  "compared with the other normalizations; motorbike retrained and evaluated on test.")
    val = grid.pivot_table(index=["K", "spatial"], columns="feature_map", values="ap_val")
    lines += ["Validation mAP (`l2` = linear kernel, for reference):", "",
              md(val[["l2", "sqrt", "sqrt_l1", "sqrt_l2"]]), ""]
    sub = grid[(grid["K"] == K) & (grid["spatial"] == spatial) & grid["feature_map"].isin(["l2", "sqrt_l2"])]
    test = withMAP(sub.pivot_table(index="feature_map", columns="category", values="ap_test")[categories])
    lines += [f"Test AP, linear (`l2`) vs. Hellinger (`sqrt_l2`), K = {K}, {spatial}:", "", md(test), ""]

    lines += ["## Stage G — number of training images", ""]
    lines += note("Train with 10, 25, 50 and 100 % of the training data, several seeds; validation AP against the "
                  "number of training examples; linear vs. Hellinger.")
    fractions = read(results, "part1_fractions.csv")
    if fractions is None:
        lines += pending("Stage G", "python -m runs.run_part1 --skip-grid")
    else:
        table = (fractions.groupby(["category", "feature_map", "fraction"])["ap_val"]
                 .apply(meanStd).unstack("fraction").reindex(categories, level="category"))
        per_seed = fractions.groupby(["feature_map", "fraction", "seed"])["ap_val"].mean()
        mean = per_seed.groupby(level=["feature_map", "fraction"]).apply(meanStd).unstack("fraction")
        table = pd.concat([table, pd.concat({"mAP": mean}, names=["category"])])
        positives = (fractions.groupby(["category", "fraction"])["num_pos"].mean().round().astype(int)
                     .unstack("fraction").reindex(categories))
        seeds = fractions["seed"].nunique()
        lines += [f"Validation AP, mean ± std over {seeds} seeds (stratified subsets; at 100 % every seed uses "
                  "the same images). `l2` = linear kernel, `sqrt_l2` = Hellinger; K = "
                  f"{fractions['K'].iloc[0]}, 2x2 tiles.", "", md(table), "",
                  "Positive training images per fraction (the negatives are subsampled in the same proportion):",
                  "", md(positives.rename_axis("category"), ".0f"), ""]
        lines += figure(results, "part1_fractions.png")
    return lines


def valTable(grid, selected, enc_type):
    """Validation mAP of every spatial layout and normalization, plus the selected one, per encoding."""
    sub = grid[grid["type"] == enc_type]
    table = sub.pivot_table(index="encoding", columns=["spatial", "feature_map"], values="ap_val", sort=False)
    table.columns = [f"{s}, {f}" for s, f in table.columns]
    chosen = selected[selected["type"] == enc_type].groupby("encoding", sort=False)[["spatial", "feature_map"]].first()
    table["selected"] = chosen["spatial"] + ", " + chosen["feature_map"]
    return table


def reportPart2(results):
    lines = ["# Part 2 — Advanced encodings: VLAD and Fisher Vectors", ""]
    grid = read(results, "part2_grid.csv")
    if grid is None:
        return lines + pending("Part 2", "python -m runs.run_part2")
    grid = labelSpatial(grid)
    selected = selectOnVal(grid, ["spatial", "feature_map"])
    test = withMAP(selected.pivot_table(index="encoding", columns="category", values="ap_test", sort=False))
    bovw, vlad, fv = ([e for e in test.index if e.startswith(t)] for t in ["bovw", "vlad", "fv"])
    test = test.loc[bovw + vlad + fv]
    info = read(results, "part2_encodings.csv")
    info = info.set_index("encoding") if info is not None else pd.DataFrame(index=test.index)

    lines += ["Setup, the same for every stage:", "",
              "- Same splits, categories and linear SVM (C = 10) as Part 1.",
              "- Local descriptors: the same dense SIFT, reduced to 80 dimensions by PCA (`C80`). VLAD: k-means "
              "with K words, residuals to the assigned word. FV: GMM with K diagonal components, soft assignment, "
              "first- and second-order statistics. PCA and codebooks are fitted on training descriptors only.",
              "- 2x2 tiles as in Part 1. Dimensions: VLAD 4·K·80; FV 4·(2·K·80 + K), i.e. the 2K × d of the "
              "assignment plus the K mixture-weight terms of the implementation, per tile. `global` = the 4 tile "
              "vectors summed. VLAD tiles are stored L2-normalized and FV tiles are averages over the tile's "
              "descriptors, so this is an approximation of a VLAD / FV computed on the whole image, not the same "
              "vector.",
              "- Normalizations: `none`, `l2`, `ssr_l2` = signed square root + L2 (power normalization), "
              "`intra_l2` = L2 of each word's residual block, then global L2 (VLAD only), `ssr_intra_l2` = both.",
              "- Selection: per encoding, the layout and normalization with the highest validation mAP; the test "
              "set only reports that setting.",
              f"- BoVW baseline: {', '.join(bovw)}, setting selected in the same way (the Part 1 choice).", ""]

    lines += ["## Stage H — VLAD", ""]
    lines += note("Replace BoVW with VLAD (same descriptors and splits), normalization selected on validation, "
                  "repeat the classification and retrieval experiments, note the gain over BoVW.")
    lines += ["Validation mAP per layout and normalization:", "", md(fmt(valTable(grid, selected, "vlad"))), ""]
    table = test.loc[bovw + vlad].copy()
    table["gain vs. BoVW"] = table["mAP"] - test.loc[bovw[0], "mAP"]
    table = table.join(info[["dim", "kb_per_image"]])
    lines += ["Test AP of the selected setting:", "", md(table), ""]

    lines += ["## Stage I — Fisher Vector", ""]
    lines += note("GMM fitted on training descriptors only; Fisher Vectors with the same classifier and splits; "
                  "compare FV, VLAD and BoVW: AP / mAP, feature dimension, extraction time and storage.")
    lines += ["Validation mAP per layout and normalization:", "", md(fmt(valTable(grid, selected, "fv"))), ""]
    lines += ["Test AP of the selected setting:", "", md(test.loc[fv]), ""]
    costs = selected.groupby("encoding", sort=False).agg(
        svm_train_s=("train_time_s", "mean"), infer_us=("infer_ms_per_image", lambda t: 1000 * t.mean()))
    table = test[["mAP"]].join(info).join(costs)
    lines += ["Comparison of the three encodings (selected settings). Codebook fit = PCA + k-means / GMM, once; "
              "SIFT and encoding time per image on 5 test images (CPU); storage as stored on disk; SVM training "
              "and inference (per test image) averaged over the categories:", "", md(table), ""]
    lines += ["Retrieval: test images ranked by the SVM score of the selected setting, precision-recall curves "
              "(dotted line = positive rate, i.e. a random ranking):", ""]
    lines += figure(results, "part2_pr_curves.png")
    return lines


def reportPart3(results):
    lines = ["# Part 3 — Large visual vocabularies with PCA vs. smaller vocabularies", ""]
    summary = read(results, "part3_summary.csv")
    if summary is None:
        return lines + pending("Part 3", "python -m runs.run_part3")
    summary["mAP"] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary["mAP_test"], summary["mAP_test_std"])]
    summary["PCA d"] = summary["pca_dim"].map(lambda d: str(d) if d else "none")
    summary["extract_ms"] = summary["sift_ms_per_image"] + summary["encode_ms_per_image"]
    no_pca = summary[summary["pca_dim"] == 0].set_index("encoding")

    lines += ["Setup, the same for every table:", "",
              "- BoVW: one global histogram per image (the 2x2 tiles of raw counts summed, dimension K), "
              "Hellinger map. VLAD / FV: 2x2 tiles, signed square root + L2. Dimensions: VLAD 4·K·80; FV "
              "4·(2·K·80 + K), i.e. the 2K × d of the assignment plus the K mixture-weight terms of the "
              "implementation, per tile.",
              "- VLAD normalization differs from Part 2 on purpose: on validation, Part 2 picked `ssr_l2` for "
              "K = 16 and `ssr_intra_l2` for K = 32 / 64, so following it would change the normalization along "
              "K. Part 3 keeps one normalization for every K, and the same one for VLAD and FV, so that only K "
              "and PCA vary.",
              "- PCA is fitted on the training features only and its output is L2-normalized; "
              "d < original dimension and d ≤ number of training images − 1.",
              "- Linear SVM per category (positives vs. background), C = 10, the Part 1 splits.",
              "- Nothing is selected in Part 3: every condition is reported on the test set, as the "
              "assignment asks; validation mAP is in `part3_summary.csv`.",
              "- 3 seeds: each trains on a different stratified 90 % of the training set (PCA refitted); "
              "validation and test are complete. mAP = test AP averaged over the 5 categories, ± = std over seeds. "
              "Accuracy and macro-F1 use the SVM threshold 0.",
              "- Codebooks (PCA80 + k-means / GMM, fitted once per K) are learned without labels from "
              "descriptors of the protocol training images only (733 images, 400 descriptors each, the same "
              "sample for every encoder); validation and test images are never seen.",
              "- Times, all on CPU: SIFT + encoding per image on 5 images, codebook fit once, PCA and SVM fit per "
              "category, inference = PCA projection + SVM per test image.",
              "- Storage (`kb_per_image`) = final dimension × 4 bytes (float32) for every representation, so "
              "that sizes are comparable; Part 2 reports the arrays as stored (BoVW int64, FV float64).", ""]

    lines += ["## Stage A — conditions A–D (BoVW)", ""]
    lines += note("The assignment's table. A: small codebook (K = 256), no PCA. B: large codebook (K = 1024), "
                  "no PCA. C: the large codebook + PCA to d. D: a codebook with K = d (C's final dimension), no PCA.")
    conditions = read(results, "part3_conditions.csv")
    if conditions is not None:
        table = conditions[["condition", "encoding", "pca_dim"]].merge(summary, on=["encoding", "pca_dim"])
        cols = ["condition", "encoding", "PCA d", "dim", "mAP", "acc_test", "f1_test", "codebook_fit_s",
                "extract_ms", "pca_fit_s", "svm_train_s", "infer_us", "kb_per_image"]
        lines += [md(table[cols].rename(columns={"acc_test": "accuracy", "f1_test": "macro_F1"})
                     .set_index("condition")), ""]

    lines += ["## Q1 — larger codebook, before PCA", ""]
    lines += note("Test AP per category without PCA, as K grows.")
    per_class = read(results, "part3_per_class.csv")
    if per_class is not None:
        lines += [md(withMAP(per_class.set_index("encoding"))), ""]

    lines += ["## Q2 — how much PCA keeps", ""]
    lines += note("Test mAP for every final dimension d (`none` = no PCA). A row is one codebook; reading it "
                  "from right to left shows how far PCA can go before mAP drops.")
    dims = [str(d) for d in sorted(summary.loc[summary["pca_dim"] > 0, "pca_dim"].unique())] + ["none"]
    table = summary.pivot(index="encoding", columns="PCA d", values="mAP")
    lines += [md(fmt(table.reindex(index=summary["encoding"].unique(), columns=dims))), "",
              "![part3_map_vs_dim](part3_map_vs_dim.png)", ""]

    lines += ["## Q3 — same final dimension: large codebook + PCA (C) vs. small codebook (D)", ""]
    lines += note("BoVW test mAP. Each row is one final dimension; the first column is a codebook with K = d "
                  "used directly, the others are larger codebooks reduced with PCA to d.")
    bovw = summary[summary["type"] == "bovw"]
    direct = bovw[bovw["pca_dim"] == 0].set_index("dim")["mAP_test"].rename("D: K = d, no PCA")
    reduced = bovw[bovw["pca_dim"] > 0].pivot(index="pca_dim", columns="K", values="mAP_test")
    reduced.columns = [f"C: K = {k} + PCA" for k in reduced.columns]
    table = reduced.join(direct)[[direct.name, *reduced.columns]].rename_axis("final dim")
    lines += [md(fmt(table)), ""]

    lines += ["## Q4 — accuracy vs. memory and time", ""]
    lines += note("Every codebook without PCA and with PCA to 128. The large representation is paid in full "
                  "(codebook fit, SIFT + encoding) before PCA; PCA only lowers storage, SVM training and inference.")
    cols = ["encoding", "PCA d", "dim", "kb_per_image", "mAP", "codebook_fit_s", "extract_ms", "pca_fit_s",
            "svm_train_s", "infer_us"]
    lines += [md(summary.loc[summary["pca_dim"].isin([0, 128]), cols].set_index("encoding")), ""]

    lines += ["## Q5 — BoVW vs. VLAD / Fisher Vector", ""]
    lines += note("Share of dimensions removed and of mAP kept when each representation is reduced to the "
                  "same d = 128. BoVW, VLAD and FV dimensions are not interchangeable: VLAD / FV grow with "
                  "K × 80, so the same d removes far more of them.")
    at = summary[summary["pca_dim"] == 128].set_index("encoding")
    table = pd.DataFrame({"orig_dim": no_pca["orig_dim"], "mAP (no PCA)": no_pca["mAP_test"],
                          "mAP (d = 128)": at["mAP_test"], "removed_%": at["removed_%"]})
    table = table.reindex(no_pca.index).dropna()
    table["mAP kept_%"] = 100 * table["mAP (d = 128)"] / table["mAP (no PCA)"]
    lines += [md(table.rename_axis("encoding")), ""]
    return lines


def deepSection(results, name, title, asked, command):
    """Test AP per encoding, with validation mAP, dimension and costs."""
    lines = [f"## {title}", ""] + note(asked)
    grid, info = read(results, f"{name}_grid.csv"), read(results, f"{name}_encodings.csv")
    if grid is None or info is None:
        return lines + pending(title, command)

    test = withMAP(grid.pivot_table(index="encoding", columns="category", values="ap_test", sort=False))
    per_encoding = grid.groupby("encoding", sort=False)
    table = (test.join(per_encoding["ap_val"].mean().rename("mAP_val"))
             .join(info.set_index("encoding")[["dim", "extract_ms_per_image"]])
             .join((1000 * per_encoding["infer_ms_per_image"].mean()).rename("infer_us_per_image")))
    lines += ["Test AP per category and mAP, validation mAP, feature dimension, extraction time and inference "
              "cost per image:", "", md(table), ""]
    return lines


def reportPart4(results):
    lines = ["# Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP", ""]
    lines += ["Protocol, the same for every stage:", "",
              "- Splits: every method reads the same `data/splits_protocol` lists (train / validation = 80 / 20 "
              "of the supplied train files, test = the supplied val files): the same images, in the same order, "
              "with the same labels; no image is in two subsets.",
              "- Selection: deep features are L2-normalized and the SVM uses C = 10, both fixed in advance; the "
              "choices of Stages D and E (C, prompt set) are made on validation AP and the test set only reports "
              "the chosen setting.",
              "- Costs: feature dimension and extraction time per image are stored by `src.extract_deep` in "
              "`meta.json` (forward pass on CPU over all 4946 images, original + flip, including model loading, "
              "JPEG decoding and preprocessing). Inference = feature map + linear SVM per test image, features "
              "already extracted.",
              "- Models (frozen, official preprocessing, center crop 224): ResNet-18 and ViT-B/32 with "
              "torchvision `IMAGENET1K_V1` weights; DINOv2 ViT-S/14 from torch.hub `facebookresearch/dinov2`; "
              "CLIP ViT-B/32 from `openai/CLIP`.",
              f"- Environment: {environment()}.", ""]

    lines += deepSection(results, "part4_resnet", "Stage A — frozen CNN features (ResNet-18)",
                         "Global average pooling of each of the 3 last ResNet-18 stages, L2, linear SVM on the "
                         "frozen features, compared with the BoVW / VLAD / FV settings selected in Part 2 (their "
                         "costs as measured in Part 2).",
                         "python -m runs.run_part4_resnet")

    lines += deepSection(results, "part4_dino", "Stage B — DINOv2 ViT-S/14",
                         "Frozen DINOv2 global embedding (CLS token) + linear SVM: per-class AP, mAP, confusion "
                         "matrix, dimension, extraction and inference cost. Optional extension: the mean of the "
                         "patch tokens (`patchmean`). The ViT-B/32 class token (the \"CNN/ViT\" row of the final "
                         "table) and CLIP are shown for reference.",
                         "python -m runs.run_part4_dino")
    for csv in sorted(results.glob("part4_dino_confusion_dinov2_*.csv")):
        cm = pd.read_csv(csv, index_col=0)
        lines += [f"Test confusion matrix — `{csv.stem.removeprefix('part4_dino_confusion_')}` (rows: true, "
                  "columns: predicted). Each image goes to the category whose SVM scores highest, or to "
                  "background when every score is negative; images with several categories are left out.", "",
                  md(cm, floatfmt=".0f"), ""]

    lines += fewShotSection(results)
    lines += clipSection(results)
    lines += visualSection(results)
    lines += comparisonSection(results)
    lines += limitationsSection()
    return lines


def fewShotSection(results):
    lines = ["## Stage C — few-shot learning and flip augmentation", ""]
    lines += note("One representation (DINOv2 CLS, L2) trained with 1, 5, 10, 20, 50 and all positives; "
                  "no augmentation vs. horizontal flip of the training images; validation AP, mean ± std over "
                  "5 random draws of the positives.")
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
    lines += ["Validation AP against the number of positive training images. Negatives: every background "
              "training image. `flip` adds the flipped training images; validation images are never flipped. "
              "`all` is a single draw.", "", md(table), "", "![part4_fewshot](part4_fewshot.png)", ""]

    tta = read(results, "part4_tta.csv")
    if tta is not None:
        cols = ["no_tta", "score_avg", "feature_avg"]
        lines += ["Test-time flip, a separate condition (all positives, validation mAP): `score_avg` averages "
                  "the SVM scores of an image and its flip, `feature_avg` scores the mean of the two features.", "",
                  md(tta.groupby("train_aug")[cols].mean().rename_axis("train_aug"), ".4f"), ""]

    lines += ["## Stage D — choice of the SVM C", ""]
    lines += note("C in 0.1 … 1000, train and validation AP to diagnose overfitting, C selected on validation "
                  "and then evaluated once on the test set. Done with 5 positives and with all positives.")
    sweep, test = read(results, "part4_svm_c.csv"), read(results, "part4_svm_c_test.csv")
    if sweep is not None:
        per_seed = sweep.groupby(["regime", "C", "seed"], sort=False)[["ap_train", "ap_val"]].mean()
        table = per_seed.groupby(level=["regime", "C"], sort=False).agg(meanStd)
        lines += ["Train and validation mAP (mean ± std over draws) per C:", "",
                  md(table), "", "![part4_svm_c](part4_svm_c.png)", ""]
    if test is not None:
        lines += ["Selected C per regime (highest validation mAP), test AP (mean over draws):", "",
                  md(withMAP(test.pivot_table(index=["regime", "C"], columns="category", values="ap_test"))), ""]
    return lines


def clipSection(results):
    lines = ["## Stage E — CLIP zero-shot classification", ""]
    lines += note("CLIP ViT-B/32 with its own preprocessing; class prompts encoded by the text encoder; "
                  "image–text cosine similarity. Retrieval: rank by similarity to the class prompt → AP / mAP. "
                  "Classification: most similar prompt → confusion matrix and per-class metrics. One prompt vs. "
                  "prompt sets, chosen on validation only.")
    prompts = read(results, "part4_clip_prompts.csv")
    if prompts is None:
        return lines + pending("Stage E", "python -m runs.run_part4_clip")
    lines += ["Validation AP per prompt set. Only the best set is evaluated on test:", "",
              md(prompts.set_index("prompt_set")), ""]

    test = read(results, "part4_clip_test.csv")
    if test is not None:
        lines += ["Test AP: zero-shot with the selected prompts vs. the linear SVM (L2, C = 10) on the same CLIP "
                  "image embeddings:", "", md(test.set_index("category")), ""]
    if (results / "part4_clip_confusion.csv").exists():
        cm = pd.read_csv(results / "part4_clip_confusion.csv", index_col=0)
        lines += ["Zero-shot classification of the single-category test images (most similar of the 5 class "
                  "prompts and a background prompt; rows: true, columns: predicted):", "",
                  md(cm, floatfmt=".0f"), ""]
    report = read(results, "part4_clip_classification.csv")
    if report is not None:
        lines += [md(report.set_index(report.columns[0]).rename_axis("class")), ""]
    costs = read(results, "part4_clip_costs.csv")
    if costs is not None:
        lines += ["Costs:", "", md(costs.set_index("encoding"), ".4f"), ""]
    return lines


def visualSection(results):
    lines = ["## Stage F — visual analysis and failure cases", ""]
    lines += note("Correct and incorrect test examples of the CNN, DINOv2 and CLIP experiments, with the image "
                  "regions that support the class where the method allows it.")
    figures = sorted(results.glob("part4_examples_*.png"))
    if not figures:
        return lines + pending("Stage F", "python -m runs.run_part4_visual")
    lines += ["Per category: correctly detected positives (TP, highest scores), background images wrongly "
              "detected (FP, highest scores) and missed positives (FN, lowest scores). Only real errors: SVM "
              "score above / below 0; for CLIP, the 6-way prompt argmax of Stage E. A category with fewer "
              "errors shows fewer images.", "",
              "Models: ResNet-18 layer4 and DINOv2 `patchmean`, both L2 + SVM with C = 10 (the settings selected "
              "on validation). Their maps split the SVM score over image positions (positive part shown); the "
              "DINOv2 CLS token has no such split, so `patchmean` is used. CLIP: examples only. Image ids and "
              "scores are in `part4_examples.csv`.", ""]
    lines += [f"![{png.stem}]({png.name})\n" for png in figures]
    return lines


# Required comparison table: method, encoding prefix, supervision.
METHODS = [("BoVW + SVM", "bovw", "Task labels"),
           ("VLAD + SVM", "vlad", "Task labels"),
           ("Fisher Vector + SVM", "fv", "Task labels"),
           ("CNN (ResNet-18) + SVM", "resnet18", "Pretrained (ImageNet) + task labels"),
           ("ViT-B/32 + SVM", "vit", "Pretrained (ImageNet) + task labels"),
           ("DINOv2 + SVM", "dinov2", "Self-supervised + task labels")]


def finalSelection(results):
    """Per SVM method of the comparison table, the encoding with the best validation mAP."""
    selected = []
    for name in ["part4_resnet", "part4_dino"]:
        grid, info = read(results, f"{name}_grid.csv"), read(results, f"{name}_encodings.csv")
        if grid is not None and info is not None:
            per_encoding = (selectOnVal(grid, ["feature_map"]).groupby("encoding", sort=False)
                            .agg(feature_map=("feature_map", "first"), mAP_val=("ap_val", "mean"),
                                 mAP_test=("ap_test", "mean"), infer_ms=("infer_ms_per_image", "mean")))
            selected.append(per_encoding.join(info.set_index("encoding")[["dim", "extract_ms_per_image"]]))
    if not selected:
        return None
    selected = pd.concat(selected)

    rows = []
    for method, prefix, supervision in METHODS:
        family = selected[selected.index.str.startswith(prefix + "_")]
        if not family.empty:
            encoding = family["mAP_val"].idxmax()
            rows.append({"method": method, "encoding": encoding, "supervision": supervision,
                         **family.loc[encoding]})
    return pd.DataFrame(rows).set_index("method")


def comparisonSection(results):
    lines = ["## Required comparison table", ""]
    lines += note("One row per method. Within a method, the representation (layer or token) with the highest "
                  "validation mAP. Every SVM uses C = 10, fixed in advance as in Part 1 (Stage D studies C "
                  "separately). Extraction: deep models = forward pass on CPU (see the protocol above); "
                  "handcrafted = dense SIFT + encoding on CPU, timed on 5 images in Part 2. Inference: feature "
                  "map + classifier (or text similarity) per image, features already extracted.")
    selection = finalSelection(results)
    if selection is None:
        return lines + pending("comparison table", "the Part 4 scripts")
    table = pd.DataFrame({
        "representation": selection["encoding"] + " (" + selection["feature_map"] + ")",
        "supervision": selection["supervision"], "C": 10, "test mAP": selection["mAP_test"],
        "dim": selection["dim"].astype(int), "extract_ms_per_image": selection["extract_ms_per_image"],
        "infer_us_per_image": 1000 * selection["infer_ms"]})

    test, costs = read(results, "part4_clip_test.csv"), read(results, "part4_clip_costs.csv")
    if test is not None and costs is not None:
        zero_shot = [c for c in test.columns if c.startswith("zero_shot")][0]
        table.loc["CLIP zero-shot"] = {
            "representation": f"clip_vit_b_32, prompts {zero_shot[10:]}",
            "supervision": "Image–text pretraining, no task labels", "C": "–",
            "test mAP": test.set_index("category").loc["mAP", zero_shot], "dim": int(costs["dim"].iloc[0]),
            "extract_ms_per_image": costs["extract_ms_per_image"].iloc[0],
            "infer_us_per_image": costs["infer_us_per_image"].iloc[0]}
    lines += [md(table.rename_axis("method")), ""]
    return lines


def limitationsSection():
    return ["## Limitations and deviations from the protocol", "",
            "- AP compares each category only with background images, so it does not measure confusions "
            "between categories; the confusion matrices (Stages B and E) do.",
            "- Images with several categories (e.g. horse + person) are left out of the confusion matrices, "
            "which need one true class per image.",
            "- CLIP zero-shot classification uses 6 prompts (5 categories + background) because the test set "
            "contains background images; the assignment mentions the 5 categories. The 5-way accuracy on "
            "object images is printed by `run_part4_clip`.",
            "- Extraction times of deep and handcrafted methods are measured differently (all images incl. "
            "model loading vs. 5 images) and are approximate.",
            "- All runs on CPU, one machine; see the protocol for hardware and versions.", ""]


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

    # The setup block of each part (between its title and the first section) goes to one
    # shared report_setup.md, so the per-part reports start directly with the results.
    setups = {}
    for part in args.parts:
        lines = REPORTS[part](results)
        first = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
        title, setup = lines[0], lines[1:first]
        if any(line.strip() for line in setup):
            setups[part] = [title.replace("# ", "## ", 1), *setup]
        out_file = results / f"report_part{part}.md"
        out_file.write_text("\n".join([title, "", *lines[first:]]) + "\n", encoding="utf-8")
        print(f"Wrote {out_file}")

    # Parts not rebuilt in this run keep their previous setup block.
    setup_file = results / "report_setup.md"
    if setup_file.exists():
        previous = setup_file.read_text(encoding="utf-8").split("\n## ")[1:]
        for block in previous:
            part = int(block.split()[1])
            setups.setdefault(part, ["## " + block.rstrip("\n")])
    setup_lines = ["# Setup and protocol of every part", ""]
    for part in sorted(setups):
        setup_lines += [*setups[part], ""]
    setup_file.write_text("\n".join(setup_lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"Wrote {setup_file}")
