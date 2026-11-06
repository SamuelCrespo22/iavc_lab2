"""Part 5: open-ended image retrieval challenge.

Stage A, setup (part5_corpus.csv, part5_queries.csv):

  corpus    every image of the protocol test split (the five categories and
            background, multi-category images once). An image is relevant when
            it is in <category>_test.txt; the other categories and the
            background scenes (cows, dogs, sheep...) are the irrelevant ones.
  queries   --num-queries images of the category from the protocol val split,
            half with and half without a person (chosen from labels only, so
            no method picks its own queries), drawn with --seed. Val and test
            come from different supplied VOC splits and share no image;
            candidates whose DINOv2 cosine similarity to a corpus image is above
            --dup-threshold are still dropped as near duplicates. Near-duplicate
            pairs inside the corpus go to part5_corpus_duplicates.csv.

Stage B, methods (METHODS): query and corpus images use the same cached
encoding and feature map, and the corpus is ranked by cosine similarity
(histogram intersection where stated). No classifier is trained and nothing is
chosen on the corpus: the main settings are the ones selected on validation in
Parts 2 and 4, and every ablation is reported in full. CLIP text-to-image ranks
the corpus by similarity to a prompt embedding (main entry: the prompt set
selected on validation in Part 4).

Stage C: AP, P@K and R@K per query (every relevant corpus image is known);
mAP and mean P@K / R@K over the queries. Costs: dimension, corpus storage as
float32, extraction time per image (meta.json of the deep encodings, including
model loading; Part 2 / 3 SIFT + encoding timings for the handcrafted ones) and
query time (similarity to the whole corpus + full sort).

Experiments (results in --out):
  methods   main comparison and costs               part5_per_query.csv, part5_methods.csv
  vocab     BoVW K x tiling x similarity            part5_vocab.csv
  pca       PCA (+ whitening) of BoVW / VLAD / FV   part5_pca.csv
  prompts   CLIP text queries of increasing detail  part5_prompts.csv
  corpus    fewer or other irrelevant images        part5_corpus_size.csv

Usage (from the directory containing README.md):
    python -m runs.run_part5
    python -m runs.run_part5 --category motorbike --skip corpus
"""

import os

# torch must be imported before numpy in this env (see src.extract_deep).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import clip

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

from runs.run_part1 import CATEGORIES, l1, l2, removeSpatialInformation
from runs.run_part2 import ENCODINGS_DIR, parseEncoding, prepare
from runs.run_part4_clip import NAMES, PROMPT_SETS, SYNONYMS, TEMPLATES, loadClip, withArticle
from runs.run_part4_resnet import BASELINES
from runs.run_part4_visual import selectedPromptSet

PROTOCOL_DIR = Path("data/splits_protocol")
IMG_DIR = Path("data/images")
DUP_ENCODING = "dinov2_vits14_cls"
CLIP_ENCODING = "clip_vit_b_32"
EXPERIMENTS = ["vocab", "pca", "prompts", "corpus"]


def deep(x):
    return l2(x.astype(np.float64))


def handcrafted(encoding, spatial, feature_map):
    """The Part 2 feature map of an encoding; 'l1' is the BoVW histogram for intersection."""
    enc_type, _, pca_dim = parseEncoding(encoding)
    if feature_map == "l1":
        return lambda x: l1(x.astype(np.float64) if spatial else removeSpatialInformation(x.astype(np.float64)))
    return lambda x: prepare(x, spatial, enc_type, feature_map, pca_dim or 128)


def handcraftedName(encoding):
    enc_type, K, _ = parseEncoding(encoding)
    return f"{ {'bovw': 'BoVW', 'vlad': 'VLAD', 'fv': 'FV'}[enc_type]} K{K}"


# method -> (encoding, feature map, similarity). Handcrafted: the Part 2 settings
# (BASELINES); deep: L2-normalized embeddings, so the dot product is the cosine.
METHODS = {handcraftedName(e): (e, handcrafted(e, spatial, fm), "cosine") for e, spatial, fm in BASELINES}
METHODS.update({
    "ResNet-18": ("resnet18_layer4", deep, "cosine"),
    "ViT-B/32": ("vit_b_32", deep, "cosine"),
    "DINOv2 CLS": ("dinov2_vits14_cls", deep, "cosine"),
    "DINOv2 patch mean": ("dinov2_vits14_patchmean", deep, "cosine"),
    "CLIP image": (CLIP_ENCODING, deep, "cosine"),
})

# BoVW variants of the vocabulary experiment: similarity -> (feature map, measure).
BOVW_SIMILARITIES = {
    "cosine (L2)": ("l2", "cosine"),
    "cosine (Hellinger)": ("sqrt_l2", "cosine"),
    "intersection (L1)": ("l1", "intersection"),
}
VOCAB_KS = [32, 64, 128, 256, 512, 1024]

# PCA experiment: the main handcrafted settings and a large global BoVW, to
# compare "large K + PCA to d" with "K = d" from the vocabulary experiment.
PCA_CONFIGS = BASELINES + [("bovw_K1024_C0", False, "sqrt_l2")]
PCA_DIMS = [16, 32, 64, 128, 256, 512]

# Text queries beyond the Part 4 prompt sets: (description, scene, over-specific).
# Written before looking at any ranking.
DESCRIPTIONS = {
    "aeroplane": ("an aeroplane, an aircraft with two wings, a tail and jet engines.",
                  "a passenger airplane parked at an airport next to the terminal.",
                  "a red propeller biplane doing aerobatics at an air show."),
    "motorbike": ("a motorbike, a two-wheeled motor vehicle with a seat, handlebars and an engine.",
                  "a person riding a motorcycle on a road.",
                  "a red racing motorcycle leaning into a corner on a race track."),
    "person": ("a person, a human being with a head, arms and legs.",
               "a group of people standing together in a room.",
               "a woman with long hair smiling at the camera in a close-up portrait."),
    "horse": ("a horse, a large four-legged animal with a mane, hooves and a long tail.",
              "a person riding a horse in a field.",
              "a white horse jumping over a fence at a show-jumping competition."),
    "car": ("a car, a four-wheeled motor vehicle with doors, windows and headlights.",
            "cars parked along a city street.",
            "a red sports car racing on a track."),
}


def promptQueries(category):
    """``{level: prompt or prompts}``, from the bare class name to an over-specific scene.

    Several prompts are averaged into one query (prompt ensembling, as in Part 4).
    """
    noun = NAMES[category]
    description, scene, specific = DESCRIPTIONS[category]
    return {
        "name": noun,
        "photo": f"a photo of {withArticle(noun)}.",
        "templates": [t.format(a=withArticle(noun), n=noun) for t in TEMPLATES],
        "synonyms": [f"a photo of {withArticle(s)}." for s in SYNONYMS[category]],
        "description": description,
        "scene": scene,
        "over-specific": specific,
    }


def encodeText(model, prompts):
    """Unit-norm CLIP text embedding; several prompts are averaged as in run_part4_clip."""
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    with torch.inference_mode():
        e = model.encode_text(clip.tokenize(prompts)).float()
    e = (e / e.norm(dim=-1, keepdim=True)).mean(dim=0)
    return (e / e.norm()).numpy().astype(np.float64)


################################################################################
# Data

def readList(category, subset):
    with open(PROTOCOL_DIR / f"{category}_{subset}.txt", "r") as f:
        return [l.strip() for l in f if l.strip()]


def imageLabels(subset):
    """``{image: '+'-joined categories}`` of a protocol subset ('background' for background images)."""
    labels = {}
    for category in CATEGORIES + ["background"]:
        for name in readList(category, subset):
            labels.setdefault(name, []).append(category)
    return {name: "+".join(sorted(cats)) for name, cats in labels.items()}


def lookup(encoding, names, source):
    """Rows of ``names``, in that order, from the ``*_<source>.npz`` files of an encoding.

    source: 'val' for the corpus (protocol test) and 'train' for the queries and the
    PCA images (protocol val / train), as in src.splits.loadProtocolSubset.
    """
    wanted, rows = set(names), {}
    for path in sorted((ENCODINGS_DIR / encoding).glob(f"*_{source}.npz")):
        with np.load(path) as data:
            file_names = data["file_names"]
            idx = [i for i, n in enumerate(file_names) if n in wanted and n not in rows]
            if idx:
                rows.update(zip(file_names[idx], data["histograms"][idx].astype(np.float32)))
    missing = wanted - rows.keys()
    assert not missing, f"{encoding}: {len(missing)} images missing in {ENCODINGS_DIR / encoding}"
    return np.stack([rows[n] for n in names])


@lru_cache(maxsize=6)
def rawVectors(encoding, queries, corpus):
    """Raw (query, corpus) encodings; names are tuples so the result can be cached."""
    return lookup(encoding, list(queries), "train"), lookup(encoding, list(corpus), "val")


def vectors(setup, encoding, transform):
    Q, X = rawVectors(encoding, tuple(setup["queries"]["name"]), tuple(setup["corpus"]["name"]))
    return transform(Q), transform(X)


def buildSetup(category, num_queries, seed, dup_threshold):
    corpus_labels = imageLabels("test")
    corpus = pd.DataFrame({"name": sorted(corpus_labels)})
    corpus["labels"] = corpus["name"].map(corpus_labels)
    corpus["relevant"] = corpus["labels"].str.split("+").apply(lambda cats: category in cats)

    val_labels = imageLabels("val")
    candidates = pd.DataFrame({"name": readList(category, "val")})
    candidates["labels"] = candidates["name"].map(val_labels)
    candidates["group"] = np.where(candidates["labels"].str.contains("person"), "with person", "without person")
    sims = (deep(lookup(DUP_ENCODING, candidates["name"].tolist(), "train"))
            @ deep(lookup(DUP_ENCODING, corpus["name"].tolist(), "val")).T)
    candidates["max_sim_corpus"] = sims.max(axis=1)
    candidates["nearest_corpus"] = corpus["name"].to_numpy()[sims.argmax(axis=1)]
    num_dup = int((candidates["max_sim_corpus"] > dup_threshold).sum())
    candidates = candidates[candidates["max_sim_corpus"] <= dup_threshold]

    # The same number of queries per group (the first groups take the remainder).
    rng = np.random.default_rng(seed)
    groups = [df for _, df in candidates.groupby("group")]
    picked = [df.iloc[rng.permutation(len(df))[:num_queries // len(groups) + (i < num_queries % len(groups))]]
              for i, df in enumerate(groups)]
    queries = pd.concat(picked).reset_index(drop=True)
    queries.insert(0, "category", category)
    print(f"Corpus: {len(corpus)} images, {corpus['relevant'].sum()} relevant ({category}). "
          f"Queries: {len(queries)} of {len(candidates) + num_dup} val candidates "
          f"({num_dup} near duplicates dropped)")
    return {"category": category, "corpus": corpus, "queries": queries}


def loadSetup(out_dir):
    """The setup written by a previous run (part5_corpus.csv, part5_queries.csv)."""
    read = lambda name: pd.read_csv(out_dir / name, dtype={"name": str, "nearest_corpus": str})
    queries = read("part5_queries.csv")
    return {"category": queries["category"].iloc[0], "corpus": read("part5_corpus.csv"), "queries": queries}


def corpusDuplicates(setup, threshold):
    """Pairs of corpus images whose DINOv2 cosine similarity is above ``threshold``."""
    corpus = setup["corpus"]
    x = deep(lookup(DUP_ENCODING, corpus["name"].tolist(), "val"))
    sims = np.triu(x @ x.T, 1)
    i, j = np.nonzero(sims > threshold)
    return pd.DataFrame({"name_a": corpus["name"].to_numpy()[i], "name_b": corpus["name"].to_numpy()[j],
                         "similarity": sims[i, j], "relevant_a": corpus["relevant"].to_numpy()[i],
                         "relevant_b": corpus["relevant"].to_numpy()[j]})


def pcaImages(num, seed):
    """``num`` protocol train images of any category, to fit PCA away from queries and corpus."""
    names = sorted({n for category in CATEGORIES + ["background"] for n in readList(category, "train")})
    return sorted(np.random.default_rng(seed).permutation(names)[:num].tolist())


################################################################################
# Ranking and metrics

def scores(Q, X, measure):
    """(num_queries, num_corpus) similarities."""
    if measure == "intersection":
        return np.stack([np.minimum(q, X).sum(axis=1) for q in Q])
    return Q @ X.T


def rankingMetrics(relevant, s, ks):
    """AP, P@K and R@K of one ranking; ``relevant`` is a boolean array over the corpus."""
    top = relevant[np.argsort(-s, kind="stable")]
    row = {"AP": average_precision_score(relevant, s)}
    row.update({f"P@{k}": top[:k].mean() for k in ks})
    row.update({f"R@{k}": top[:k].sum() / relevant.sum() for k in ks})
    return row


def meanMetrics(relevant, S, ks):
    """mAP, AP std and mean P@K / R@K over the rows (queries) of a score matrix."""
    df = pd.DataFrame([rankingMetrics(relevant, s, ks) for s in S])
    return {**df.mean().rename({"AP": "mAP"}).to_dict(), "AP_std": df["AP"].std()}


def queryTime(Q, X, measure, repeats=10):
    """ms per query: similarities to the whole corpus + full sort (best of ``repeats`` passes)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for q in Q:
            np.argsort(-scores(q[None], X, measure)[0])
        times.append(time.perf_counter() - start)
    return 1000 * min(times) / len(Q)


def extractionCost(encoding, results_dir):
    """ms per image to extract an encoding (NaN when it was never timed)."""
    meta = ENCODINGS_DIR / encoding / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["extract_ms_per_image"]
    for name in ["part3_summary.csv", "part2_encodings.csv"]:
        if (results_dir / name).exists():
            table = pd.read_csv(results_dir / name).drop_duplicates("encoding").set_index("encoding")
            if encoding in table.index:
                return table.loc[encoding, "sift_ms_per_image"] + table.loc[encoding, "encode_ms_per_image"]
    return np.nan


def costs(encoding, Q, X, measure, results_dir):
    return {"encoding": encoding, "dim": X.shape[1], "kb_per_image": X.shape[1] * 4 / 1024,
            "corpus_mb": X.size * 4 / 2**20, "extract_ms_per_image": extractionCost(encoding, results_dir),
            "query_ms": queryTime(Q, X, measure)}


################################################################################
# Experiments

def runMethods(setup, model, prompt_set, ks, results_dir):
    """Per-query metrics and costs of every method; also ``{method: score matrix}``."""
    relevant = setup["corpus"]["relevant"].to_numpy()
    queries = setup["queries"]
    rows, cost_rows, score_mats = [], [], {}
    for method, (encoding, transform, measure) in METHODS.items():
        Q, X = vectors(setup, encoding, transform)
        S = score_mats[method] = scores(Q, X, measure)
        rows += [{"method": method, "query": q, "group": g, **rankingMetrics(relevant, s, ks)}
                 for q, g, s in zip(queries["name"], queries["group"], S)]
        cost_rows.append({"method": method, **costs(encoding, Q, X, measure, results_dir)})
        print(f"{method:18s} mAP={np.mean([r['AP'] for r in rows[-len(S):]]):.3f}")

    # CLIP text-to-image: a single query, the Part 4 prompt set of the category.
    method = f"CLIP text ({prompt_set})"
    prompts = PROMPT_SETS[prompt_set][setup["category"]]
    start = time.perf_counter()
    t = encodeText(model, prompts)
    text_ms = 1000 * (time.perf_counter() - start)
    _, X = vectors(setup, CLIP_ENCODING, deep)
    S = score_mats[method] = scores(t[None], X, "cosine")
    rows.append({"method": method, "query": " | ".join(prompts), "group": "text",
                 **rankingMetrics(relevant, S[0], ks)})
    cost_rows.append({"method": method, **costs(CLIP_ENCODING, t[None], X, "cosine", results_dir),
                      "text_encode_ms": text_ms})
    print(f"{method:18s} AP={rows[-1]['AP']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(cost_rows), score_mats


def summarizeMethods(per_query, cost_table):
    metrics = ["AP"] + [c for c in per_query.columns if c.startswith(("P@", "R@"))]
    table = per_query.groupby("method", sort=False)[metrics].mean().rename(columns={"AP": "mAP"})
    table["AP_std"] = per_query.groupby("method", sort=False)["AP"].std()
    images = per_query[per_query["group"] != "text"]
    table = table.join(images.pivot_table(index="method", columns="group", values="AP").add_prefix("mAP "))
    return table.join(cost_table.set_index("method")).reset_index()


def runVocab(setup, vocab_ks, ks):
    relevant = setup["corpus"]["relevant"].to_numpy()
    rows = []
    for K in vocab_ks:
        encoding = f"bovw_K{K}_C0"
        for spatial in [False, True]:
            for similarity, (feature_map, measure) in BOVW_SIMILARITIES.items():
                Q, X = vectors(setup, encoding, handcrafted(encoding, spatial, feature_map))
                rows.append({"K": K, "spatial": spatial, "similarity": similarity, "dim": X.shape[1],
                             **meanMetrics(relevant, scores(Q, X, measure), ks),
                             "query_ms": queryTime(Q, X, measure)})
        print(f"vocab: K={K} done")
    return pd.DataFrame(rows)


def project(pca, x, d, whiten):
    """First ``d`` PCA components (one fit serves every d), L2 re-normalized as in Part 3."""
    z = (x - pca.mean_) @ pca.components_[:d].T
    if whiten:
        z = z / np.sqrt(pca.explained_variance_[:d])
    return l2(z)


def runPCA(setup, configs, dims, fit_names, seed, ks):
    relevant = setup["corpus"]["relevant"].to_numpy()
    rows = []
    for encoding, spatial, feature_map in configs:
        transform = handcrafted(encoding, spatial, feature_map)
        Q, X = vectors(setup, encoding, transform)
        base = {"encoding": encoding, "spatial": spatial, "feature_map": feature_map, "orig_dim": X.shape[1]}
        rows.append({**base, "pca_dim": 0, "whiten": False, "dim": X.shape[1],
                     **meanMetrics(relevant, scores(Q, X, "cosine"), ks),
                     "query_ms": queryTime(Q, X, "cosine"), "pca_fit_s": 0.0})

        dims_ok = [d for d in dims if d < min(X.shape[1], len(fit_names))]
        if not dims_ok:
            continue
        F = transform(lookup(encoding, fit_names, "train"))
        start = time.perf_counter()
        pca = PCA(n_components=max(dims_ok), svd_solver="randomized", random_state=seed).fit(F)
        fit_time = time.perf_counter() - start
        for d in dims_ok:
            for whiten in [False, True]:
                Qd, Xd = project(pca, Q, d, whiten), project(pca, X, d, whiten)
                rows.append({**base, "pca_dim": d, "whiten": whiten, "dim": d,
                             **meanMetrics(relevant, scores(Qd, Xd, "cosine"), ks),
                             "query_ms": queryTime(Qd, Xd, "cosine"), "pca_fit_s": fit_time})
        print(f"pca: {encoding} done")
    return pd.DataFrame(rows)


def runPrompts(setup, model, ks, reference="photo", depth=20):
    """One ranking per text query; Spearman and top-``depth`` overlap against ``reference``."""
    relevant = setup["corpus"]["relevant"].to_numpy()
    _, X = vectors(setup, CLIP_ENCODING, deep)
    queries = promptQueries(setup["category"])
    sims = {level: X @ encodeText(model, prompts) for level, prompts in queries.items()}
    ref_top = set(np.argsort(-sims[reference])[:depth])
    rows = []
    for level, prompts in queries.items():
        s = sims[level]
        prompts = [prompts] if isinstance(prompts, str) else prompts
        rows.append({"level": level, "num_prompts": len(prompts), "prompt": " | ".join(prompts),
                     **rankingMetrics(relevant, s, ks),
                     f"spearman_vs_{reference}": spearmanr(s, sims[reference]).statistic,
                     f"top{depth}_overlap_vs_{reference}": len(ref_top & set(np.argsort(-s)[:depth])) / depth})
    return pd.DataFrame(rows)


def runCorpus(setup, score_mats, fractions, seeds, ks):
    """Metrics on smaller corpora: all relevant images kept, fewer or other irrelevant ones."""
    corpus = setup["corpus"]
    relevant = corpus["relevant"].to_numpy()
    background = corpus["labels"].eq("background").to_numpy()
    negatives = np.flatnonzero(~relevant)
    variants = []
    for fraction in fractions:
        for seed in (seeds if fraction < 1 else seeds[:1]):
            keep = relevant.copy()
            keep[np.random.default_rng(seed).permutation(negatives)[:int(round(fraction * len(negatives)))]] = True
            variants.append(("irrelevant fraction", fraction, seed, keep))
    variants.append(("background only", np.nan, np.nan, relevant | background))
    variants.append(("other categories only", np.nan, np.nan, relevant | ~background))

    rows = []
    for method, S in score_mats.items():
        for variant, fraction, seed, keep in variants:
            rows.append({"method": method, "variant": variant, "fraction": fraction, "seed": seed,
                         "corpus_size": int(keep.sum()), "prevalence": relevant[keep].mean(),
                         **meanMetrics(relevant[keep], S[:, keep], ks)})
    return pd.DataFrame(rows)


################################################################################

def parseArgs():
    parser = argparse.ArgumentParser(description="Part 5: open-ended image retrieval challenge.")
    parser.add_argument("--category", default="horse", choices=CATEGORIES)
    parser.add_argument("--num-queries", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the query draw and the PCA images")
    parser.add_argument("--dup-threshold", type=float, default=0.9,
                        help="DINOv2 cosine similarity above which two images are near duplicates")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20, 50], help="K of P@K and R@K")
    parser.add_argument("--vocab-ks", nargs="+", type=int, default=VOCAB_KS)
    parser.add_argument("--pca-dims", nargs="+", type=int, default=PCA_DIMS)
    parser.add_argument("--pca-fit", type=int, default=1000, help="Train images used to fit PCA")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0],
                        help="Fractions of the irrelevant corpus images kept")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds of the corpus subsets")
    parser.add_argument("--skip", nargs="+", default=[], choices=EXPERIMENTS)
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    setup = buildSetup(args.category, args.num_queries, args.seed, args.dup_threshold)
    setup["corpus"].to_csv(out_dir / "part5_corpus.csv", index=False)
    setup["queries"].to_csv(out_dir / "part5_queries.csv", index=False)
    duplicates = corpusDuplicates(setup, args.dup_threshold)
    duplicates.to_csv(out_dir / "part5_corpus_duplicates.csv", index=False)
    print(f"Near-duplicate pairs inside the corpus: {len(duplicates)} "
          f"({int((duplicates['relevant_a'] & duplicates['relevant_b']).sum())} both relevant)")
    print(setup["queries"].round(3).to_string(index=False))

    model = loadClip()
    per_query, cost_table, score_mats = runMethods(setup, model, selectedPromptSet(out_dir), args.ks, out_dir)
    per_query.to_csv(out_dir / "part5_per_query.csv", index=False)
    summary = summarizeMethods(per_query, cost_table)
    summary.to_csv(out_dir / "part5_methods.csv", index=False)
    print("\n=== Methods: mean over queries ===")
    print(summary.round(3).to_string(index=False))

    experiments = {
        "vocab": ("part5_vocab.csv", lambda: runVocab(setup, args.vocab_ks, args.ks)),
        "pca": ("part5_pca.csv", lambda: runPCA(setup, PCA_CONFIGS, args.pca_dims,
                                                pcaImages(args.pca_fit, args.seed), args.seed, args.ks)),
        "prompts": ("part5_prompts.csv", lambda: runPrompts(setup, model, args.ks)),
        "corpus": ("part5_corpus_size.csv", lambda: runCorpus(setup, score_mats, args.fractions, args.seeds, args.ks)),
    }
    for name, (file_name, run) in experiments.items():
        if name in args.skip:
            continue
        table = run()
        table.to_csv(out_dir / file_name, index=False)
        if name == "corpus":
            table = table.groupby(["method", "variant", "fraction"], sort=False, dropna=False)[
                ["corpus_size", "prevalence", "mAP", "P@10"]].mean()
        print(f"\n=== {name} ===")
        print(table.round(3).to_string())
    print(f"\nResults written to {out_dir.resolve()}")
