"""Part 5: figures of the retrieval challenge.

Reads the CSVs written by runs.run_part5 (run it first, with the same --out)
and draws:

  part5_per_query.png        AP of every query per method, by query group
  part5_vocab_pca.png        BoVW mAP vs. K; mAP vs. dimension with PCA
  part5_prompts.png          CLIP text-to-image AP per wording of the query
  part5_corpus_size.png      mAP vs. corpus size (fewer irrelevant images)
  part5_ranked_hard.png      top results of every image method for the query with
                             the lowest mean AP over methods (or --query)

In the ranked grid the query has a blue frame, relevant results a green one and
irrelevant results a red one, with the similarity score above.

Usage:
    python -m runs.run_part5_visual
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
from PIL import Image, ImageOps

from runs.run_part5 import IMG_DIR, METHODS, loadSetup, scores, vectors

FRAME = {"query": "tab:blue", True: "tab:green", False: "tab:red"}


def optional(out_dir, name):
    path = out_dir / name
    return pd.read_csv(path) if path.exists() else None


def plotPerQuery(per_query, prevalence, path):
    methods = list(dict.fromkeys(per_query["method"]))
    colors = {"with person": "tab:orange", "without person": "tab:blue", "text": "tab:red"}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 2, 4))
    seen = set()
    for i, method in enumerate(methods):
        df = per_query[per_query["method"] == method]
        for group, gdf in df.groupby("group"):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(gdf)), gdf["AP"], s=20, color=colors.get(group, "k"),
                       marker="D" if group == "text" else "o", label=None if group in seen else group)
            seen.add(group)
        ax.hlines(df["AP"].mean(), i - 0.3, i + 0.3, color="k", lw=1.5)
    ax.axhline(prevalence, color="gray", ls=":", label="random ranking")
    ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AP")
    ax.set_title("AP per query (black line: mAP)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plotVocabPCA(vocab, pca, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    if vocab is not None:
        ax.plot(vocab["K"], vocab["mAP"], marker="o", label="global BoVW, Hellinger")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=7)
    ax.set_xlabel("Vocabulary size K")
    ax.set_ylabel("mAP")
    ax.set_title("BoVW: vocabulary size")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if pca is not None:
        for i, (encoding, df) in enumerate(pca.groupby("encoding", sort=False)):
            reduced = df[df["pca_dim"] > 0]
            ax.plot(reduced["dim"], reduced["mAP"], marker="o", color=f"C{i}", label=f"{encoding} + PCA")
            full = df[df["pca_dim"] == 0].iloc[0]
            ax.scatter(full["dim"], full["mAP"], marker="*", s=140, color=f"C{i}", label=f"{encoding}, no PCA")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Final dimension")
    ax.set_title("PCA: dimension vs. retrieval quality")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plotPrompts(prompts, prevalence, path):
    overlap = next(c for c in prompts.columns if c.startswith("top"))
    y = np.arange(len(prompts))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(prompts) + 1.3))
    ax.barh(y, prompts["AP"], color="tab:blue")
    for yi, (_, row) in zip(y, prompts.iterrows()):
        ax.text(row["AP"] + 0.01, yi, f"AP {row['AP']:.3f}   P@10 {row['P@10']:.2f}   "
                f"top-20 overlap {row[overlap]:.2f}", va="center", fontsize=7)
    ax.axvline(prevalence, color="gray", ls=":")
    ax.set_yticks(y, prompts["level"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.6)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("AP of the text query")
    ax.set_title("CLIP text-to-image: wording of the query (overlap with the 'photo' top 20)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plotCorpus(corpus, path):
    subsets = corpus[corpus["variant"] == "irrelevant fraction"]
    stats = subsets.groupby(["method", "corpus_size"], sort=False)["mAP"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, df in stats.groupby("method", sort=False):
        ax.errorbar(df["corpus_size"], df["mean"], yerr=df["std"].fillna(0), marker="o", capsize=3, label=method)
    prevalence = subsets.groupby("corpus_size")["prevalence"].mean()
    ax.plot(prevalence.index, prevalence.values, color="gray", ls=":", label="random ranking")
    ax.set_xlabel("Corpus size (every relevant image kept, irrelevant ones subsampled)")
    ax.set_ylabel("mAP (mean +- std over subsets)")
    ax.set_title("Corpus size and class imbalance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################################################################################
# Ranked lists

def thumbnail(name, size=160):
    return ImageOps.fit(Image.open(IMG_DIR / f"{name}.jpg").convert("RGB"), (size, size))


def imageRankings(setup, methods, query, num):
    """``{method: [(name, relevant, score)]}``: the top ``num`` corpus images for one query image."""
    corpus = setup["corpus"]
    idx = setup["queries"].index[setup["queries"]["name"] == query][0]
    out = {}
    for method in methods:
        Q, X = vectors(setup, *METHODS[method])
        s = scores(Q[idx:idx + 1], X)[0]
        top = np.argsort(-s, kind="stable")[:num]
        out[method] = list(zip(corpus["name"].iloc[top], corpus["relevant"].iloc[top], s[top]))
    return out


def frame(ax, color):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)


def plotRanked(rankings, query, title, path):
    num = max(len(r) for r in rankings.values())
    fig, axes = plt.subplots(len(rankings), num + 1, figsize=(1.45 * (num + 1), 1.6 * len(rankings)),
                             squeeze=False)
    for r, (row_name, ranking) in enumerate(rankings.items()):
        axes[r][0].imshow(thumbnail(query))
        frame(axes[r][0], FRAME["query"])
        axes[r][0].set_title("query", fontsize=7)
        axes[r][0].set_ylabel(row_name, fontsize=8)
        for c, (name, relevant, score) in enumerate(ranking):
            ax = axes[r][c + 1]
            ax.imshow(thumbnail(name))
            frame(ax, FRAME[bool(relevant)])
            ax.set_title(f"{c + 1}: {score:.2f}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 5: figures of the retrieval challenge.")
    parser.add_argument("--num", type=int, default=8, help="Results shown per ranked list")
    parser.add_argument("--query", default=None,
                        help="Query image of the ranked grid (default: lowest mean AP over methods)")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    if not (out_dir / "part5_methods.csv").exists():
        raise SystemExit("Missing part5_methods.csv: run `python -m runs.run_part5` first")
    setup = loadSetup(out_dir)
    prevalence = setup["corpus"]["relevant"].mean()
    per_query = pd.read_csv(out_dir / "part5_per_query.csv", dtype={"query": str})

    plotPerQuery(per_query, prevalence, out_dir / "part5_per_query.png")
    plotVocabPCA(optional(out_dir, "part5_vocab.csv"), optional(out_dir, "part5_pca.csv"),
                 out_dir / "part5_vocab_pca.png")
    prompts = optional(out_dir, "part5_prompts.csv")
    if prompts is not None:
        plotPrompts(prompts, prevalence, out_dir / "part5_prompts.png")
    corpus = optional(out_dir, "part5_corpus_size.csv")
    if corpus is not None:
        plotCorpus(corpus, out_dir / "part5_corpus_size.png")

    # Ranked list of every image method for the hardest query (successes and failures).
    methods = list(METHODS)
    mean_ap = per_query[per_query["method"].isin(methods)].groupby("query")["AP"].mean()
    query = args.query or mean_ap.idxmin()
    plotRanked(imageRankings(setup, methods, query, args.num), query,
               f"Query {query} (mean AP over methods {mean_ap[query]:.3f}): green = relevant, red = irrelevant",
               out_dir / "part5_ranked_hard.png")
    print(f"Figures written to {out_dir.resolve()}")
