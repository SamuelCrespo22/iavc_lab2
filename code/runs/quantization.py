"""Quantization error of the trained encoders, measured after the fact.

Dense SIFT is computed for a random sample of protocol train images and of test
images (never seen by any codebook), keeping --per-image descriptors per image
as in src.train. The same descriptors are used for every encoder, so the
numbers are comparable across K. For each encoder in data/encoders:

  qerr_proj       mean squared distance to the nearest word, in the encoder's
                  own space (128-d SIFT, or the C-d PCA space)
  qerr_sift       the same in the original 128-d SIFT space, with the words
                  back-projected (includes the PCA truncation error), so C0 and
                  C80 encoders can be compared
  rel_qerr_sift   qerr_sift / total SIFT variance: fraction of the descriptor
                  variance the codebook does not capture
  unused_words_%  words that no sampled descriptor is assigned to
  pca_var_%       SIFT variance retained by the PCA projection (C80 only)
  loglik          FV only: mean GMM log-likelihood per descriptor (PCA space)

For FV the GMM means play the role of the words (hard, Euclidean assignment).

Results: <out>/quantization.csv and <out>/quantization.png.

Usage (from the directory containing README.md):
    python -m runs.quantization
    python -m runs.quantization --images 200 --encoders bovw_K256_C0 bovw_K1024_C0
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.encoders import FisherVectorEncoder, KMeansPredictor, PCAProjector
from src.features import extractFeatures
from src.utils import load_npz

ENCODERS_DIR = Path("data/encoders")
PROTOCOL_DIR = Path("data/splits_protocol")
IMG_DIR = Path("data/images")


def sampleImages(subset, num_images, seed):
    names = set()
    for path in PROTOCOL_DIR.glob(f"*_{subset}.txt"):
        with open(path, "r") as f:
            names.update(l.strip() for l in f if l.strip())
    rng = np.random.default_rng(seed)
    return rng.choice(sorted(names), size=min(num_images, len(names)), replace=False)


def sampleDescriptors(names, per_image, seed):
    descs = []
    for i, name in enumerate(names):
        _, desc, _ = extractFeatures(str(IMG_DIR / f"{name}.jpg"))
        if desc.shape[0] > per_image:
            rng = np.random.default_rng(seed + i)
            desc = desc[rng.choice(desc.shape[0], size=per_image, replace=False)]
        descs.append(desc.astype(np.float32))
        print(f"\rSIFT {i + 1}/{len(names)} images", end="")
    print()
    return np.vstack(descs)


def nearestWords(desc, centers, chunk=20000):
    """Index of and distance to the nearest center, in chunks to bound memory."""
    predictor = KMeansPredictor(centers)
    idx, dist = zip(*(predictor.predict_transform(desc[i:i + chunk])
                      for i in range(0, len(desc), chunk)))
    return np.concatenate(idx), np.concatenate(dist)


def quantizationStats(cfg, desc):
    proj = PCAProjector(cfg)
    desc_proj = proj.transform(desc).astype(np.float32)
    centers = cfg["means"] if cfg["type"] == "fv" else cfg["words"]
    idx, dist = nearestWords(desc_proj, centers)

    # Back-project the words to SIFT space (identity when there is no PCA).
    centers_sift = centers @ cfg["projection"] + cfg["projection_mean"] if proj.pca is not None else centers
    qerr_sift = np.mean(np.sum((desc - centers_sift[idx]) ** 2, axis=1))
    total_var = desc.var(axis=0).sum()

    stats = {
        "qerr_proj": np.mean(dist.astype(np.float64) ** 2),
        "qerr_sift": qerr_sift,
        "rel_qerr_sift": qerr_sift / total_var,
        "unused_words_%": 100 * (1 - len(np.unique(idx)) / len(centers)),
        "pca_var_%": 100 * desc_proj.var(axis=0).sum() / total_var if proj.pca is not None else 100.0,
    }
    if cfg["type"] == "fv":
        stats["loglik"] = FisherVectorEncoder(cfg).gmm.score(desc_proj.astype(np.float64))
    return stats


def plotErrors(df, out_file):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (enc_type, pca_dim, subset), group in df.groupby(["type", "pca_dim", "subset"]):
        group = group.sort_values("K")
        ax.plot(group["K"], group["rel_qerr_sift"], marker="o",
                linestyle="-" if subset == "test" else "--",
                label=f"{enc_type} C{pca_dim} ({subset})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of visual words K")
    ax.set_ylabel("Quantization error / SIFT variance")
    ax.set_title("Quantization error vs. vocabulary size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def parseArgs():
    parser = argparse.ArgumentParser(description="Quantization error of the trained encoders.")
    parser.add_argument("--encoders", nargs="+", default=None,
                        help="Encoder names in data/encoders (default: all)")
    parser.add_argument("--images", type=int, default=100, help="Images sampled per subset")
    parser.add_argument("--per-image", type=int, default=400, help="Descriptors kept per image")
    parser.add_argument("--subsets", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.encoders or sorted(p.stem for p in ENCODERS_DIR.glob("*.npz"))
    descs = {}
    for subset in args.subsets:
        print(f"Sampling {args.images} {subset} images")
        descs[subset] = sampleDescriptors(sampleImages(subset, args.images, args.seed),
                                          args.per_image, args.seed)

    rows = []
    for name in names:
        cfg = load_npz(ENCODERS_DIR / f"{name}.npz")
        for subset, desc in descs.items():
            rows.append({"encoder": name, "type": str(cfg["type"]), "K": int(cfg["num_words"]),
                         "pca_dim": int(cfg["pca_dim"]), "subset": subset,
                         "num_descriptors": len(desc), **quantizationStats(cfg, desc)})
            print(f"{name:14s} {subset:5s} rel_qerr_sift={rows[-1]['rel_qerr_sift']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "quantization.csv", index=False)
    plotErrors(df, out_dir / "quantization.png")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print()
    print(df.drop(columns=["num_descriptors"]).round(3).to_string(index=False))
    print(f"\nResults written to {out_dir.resolve()}")
