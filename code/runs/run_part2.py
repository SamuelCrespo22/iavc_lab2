"""Part 2 experiments: VLAD and Fisher Vectors vs. BoVW with a linear SVM.

For each encoding in data/encodings and each category, trains the same linear SVM as Part 1 on the
same protocol splits, over:

  spatial      2x2 tiles vs. a single global vector (tiles summed)
  feature map  normalizations that suit the encoding type (see FEATURE_MAPS)

Per encoding, one setting (spatial + feature map) is selected on mean
validation AP over the categories; its per-class test AP and mAP are reported
together with feature dimension, storage, extraction and SVM times.

Results are written to <out>/part2_grid.csv and <out>/part2_encodings.csv.

Usage:
    python -m runs.run_part2
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoders import ImageEncoder
from src.splits import loadProtocolSubset
from runs.run_part1 import CATEGORIES, NUM_TILES, l2, loadData, removeSpatialInformation, trainAndEvaluate

ENCODINGS_DIR = Path("data/encodings")
ENCODERS_DIR = Path("data/encoders")


def signedSqrt(x):
    """Power normalization with alpha = 0.5: sign(x) * sqrt(|x|)."""
    return np.sign(x) * np.sqrt(np.abs(x))


def intraNorm(x, D):
    """L2-normalize each D-dimensional block (one VLAD residual per visual word).

    cyvlfeat stores VLAD as [v_1 | v_2 | ... | v_K] per tile, so the blocks are
    contiguous both with and without tiling.
    """
    N = x.shape[0]
    return l2(x.reshape(-1, D)).reshape(N, -1)


# Feature maps per encoding type, applied after the optional tile merging.
# VLAD tiles are stored L2-normalized; FV tiles are stored raw (no improved FV).
FEATURE_MAPS = {
    "bovw": {
        "l2": lambda x, D: l2(x),
        "sqrt_l2": lambda x, D: l2(np.sqrt(x)),
    },
    "vlad": {
        "none": lambda x, D: x,
        "l2": lambda x, D: l2(x),
        "ssr_l2": lambda x, D: l2(signedSqrt(x)),
        "intra_l2": lambda x, D: l2(intraNorm(x, D)),
        "ssr_intra_l2": lambda x, D: l2(intraNorm(signedSqrt(x), D)),
    },
    "fv": {
        "none": lambda x, D: x,
        "l2": lambda x, D: l2(x),
        "ssr_l2": lambda x, D: l2(signedSqrt(x)),
    },
}


def parseEncoding(name):
    """'vlad_K32_C80' -> ('vlad', 32, 80). C=0 means no PCA (D = 128)."""
    enc_type, K, C = re.fullmatch(r"(bovw|vlad|fv)_K(\d+)_C(\d+)", name).groups()
    return enc_type, int(K), int(C)


def defaultEncodings():
    found = sorted(p.name for p in ENCODINGS_DIR.iterdir()
                   if p.is_dir() and p.name.startswith(("vlad_", "fv_")))
    return ["bovw_K512_C0"] + found


def prepare(x, spatial, enc_type, feature_map, D):
    x = x.astype(np.float64)
    if not spatial:
        x = removeSpatialInformation(x)
    return FEATURE_MAPS[enc_type][feature_map](x, D)


def runGrid(encodings, categories, spatials, C):
    rows = []
    for encoding in encodings:
        enc_type, K, pca_dim = parseEncoding(encoding)
        D = pca_dim or 128
        for category in categories:
            data = loadData(category, encoding)
            for spatial in spatials:
                for feature_map in FEATURE_MAPS[enc_type]:
                    transform = lambda x: prepare(x, spatial, enc_type, feature_map, D)
                    res = trainAndEvaluate(data, transform, C)
                    rows.append({"encoding": encoding, "type": enc_type, "K": K,
                                 "category": category, "spatial": spatial,
                                 "feature_map": feature_map, **res})
                    print(f"{encoding:14s} {category:10s} spatial={spatial!s:5s} {feature_map:13s} "
                          f"dim={res['dim']:6d} AP train={res['ap_train']:.3f} "
                          f"val={res['ap_val']:.3f} test={res['ap_test']:.3f}")
    return pd.DataFrame(rows)


def measureEncodings(encodings, num_images):
    """Storage per encoding and, on ``num_images`` test images, extraction times.

    Dense SIFT is computed once per image (shared by all encoders); the
    encoding time covers PCA projection + encoding of the 4 tiles.
    """
    rows = {}
    for encoding in encodings:
        files = list((ENCODINGS_DIR / encoding).glob("*.npz"))
        sample = np.load(files[0])["histograms"]
        with np.load(ENCODERS_DIR / f"{encoding}.npz") as cfg:
            # Saved by src.train since fit times are recorded; NaN for older encoders.
            fit_time = float(cfg["fit_time_s"]) if "fit_time_s" in cfg.files else np.nan
        rows[encoding] = {
            "encoding": encoding,
            "dim": sample.shape[1],
            "dtype": str(sample.dtype),
            "kb_per_image": sample.shape[1] * sample.dtype.itemsize / 1024,
            "disk_mb": sum(f.stat().st_size for f in files) / 2**20,
            "codebook_fit_s": fit_time,
        }

    if num_images > 0:
        _, _, names = loadProtocolSubset(CATEGORIES[0], encodings[0], subset="test")
        images = [f"data/images/{name}.jpg" for name in names[:num_images]]
        encoders = {e: ImageEncoder(ENCODERS_DIR / f"{e}.npz") for e in encodings}
        sift_times, enc_times = [], {e: [] for e in encodings}
        # Untimed warm-up: the first call pays one-off setup, which would land on encodings[0].
        warm = encoders[encodings[0]].computeFeatures(images[0])
        for encoder in encoders.values():
            encoder.encode_tiling(*warm)
        for image in images:
            start = time.perf_counter()
            kpts, desc, isize = encoders[encodings[0]].computeFeatures(image)
            sift_times.append(time.perf_counter() - start)
            for encoding, encoder in encoders.items():
                start = time.perf_counter()
                encoder.encode_tiling(kpts, desc, isize)
                enc_times[encoding].append(time.perf_counter() - start)
        for encoding in encodings:
            rows[encoding]["sift_ms_per_image"] = 1000 * np.mean(sift_times)
            rows[encoding]["encode_ms_per_image"] = 1000 * np.mean(enc_times[encoding])

    return pd.DataFrame(rows.values())


def summarize(grid, encodings_info):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== Test AP per setting (rows) / category (cols) ===")
    for encoding, enc_df in grid.groupby("encoding", sort=False):
        print(f"\n{encoding}")
        print(enc_df.pivot_table(index=["spatial", "feature_map"], columns="category",
                                 values="ap_test").round(3))

    # One setting per encoding, chosen on mean val AP over the categories.
    mean_val = grid.groupby(["encoding", "spatial", "feature_map"], sort=False)["ap_val"].mean()
    best = mean_val.groupby(level="encoding", sort=False).idxmax()
    selected = pd.concat([grid.set_index(["encoding", "spatial", "feature_map"]).loc[[key]]
                          for key in best]).reset_index()

    print("\n=== Selected setting per encoding (on mean val AP): test AP ===")
    table = selected.pivot_table(index=["encoding", "spatial", "feature_map"],
                                 columns="category", values="ap_test", sort=False)
    table["mAP"] = table.mean(axis=1)
    print(table.round(3))

    costs = selected.groupby("encoding", sort=False)[["train_time_s", "infer_ms_per_image"]].mean()
    print("\n=== Cost per encoding (selected setting) ===")
    print(encodings_info.set_index("encoding").join(costs).join(
        table["mAP"].droplevel(["spatial", "feature_map"])).round(3))


def parseArgs():
    parser = argparse.ArgumentParser(description="Run all Part 2 (VLAD / FV vs. BoVW) experiments.")
    parser.add_argument("--encodings", nargs="+", default=None,
                        help="Encoding names in data/encodings (default: bovw_K512_C0 + all vlad_*/fv_*)")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (fixed, as in Part 1)")
    parser.add_argument("--tiled-only", action="store_true", help="Skip the global (no tiles) variant")
    parser.add_argument("--timing-images", type=int, default=5,
                        help="Images used to time SIFT + encoding; 0 disables")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    encodings = args.encodings or defaultEncodings()
    print("Encodings:", ", ".join(encodings))

    encodings_info = measureEncodings(encodings, args.timing_images)
    encodings_info.to_csv(out_dir / "part2_encodings.csv", index=False)

    spatials = [True] if args.tiled_only else [True, False]
    grid = runGrid(encodings, args.categories, spatials, args.C)
    grid.to_csv(out_dir / "part2_grid.csv", index=False)

    summarize(grid, encodings_info)
    print(f"\nResults written to {out_dir.resolve()}")
