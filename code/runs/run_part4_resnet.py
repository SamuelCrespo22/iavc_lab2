"""Part 4, Stage A: frozen ResNet-18 features + linear SVM.

Evaluates the ResNet-18 encodings written by src.extract_deep (GAP of layer2 /
layer3 / layer4, 128 / 256 / 512-d, L2-normalized) with the Part 1 linear SVM
per category, against the BoVW / VLAD / FV settings selected in Part 2 (their
dimension, storage and extraction cost are read from part2_encodings.csv).
Extract the features first:

    python -m src.extract_deep --flip

Results: <out>/part4_resnet_grid.csv and <out>/part4_resnet_encodings.csv.

Usage (from the directory containing README.md):
    python -m runs.run_part4_resnet
    python -m runs.run_part4_resnet --no-baselines --categories motorbike
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runs.run_part1 import CATEGORIES, l2, loadData, trainAndEvaluate
from runs.run_part2 import ENCODINGS_DIR, parseEncoding
from runs.run_part2 import prepare as prepareHandcrafted

# Same names as src.extract_deep (not imported: it loads torch, which clashes
# with numpy's OpenMP runtime in the cvc-a2 env).
LAYERS = ["layer2", "layer3", "layer4"]

# Deep features are L2-normalized, the same map for every model.
def deepTransform(x):
    return l2(x.astype(np.float64))

# Handcrafted baselines: settings selected on validation AP in Part 2.
BASELINES = [
    ("bovw_K512_C0", True, "sqrt_l2"),
    ("vlad_K32_C80", True, "ssr_intra_l2"),
    ("fv_K32_C80", True, "ssr_l2"),
]


def encodingName(layer):
    return f"resnet18_{layer}"


def runGrid(categories, baselines, C):
    configs = [(encodingName(layer), False, "l2", deepTransform) for layer in LAYERS]
    for encoding, spatial, fm in baselines:
        enc_type, _, pca_dim = parseEncoding(encoding)
        configs.append((encoding, spatial, fm,
                        lambda x, s=spatial, t=enc_type, fm=fm, D=pca_dim or 128:
                            prepareHandcrafted(x, s, t, fm, D)))

    rows = []
    for encoding, spatial, fm, transform in configs:
        for category in categories:
            data = loadData(category, encoding)
            res = trainAndEvaluate(data, transform, C)
            rows.append({"encoding": encoding, "spatial": spatial, "feature_map": fm,
                         "category": category, **res})
            print(f"{encoding:16s} {category:10s} {fm:8s} dim={res['dim']:6d} "
                  f"AP train={res['ap_train']:.3f} val={res['ap_val']:.3f} test={res['ap_test']:.3f}")
    return pd.DataFrame(rows)


def deepEncodingInfo(encoding):
    """Dimension, storage and extraction cost of an encoding written by src.extract_deep."""
    enc_dir = ENCODINGS_DIR / encoding
    meta = json.loads((enc_dir / "meta.json").read_text())
    return {"encoding": encoding, "dim": meta["dim"], "dtype": "float32",
            "kb_per_image": meta["dim"] * 4 / 1024,
            "disk_mb": sum(f.stat().st_size for f in enc_dir.glob("*.npz")) / 2**20,
            "extract_ms_per_image": meta["extract_ms_per_image"],
            "device": meta["device"]}


def encodingsInfo(baselines, out_dir):
    """Deep encodings from meta.json; handcrafted ones from the Part 2 measurements."""
    info = pd.DataFrame([deepEncodingInfo(encodingName(layer)) for layer in LAYERS])
    if baselines:
        part2 = out_dir / "part2_encodings.csv"
        if not part2.exists():
            raise SystemExit(f"Missing {part2}: run `python -m runs.run_part2` first")
        hand = pd.read_csv(part2).set_index("encoding").loc[[b[0] for b in baselines]].reset_index()
        hand["extract_ms_per_image"] = hand["sift_ms_per_image"] + hand["encode_ms_per_image"]
        hand["device"] = "cpu"
        info = pd.concat([info, hand[[c for c in info.columns if c in hand]]], ignore_index=True)
    return info


def summarize(grid, info):
    """Print test AP per encoding, then its costs."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== Test AP per encoding ===")
    table = grid.pivot_table(index="encoding", columns="category", values="ap_test", sort=False)
    table["mAP"] = table.mean(axis=1)
    print(table.round(3).to_string())

    print("\n=== Costs per encoding ===")
    costs = grid.groupby("encoding", sort=False)[["train_time_s", "infer_ms_per_image"]].mean()
    summary = info.set_index("encoding").join(costs)
    summary["mAP"] = table["mAP"]
    print(summary.round(3).to_string())


def parseArgs():
    parser = argparse.ArgumentParser(description="Part 4 Stage A: frozen ResNet-18 features + linear SVM.")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--C", type=float, default=10.0, help="SVM C (fixed, as in Part 1)")
    parser.add_argument("--no-baselines", action="store_true", help="Skip BoVW / VLAD / FV")
    parser.add_argument("--out", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [encodingName(l) for l in LAYERS if not (ENCODINGS_DIR / encodingName(l) / "meta.json").exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}: run `python -m src.extract_deep -m resnet18` first")
    baselines = [] if args.no_baselines else [b for b in BASELINES if (ENCODINGS_DIR / b[0]).is_dir()]

    info = encodingsInfo(baselines, out_dir)
    info.to_csv(out_dir / "part4_resnet_encodings.csv", index=False)

    grid = runGrid(args.categories, baselines, args.C)
    grid.to_csv(out_dir / "part4_resnet_grid.csv", index=False)

    summarize(grid, info)
    print(f"\nResults written to {out_dir.resolve()}")
