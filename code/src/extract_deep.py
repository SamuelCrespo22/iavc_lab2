"""Extract frozen pretrained-CNN descriptors as encodings.

Every image listed in data/splits goes through ImageNet ResNet-18 (torchvision
weights and preprocessing, backbone frozen). Global average pooling of the
outputs of the final 3 stages gives one descriptor per layer:

    layer2 (128-d)   layer3 (256-d)   layer4 (512-d)

Each is saved as <output-dir>/resnet18_<layer>/<category>_<train|val>.npz with
the same keys as src.extract ("file_names", "histograms"), so the protocol
splits and loaders apply unchanged. With --flip, the descriptors of the
horizontally flipped images are stored too ("histograms_flip"). A meta.json in
each folder records the device, dimensions and extraction time.

Usage (from the directory containing README.md):
    python -m src.extract_deep
    python -m src.extract_deep --flip --force
"""

import os

# The cvc-a2 env links two OpenMP runtimes (torch and numpy/sklearn); torch must
# be imported before numpy and the duplicate allowed, otherwise the forward pass
# crashes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .deep_models import IntermediateFeatureExtractor, getModel

# Output block of each of the final 3 ResNet-18 stages.
RESNET18_LAYERS = {"layer2": "layer2.1", "layer3": "layer3.1", "layer4": "layer4.1"}


def resnet18Encoding(layer):
    return f"resnet18_{layer}"


def defaultDevice():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def readImageLists(split_dir):
    """``{(category, source): [image ids]}`` for <split_dir>/<category>_<train|val>.txt."""
    lists = {}
    for path in sorted(Path(split_dir).glob("*.txt")):
        category, source = path.stem.split("_")
        with open(path, "r") as f:
            lists[(category, source)] = [l.strip() for l in f if l.strip()]
    return lists


def resnet18Descriptors(names, img_dir, device, batch_size=64, flip=False):
    """GAP descriptors per layer: ``{layer: (N, C)}``, plus ``{layer}_flip`` with ``flip``."""
    model, preprocess = getModel("resnet18", device=device)
    extractor = IntermediateFeatureExtractor(model)
    feats = {}
    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        x = torch.stack([preprocess(Image.open(Path(img_dir) / f"{n}.jpg").convert("RGB"))
                         for n in batch]).to(device)
        # Center crop is symmetric, so flipping the crop equals cropping the flipped image.
        views = {"": x, "_flip": torch.flip(x, dims=[3])} if flip else {"": x}
        with torch.inference_mode():
            for suffix, view in views.items():
                _, blocks = extractor(view)
                for layer, block in RESNET18_LAYERS.items():
                    pooled = blocks[block].mean(dim=(2, 3)).float().cpu().numpy()
                    feats.setdefault(layer + suffix, []).append(pooled)
        print(f"\rExtracted {min(start + batch_size, len(names))}/{len(names)} images", end="")
    print()
    extractor.remove_hooks()
    return {key: np.concatenate(value) for key, value in feats.items()}


def extractResNet18(split_dir="data/splits", img_dir="data/images", output_dir="data/encodings",
                    device=None, batch_size=64, flip=False, force=False):
    """Extract and save the ResNet-18 encodings; returns the meta dictionary."""
    device = device or defaultDevice()
    out_dirs = {layer: Path(output_dir) / resnet18Encoding(layer) for layer in RESNET18_LAYERS}
    meta_file = out_dirs["layer4"] / "meta.json"
    if meta_file.exists() and not force:
        print(f"{meta_file.parent} already exists. Skipping (use --force to re-extract)")
        return json.loads(meta_file.read_text())

    lists = readImageLists(split_dir)
    names = sorted({n for ids in lists.values() for n in ids})
    print(f"Extracting ResNet-18 features for {len(names)} images on {device}")
    start = time.perf_counter()
    feats = resnet18Descriptors(names, img_dir, device, batch_size, flip)
    total_time = time.perf_counter() - start

    row = {n: i for i, n in enumerate(names)}
    for layer, out_dir in out_dirs.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        for (category, source), ids in lists.items():
            idx = [row[n] for n in ids]
            arrays = {"file_names": np.asarray(ids), "histograms": feats[layer][idx]}
            if flip:
                arrays["histograms_flip"] = feats[f"{layer}_flip"][idx]
            np.savez_compressed(out_dir / f"{category}_{source}.npz", **arrays)

    meta = {"model": "resnet18 (torchvision ImageNet weights)", "device": device,
            "num_images": len(names), "flip": flip, "extract_time_s": total_time,
            # One forward pass yields all layers (two passes with flip).
            "extract_ms_per_image": 1000 * total_time / len(names) / (2 if flip else 1),
            "dims": {layer: int(feats[layer].shape[1]) for layer in RESNET18_LAYERS}}
    for out_dir in out_dirs.values():
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {total_time:.1f} s ({meta['extract_ms_per_image']:.1f} ms/image)")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frozen ResNet-18 descriptors as encodings.")
    parser.add_argument("--data-dir", default="data/splits", help="Dataset split directory.")
    parser.add_argument("--img-dir", default="data/images", help="Image directory.")
    parser.add_argument("--output-dir", default="data/encodings", help="Encodings output directory.")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flip", action="store_true", help="Also store features of flipped images.")
    parser.add_argument("-f", "--force", action="store_true", help="Re-extract when output exists.")
    args = parser.parse_args()

    extractResNet18(args.data_dir, args.img_dir, args.output_dir, args.device,
                    args.batch_size, args.flip, args.force)
