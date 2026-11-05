"""Extract frozen pretrained-model descriptors as encodings.

Every image listed in data/splits goes through a frozen pretrained model with
its official preprocessing. Each model yields one or more encodings:

    resnet18        resnet18_layer2 / _layer3 / _layer4   GAP of the final 3
                    stages (128 / 256 / 512-d), ImageNet torchvision weights
    vit_b_32        vit_b_32                  class token after the final norm (768-d),
                                              ImageNet torchvision weights
    dinov2_vits14   dinov2_vits14_cls         CLS token after the final norm (384-d)
                    dinov2_vits14_patchmean   mean of the 256 patch tokens (384-d)
    clip_vit_b_32   clip_vit_b_32             CLIP image embedding (512-d), the
                    space the text prompts are compared in

Each encoding is saved as <output-dir>/<encoding>/<category>_<train|val>.npz
with the same keys as src.extract ("file_names", "histograms"), so the protocol
splits and loaders apply unchanged. With --flip, the descriptors of the
horizontally flipped images are stored too ("histograms_flip"). A meta.json in
each folder records the model, weights, hardware, versions, dimension and
extraction time.

Usage (from the directory containing README.md):
    python -m src.extract_deep                          # resnet18
    python -m src.extract_deep -m dinov2_vits14 clip_vit_b_32 --flip
"""

import os

# The cvc-a2 env links two OpenMP runtimes (torch and numpy/sklearn); torch must
# be imported before numpy and the duplicate allowed, otherwise the forward pass
# crashes. Ops missing on Apple MPS fall back to the CPU.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

import json
import platform
import time
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image

from .deep_models import IntermediateFeatureExtractor, getModel

# Output block of each of the final 3 ResNet-18 stages.
RESNET18_LAYERS = {"layer2": "layer2.1", "layer3": "layer3.1", "layer4": "layer4.1"}


def resnet18Model(device):
    model, preprocess = getModel("resnet18", device=device)
    extractor = IntermediateFeatureExtractor(model)

    def describe(x):
        _, blocks = extractor(x)
        return {f"resnet18_{layer}": blocks[block].mean(dim=(2, 3))
                for layer, block in RESNET18_LAYERS.items()}
    return preprocess, describe


def dinov2Model(device):
    model, preprocess = getModel("dinov2_vits14", device=device)

    def describe(x):
        out = model.forward_features(x)
        return {"dinov2_vits14_cls": out["x_norm_clstoken"],
                "dinov2_vits14_patchmean": out["x_norm_patchtokens"].mean(dim=1)}
    return preprocess, describe


def vitModel(device):
    # With the classification head replaced by Identity, the forward pass returns
    # the class token after the final LayerNorm.
    model, preprocess = getModel("vit_b_32", device=device)

    def describe(x):
        return {"vit_b_32": model(x)}
    return preprocess, describe


def clipModel(device):
    model, preprocess = getModel("clip_vit_b_32", device=device)

    def describe(x):
        return {"clip_vit_b_32": model.encode_image(x)}
    return preprocess, describe


# model name -> builder returning (preprocess, describe(batch) -> {encoding: (B, C)})
MODELS = {
    "resnet18": resnet18Model,
    "vit_b_32": vitModel,
    "dinov2_vits14": dinov2Model,
    "clip_vit_b_32": clipModel,
}

# Pretrained weights loaded by src.deep_models.getModel, recorded in meta.json.
WEIGHTS = {
    "resnet18": "torchvision ResNet18_Weights.IMAGENET1K_V1",
    "vit_b_32": "torchvision ViT_B_32_Weights.IMAGENET1K_V1",
    "dinov2_vits14": "torch.hub facebookresearch/dinov2 dinov2_vits14",
    "clip_vit_b_32": "openai/CLIP ViT-B/32",
}


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


def describeImages(model_name, names, img_dir, device, batch_size=64, flip=False):
    """``{encoding: (N, C)}`` for ``names``, plus ``{encoding}_flip`` with ``flip``."""
    preprocess, describe = MODELS[model_name](device)
    feats = {}
    for start in range(0, len(names), batch_size):
        batch = names[start:start + batch_size]
        x = torch.stack([preprocess(Image.open(Path(img_dir) / f"{n}.jpg").convert("RGB"))
                         for n in batch]).to(device)
        # Center crop is symmetric, so flipping the crop equals cropping the flipped image.
        views = {"": x, "_flip": torch.flip(x, dims=[3])} if flip else {"": x}
        with torch.inference_mode():
            for suffix, view in views.items():
                for encoding, value in describe(view).items():
                    feats.setdefault(encoding + suffix, []).append(value.float().cpu().numpy())
        print(f"\r{model_name}: {min(start + batch_size, len(names))}/{len(names)} images", end="")
    print()
    return {key: np.concatenate(value) for key, value in feats.items()}


def extractModel(model_name, split_dir="data/splits", img_dir="data/images",
                 output_dir="data/encodings", device=None, batch_size=64, flip=False, force=False):
    """Extract and save the encodings of one model; returns ``{encoding: meta}``."""
    device = device or defaultDevice()
    existing = sorted(Path(output_dir).glob("*/meta.json"))
    done = {p.parent.name: json.loads(p.read_text()) for p in existing}
    done = {enc: meta for enc, meta in done.items() if meta.get("model_name") == model_name}
    if done and not force:
        print(f"{model_name}: {', '.join(done)} already exist. Skipping (use --force to re-extract)")
        return done

    lists = readImageLists(split_dir)
    names = sorted({n for ids in lists.values() for n in ids})
    print(f"Extracting {model_name} features for {len(names)} images on {device}")
    start = time.perf_counter()
    feats = describeImages(model_name, names, img_dir, device, batch_size, flip)
    total_time = time.perf_counter() - start

    row = {n: i for i, n in enumerate(names)}
    metas = {}
    for encoding in [k for k in feats if not k.endswith("_flip")]:
        out_dir = Path(output_dir) / encoding
        out_dir.mkdir(parents=True, exist_ok=True)
        for (category, source), ids in lists.items():
            idx = [row[n] for n in ids]
            arrays = {"file_names": np.asarray(ids), "histograms": feats[encoding][idx]}
            if flip:
                arrays["histograms_flip"] = feats[f"{encoding}_flip"][idx]
            np.savez_compressed(out_dir / f"{category}_{source}.npz", **arrays)
        metas[encoding] = {
            "model_name": model_name, "weights": WEIGHTS[model_name], "device": device,
            "cpu": platform.processor(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "num_images": len(names), "flip": flip, "dim": int(feats[encoding].shape[1]),
            "extract_time_s": total_time,
            # One forward pass yields all encodings of the model (two passes with flip). The
            # time includes model loading, JPEG decoding and preprocessing.
            "extract_ms_per_image": 1000 * total_time / len(names) / (2 if flip else 1),
        }
        (out_dir / "meta.json").write_text(json.dumps(metas[encoding], indent=2))
    print(f"{model_name}: done in {total_time:.1f} s "
          f"({next(iter(metas.values()))['extract_ms_per_image']:.1f} ms/image)")
    return metas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frozen pretrained-model descriptors as encodings.")
    parser.add_argument("-m", "--models", nargs="+", default=["resnet18"], choices=list(MODELS))
    parser.add_argument("--data-dir", default="data/splits", help="Dataset split directory.")
    parser.add_argument("--img-dir", default="data/images", help="Image directory.")
    parser.add_argument("--output-dir", default="data/encodings", help="Encodings output directory.")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flip", action="store_true", help="Also store features of flipped images.")
    parser.add_argument("-f", "--force", action="store_true", help="Re-extract when output exists.")
    args = parser.parse_args()

    for model_name in args.models:
        extractModel(model_name, args.data_dir, args.img_dir, args.output_dir, args.device,
                     args.batch_size, args.flip, args.force)
