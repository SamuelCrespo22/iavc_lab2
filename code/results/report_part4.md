# Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP

## Stage A — Frozen ResNet-18 features

> **Pending:** Stage A — Frozen ResNet-18 features — run `python -m src.extract_deep -m resnet18 && python -m runs.run_part4_resnet`.

## Stage B — DINOv2 ViT-S/14

> **Pending:** Stage B — DINOv2 ViT-S/14 — run `python -m src.extract_deep -m dinov2_vits14 --flip && python -m runs.run_part4_dino`.

## Stages C–F

> **Pending:** few-shot and augmentation (C), choice of C (D), CLIP zero-shot (E) and visual analysis (F) have no script yet.

## Questions

**Stage A**

- How do learned features differ from handcrafted local descriptors? Does a higher-dimensional feature necessarily give better AP?
  - *Answer:*

**Stage B**

- Why can DINOv2 features be useful with few labeled examples even though DINOv2 is not trained as a classifier for these target categories?
  - *Answer:*
- What information might be lost when patch tokens are replaced by a single global descriptor?
  - *Answer:*

**Stage C**

- How much does pretrained representation quality reduce dependence on labeled examples?
  - *Answer:*
- When could horizontal flipping harm recognition? Is averaging feature vectors equivalent to averaging classifier scores?
  - *Answer:*

**Stage D**

- How does regularization affect the bias-variance trade-off when only a few positive examples are available?
  - *Answer:*

**Stage E**

- How does CLIP zero-shot recognition differ from a linear SVM trained on frozen CLIP image embeddings?
  - *Answer:*
- How sensitive are predictions to prompt wording? What are the risks of selecting prompts based on the test set?
  - *Answer:*

**Stage F**

- Do the highlighted regions correspond to the object? What can and cannot be concluded from a saliency visualization?
  - *Answer:*
- Which categories and visual conditions remain difficult, and what evidence supports your explanation?
  - *Answer:*

