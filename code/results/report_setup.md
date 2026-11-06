# Setup and protocol of every part

## Part 1 — Handcrafted features: BoVW and linear SVM

Setup, the same for every stage:

- Splits: `data/splits_protocol` (train / validation = 80 / 20 of the supplied train files, test = the supplied val files). Each category is a binary task: its images are positives, all background images of the same subset are negatives.
- Features: dense SIFT (`vl_phow`, step 4); k-means vocabulary learned on training descriptors (supplied K = 512, plus K = 256 and 1024); histograms of the 2x2 tiles concatenated (dimension 4K). `global` = the 4 tile histograms summed into one, i.e. the histogram of the whole image (dimension K).
- Feature maps: `none`, `l1`, `l2` (Stage E); `sqrt` = square root only, `sqrt_l1` = L1 after the root, `sqrt_l2` = L2 after the root = Hellinger map (Stage F).
- Classifier: linear SVM (`LinearSVC`, hinge loss), C = 10 fixed, as the assignment suggests.
- Selection: settings are compared on validation mAP (mean AP over the categories); the test set only reports the chosen setting. AP = scikit-learn `average_precision_score`, positive class = the category.
- Chosen setting (highest validation mAP of the whole grid): K = 512, 2x2 tiles, `sqrt_l2`. It is used in Stages B–D and is the BoVW baseline of Parts 2 and 4.
- Figures of Stages A–D: `python -m runs.run_part1_inspect`.
- Environment: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, no GPU used; Python 3.11.9, numpy 2.4.6, scikit-learn 1.9.1, torch 2.14.0, torchvision 0.29.0, cyvlfeat 0.7.1.


## Part 2 — Advanced encodings: VLAD and Fisher Vectors

Setup, the same for every stage:

- Same splits, categories and linear SVM (C = 10) as Part 1.
- Local descriptors: the same dense SIFT, reduced to 80 dimensions by PCA (`C80`). VLAD: k-means with K words, residuals to the assigned word. FV: GMM with K diagonal components, soft assignment, first- and second-order statistics. PCA and codebooks are fitted on training descriptors only.
- 2x2 tiles as in Part 1. Dimensions: VLAD 4·K·80; FV 4·(2·K·80 + K), i.e. the 2K × d of the assignment plus the K mixture-weight terms of the implementation, per tile. `global` = the 4 tile vectors summed. VLAD tiles are stored L2-normalized and FV tiles are averages over the tile's descriptors, so this is an approximation of a VLAD / FV computed on the whole image, not the same vector.
- Normalizations: `none`, `l2`, `ssr_l2` = signed square root + L2 (power normalization), `intra_l2` = L2 of each word's residual block, then global L2 (VLAD only), `ssr_intra_l2` = both.
- Selection: per encoding, the layout and normalization with the highest validation mAP; the test set only reports that setting.
- BoVW baseline: bovw_K512_C0, setting selected in the same way (the Part 1 choice).

## Part 3 — Large visual vocabularies with PCA vs. smaller vocabularies

Setup, the same for every table:

- BoVW: one global histogram per image (the 2x2 tiles of raw counts summed, dimension K), Hellinger map. VLAD / FV: 2x2 tiles, signed square root + L2. Dimensions: VLAD 4·K·80; FV 4·(2·K·80 + K), i.e. the 2K × d of the assignment plus the K mixture-weight terms of the implementation, per tile.
- VLAD normalization differs from Part 2 on purpose: on validation, Part 2 picked `ssr_l2` for K = 16 and `ssr_intra_l2` for K = 32 / 64, so following it would change the normalization along K. Part 3 keeps one normalization for every K, and the same one for VLAD and FV, so that only K and PCA vary.
- PCA is fitted on the training features only and its output is L2-normalized; d < original dimension and d ≤ number of training images − 1.
- Linear SVM per category (positives vs. background), C = 10, the Part 1 splits.
- Nothing is selected in Part 3: every condition is reported on the test set, as the assignment asks; validation mAP is in `part3_summary.csv`.
- 3 seeds: each trains on a different stratified 90 % of the training set (PCA refitted); validation and test are complete. mAP = test AP averaged over the 5 categories, ± = std over seeds. Accuracy and macro-F1 use the SVM threshold 0.
- Codebooks (PCA80 + k-means / GMM, fitted once per K) are learned without labels from descriptors of the protocol training images only (733 images, 400 descriptors each, the same sample for every encoder); validation and test images are never seen.
- Times, all on CPU: SIFT + encoding per image on 5 images, codebook fit once, PCA and SVM fit per category, inference = PCA projection + SVM per test image.
- Storage (`kb_per_image`) = final dimension × 4 bytes (float32) for every representation, so that sizes are comparable; Part 2 reports the arrays as stored (BoVW int64, FV float64).


## Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP

Protocol, the same for every stage:

- Splits: every method reads the same `data/splits_protocol` lists (train / validation = 80 / 20 of the supplied train files, test = the supplied val files): the same images, in the same order, with the same labels; no image is in two subsets.
- Selection: deep features are L2-normalized and the SVM uses C = 10, both fixed in advance; the choices of Stages D and E (C, prompt set) are made on validation AP and the test set only reports the chosen setting.
- Costs: feature dimension and extraction time per image are stored by `src.extract_deep` in `meta.json` (forward pass on CPU over all 4946 images, original + flip, including model loading, JPEG decoding and preprocessing). Inference = feature map + linear SVM per test image, features already extracted.
- Models (frozen, official preprocessing, center crop 224): ResNet-18 and ViT-B/32 with torchvision `IMAGENET1K_V1` weights; DINOv2 ViT-S/14 from torch.hub `facebookresearch/dinov2`; CLIP ViT-B/32 from `openai/CLIP`.
- Environment: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, no GPU used; Python 3.11.9, numpy 2.4.6, scikit-learn 1.9.1, torch 2.14.0, torchvision 0.29.0, cyvlfeat 0.7.1.
