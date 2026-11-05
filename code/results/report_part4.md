# Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP

## Stage A — frozen CNN features (ResNet-18)

> Global average pooling of each of the 3 last ResNet-18 stages, linear SVM on the frozen features, compared with the BoVW / VLAD / FV settings selected in Part 2.

Validation mAP per feature map. The map with the highest validation mAP is selected:

| encoding        | none   | l2    | sqrt_l2   | ssr_intra_l2   | ssr_l2   | selected     |
|:----------------|:-------|:------|:----------|:---------------|:---------|:-------------|
| resnet18_layer2 | 0.803  | 0.814 | 0.792     | –              | –        | l2           |
| resnet18_layer3 | 0.903  | 0.906 | 0.905     | –              | –        | l2           |
| resnet18_layer4 | 0.955  | 0.969 | 0.968     | –              | –        | l2           |
| bovw_K512_C0    | –      | –     | 0.717     | –              | –        | sqrt_l2      |
| vlad_K32_C80    | –      | –     | –         | 0.793          | –        | ssr_intra_l2 |
| fv_K32_C80      | –      | –     | –         | –              | 0.784    | ssr_l2       |

Test AP of the selected setting, with feature dimension, extraction time and inference cost per image:

| encoding        |   motorbike |   aeroplane |   person |   horse |   car |   mAP |   dim |   extract_ms_per_image |   infer_us_per_image |
|:----------------|------------:|------------:|---------:|--------:|------:|------:|------:|-----------------------:|---------------------:|
| resnet18_layer2 |       0.767 |       0.768 |    0.857 |   0.725 | 0.786 | 0.781 |   128 |                 18.257 |                0.725 |
| resnet18_layer3 |       0.894 |       0.881 |    0.963 |   0.876 | 0.902 | 0.903 |   256 |                 18.257 |                1.597 |
| resnet18_layer4 |       0.974 |       0.968 |    0.973 |   0.954 | 0.954 | 0.964 |   512 |                 18.257 |                3.133 |
| bovw_K512_C0    |       0.632 |       0.691 |    0.750 |   0.662 | 0.734 | 0.694 |  2048 |                239.594 |               19.460 |
| vlad_K32_C80    |       0.707 |       0.726 |    0.768 |   0.735 | 0.793 | 0.746 | 10240 |                207.029 |              227.124 |
| fv_K32_C80      |       0.783 |       0.759 |    0.811 |   0.758 | 0.807 | 0.784 | 20608 |                295.513 |              360.781 |

## Stage B — DINOv2 ViT-S/14

> Frozen DINOv2 global embedding (CLS token) + linear SVM: per-class AP, mAP, confusion matrix, dimension, extraction and inference cost. Optional extension: the mean of the patch tokens (`patchmean`). ResNet-18 layer4, the ViT-B/32 class token (the "CNN/ViT" row of the final table; signed features, so no square root) and CLIP are shown for reference.

Validation mAP per feature map. The map with the highest validation mAP is selected:

| encoding                |   none |    l2 | selected   |
|:------------------------|-------:|------:|:-----------|
| dinov2_vits14_cls       |  0.966 | 0.980 | l2         |
| dinov2_vits14_patchmean |  0.982 | 0.983 | l2         |
| resnet18_layer4         |  0.955 | 0.969 | l2         |
| vit_b_32                |  0.944 | 0.957 | l2         |
| clip_vit_b_32           |  0.989 | 0.990 | l2         |

Test AP of the selected setting, with feature dimension, extraction time and inference cost per image:

| encoding                |   motorbike |   aeroplane |   person |   horse |   car |   mAP |   dim |   extract_ms_per_image |   infer_us_per_image |
|:------------------------|------------:|------------:|---------:|--------:|------:|------:|------:|-----------------------:|---------------------:|
| dinov2_vits14_cls       |       0.982 |       1.000 |    0.978 |   0.994 | 0.951 | 0.981 |   384 |                 52.278 |                2.538 |
| dinov2_vits14_patchmean |       0.980 |       1.000 |    0.988 |   0.985 | 0.967 | 0.984 |   384 |                 52.278 |                2.242 |
| resnet18_layer4         |       0.974 |       0.968 |    0.973 |   0.954 | 0.954 | 0.964 |   512 |                 18.257 |                3.354 |
| vit_b_32                |       0.982 |       0.987 |    0.937 |   0.954 | 0.945 | 0.961 |   768 |                 35.005 |                5.757 |
| clip_vit_b_32           |       0.994 |       0.998 |    0.991 |   0.992 | 0.979 | 0.991 |   512 |                 35.182 |                3.159 |

Test confusion matrix — `dinov2_vits14_cls` (rows: true, columns: predicted). Each image goes to the category whose SVM scores highest, or to background when every score is negative; images with several categories are left out.

| true       |   motorbike |   aeroplane |   person |   horse |   car |   background |
|:-----------|------------:|------------:|---------:|--------:|------:|-------------:|
| motorbike  |           8 |           0 |        8 |       0 |    18 |            3 |
| aeroplane  |           0 |          81 |       24 |       0 |     7 |            0 |
| person     |           0 |           1 |      625 |       0 |    10 |           48 |
| horse      |           0 |           0 |        5 |      23 |     0 |            3 |
| car        |           1 |           0 |       37 |       0 |   171 |           19 |
| background |           0 |           2 |       67 |       0 |    25 |          983 |

Test confusion matrix — `dinov2_vits14_patchmean` (rows: true, columns: predicted). Each image goes to the category whose SVM scores highest, or to background when every score is negative; images with several categories are left out.

| true       |   motorbike |   aeroplane |   person |   horse |   car |   background |
|:-----------|------------:|------------:|---------:|--------:|------:|-------------:|
| motorbike  |          21 |           0 |        7 |       0 |     6 |            3 |
| aeroplane  |           0 |          87 |       14 |       0 |    11 |            0 |
| person     |           1 |           1 |      617 |       0 |    15 |           50 |
| horse      |           0 |           0 |        3 |      24 |     1 |            3 |
| car        |           1 |           1 |       37 |       0 |   173 |           16 |
| background |           1 |           0 |       31 |       0 |    24 |         1021 |

## Stage C — few-shot learning and flip augmentation

> One representation (DINOv2 CLS, L2) trained with 1, 5, 10, 20, 50 and all positives; no augmentation vs. horizontal flip of the training images; validation AP, mean ± std over 5 random draws of the positives.

Validation AP against the number of positive training images. Negatives: every background training image. `flip` adds the flipped training images; validation images are never flipped. `all` is a single draw.

| category   | aug   | 1             | 5             | 10            | 20            | 50            |   all |
|:-----------|:------|:--------------|:--------------|:--------------|:--------------|:--------------|------:|
| aeroplane  | flip  | 0.831 ± 0.067 | 0.993 ± 0.008 | 0.989 ± 0.024 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |
| aeroplane  | none  | 0.832 ± 0.071 | 0.992 ± 0.007 | 0.988 ± 0.027 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |
| car        | flip  | 0.686 ± 0.142 | 0.874 ± 0.027 | 0.895 ± 0.028 | 0.909 ± 0.028 | 0.937 ± 0.020 | 0.951 |
| car        | none  | 0.685 ± 0.130 | 0.868 ± 0.032 | 0.891 ± 0.025 | 0.905 ± 0.035 | 0.935 ± 0.015 | 0.945 |
| horse      | flip  | 0.959 ± 0.033 | 0.999 ± 0.002 | 0.998 ± 0.005 | 0.998 ± 0.002 | 1.000 ± 0.001 | 0.997 |
| horse      | none  | 0.960 ± 0.036 | 0.999 ± 0.002 | 0.998 ± 0.005 | 0.998 ± 0.003 | 1.000 ± 0.001 | 0.997 |
| motorbike  | flip  | 0.725 ± 0.358 | 0.969 ± 0.016 | 0.972 ± 0.013 | 0.973 ± 0.006 | 0.980 ± 0.011 | 0.983 |
| motorbike  | none  | 0.701 ± 0.356 | 0.971 ± 0.015 | 0.973 ± 0.011 | 0.977 ± 0.005 | 0.982 ± 0.009 | 0.979 |
| person     | flip  | 0.716 ± 0.064 | 0.894 ± 0.042 | 0.900 ± 0.017 | 0.950 ± 0.009 | 0.956 ± 0.007 | 0.981 |
| person     | none  | 0.718 ± 0.058 | 0.894 ± 0.042 | 0.900 ± 0.016 | 0.949 ± 0.008 | 0.954 ± 0.005 | 0.978 |
| mAP        | flip  | 0.784 ± 0.057 | 0.946 ± 0.011 | 0.951 ± 0.004 | 0.966 ± 0.006 | 0.975 ± 0.006 | 0.982 |
| mAP        | none  | 0.779 ± 0.060 | 0.945 ± 0.013 | 0.950 ± 0.005 | 0.966 ± 0.007 | 0.974 ± 0.005 | 0.980 |

![part4_fewshot](part4_fewshot.png)

Test-time flip, a separate condition (all positives, validation mAP): `score_avg` averages the SVM scores of an image and its flip, `feature_avg` scores the mean of the two features.

| train_aug   |   no_tta |   score_avg |   feature_avg |
|:------------|---------:|------------:|--------------:|
| flip        |   0.9823 |      0.9816 |        0.9818 |
| none        |   0.9797 |      0.9796 |        0.9799 |

## Stage D — choice of the SVM C

> C in 0.1 … 1000, train and validation AP to diagnose overfitting, C selected on validation and then evaluated once on the test set. Done with 5 positives and with all positives.

Train and validation mAP (mean ± std over draws) per C:

| regime        |        C | ap_train      | ap_val        |
|:--------------|---------:|:--------------|:--------------|
| 5 positives   |    0.100 | 1.000 ± 0.000 | 0.955 ± 0.008 |
| 5 positives   |    1.000 | 1.000 ± 0.000 | 0.951 ± 0.010 |
| 5 positives   |   10.000 | 1.000 ± 0.000 | 0.945 ± 0.013 |
| 5 positives   |  100.000 | 1.000 ± 0.000 | 0.945 ± 0.013 |
| 5 positives   | 1000.000 | 1.000 ± 0.000 | 0.945 ± 0.013 |
| all positives |    0.100 | 0.987         | 0.985         |
| all positives |    1.000 | 0.993         | 0.989         |
| all positives |   10.000 | 0.999         | 0.980         |
| all positives |  100.000 | 1.000         | 0.972         |
| all positives | 1000.000 | 1.000         | 0.972         |

![part4_svm_c](part4_svm_c.png)

Selected C per regime (highest validation mAP), test AP (mean over draws):

| regime        |     C |   aeroplane |   car |   horse |   motorbike |   person |   mAP |
|:--------------|------:|------------:|------:|--------:|------------:|---------:|------:|
| 5 positives   | 0.100 |       0.992 | 0.886 |   0.985 |       0.975 |    0.910 | 0.949 |
| all positives | 1.000 |       1.000 | 0.974 |   0.995 |       0.983 |    0.982 | 0.987 |

## Stage E — CLIP zero-shot classification

> CLIP ViT-B/32 with its own preprocessing; class prompts encoded by the text encoder; image–text cosine similarity. Retrieval: rank by similarity to the class prompt → AP / mAP. Classification: most similar prompt → confusion matrix and per-class metrics. One prompt vs. prompt sets, chosen on validation only.

Validation AP per prompt set. Only the best set is evaluated on test:

| prompt_set   |   prompts_per_class |   motorbike |   aeroplane |   person |   horse |   car |   mAP_val |
|:-------------|--------------------:|------------:|------------:|---------:|--------:|------:|----------:|
| name         |                   1 |       0.946 |       0.981 |    0.596 |   0.949 | 0.791 |     0.853 |
| photo        |                   1 |       0.916 |       1.000 |    0.656 |   0.885 | 0.827 |     0.857 |
| templates    |                   8 |       0.939 |       1.000 |    0.615 |   0.892 | 0.834 |     0.856 |
| synonyms     |                   2 |       0.921 |       1.000 |    0.610 |   0.884 | 0.803 |     0.844 |

Test AP: zero-shot with the selected prompts vs. the linear SVM (L2, C = 10) on the same CLIP image embeddings:

| category   |   zero_shot (photo) |   svm_on_clip |
|:-----------|--------------------:|--------------:|
| motorbike  |               0.889 |         0.994 |
| aeroplane  |               0.982 |         0.998 |
| person     |               0.583 |         0.991 |
| horse      |               0.811 |         0.992 |
| car        |               0.865 |         0.979 |
| mAP        |               0.826 |         0.991 |

Zero-shot classification of the single-category test images (most similar of the 5 class prompts and a background prompt; rows: true, columns: predicted):

| true       |   motorbike |   aeroplane |   person |   horse |   car |   background |
|:-----------|------------:|------------:|---------:|--------:|------:|-------------:|
| motorbike  |          36 |           0 |        0 |       1 |     0 |            0 |
| aeroplane  |           0 |         112 |        0 |       0 |     0 |            0 |
| person     |          26 |          11 |      384 |      14 |    10 |          239 |
| horse      |           0 |           0 |        0 |      30 |     0 |            1 |
| car        |           3 |           1 |        0 |       0 |   165 |           59 |
| background |          31 |          42 |      288 |     106 |    32 |          578 |

| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| motorbike    |       0.375 |    0.973 |      0.541 |    37.000 |
| aeroplane    |       0.675 |    1.000 |      0.806 |   112.000 |
| person       |       0.571 |    0.561 |      0.566 |   684.000 |
| horse        |       0.199 |    0.968 |      0.330 |    31.000 |
| car          |       0.797 |    0.724 |      0.759 |   228.000 |
| background   |       0.659 |    0.537 |      0.592 |  1077.000 |
| accuracy     |       0.602 |    0.602 |      0.602 |     0.602 |
| macro avg    |       0.546 |    0.794 |      0.599 |  2169.000 |
| weighted avg |       0.635 |    0.602 |      0.608 |  2169.000 |

Costs:

| encoding      |   dim |   kb_per_image |   extract_ms_per_image | device   | prompt_set   |   num_prompts |   text_encode_s |   infer_us_per_image |
|:--------------|------:|---------------:|-----------------------:|:---------|:-------------|--------------:|----------------:|---------------------:|
| clip_vit_b_32 |   512 |         2.0000 |                35.1819 | cpu      | photo        |             6 |          0.2174 |               2.9475 |

## Stage F — visual analysis and failure cases

> Correct and incorrect test examples of the CNN, DINOv2 and CLIP experiments, with the image regions that support the class where the method allows it.

Per category: correctly detected positives (TP, highest scores), background images wrongly detected (FP, highest scores) and missed positives (FN, lowest scores). Only real errors: SVM score above / below 0; for CLIP, the 6-way prompt argmax of Stage E. A category with fewer errors shows fewer images.

Models: ResNet-18 layer4 and DINOv2 `patchmean`, both L2 + SVM with C = 10 (the settings selected on validation). Their maps split the SVM score over image positions (positive part shown); the DINOv2 CLS token has no such split, so `patchmean` is used. CLIP: examples only. Image ids and scores are in `part4_examples.csv`.

![part4_examples_clip](part4_examples_clip.png)

![part4_examples_dinov2](part4_examples_dinov2.png)

![part4_examples_resnet18](part4_examples_resnet18.png)

## Required comparison table

> One row per method. Within a method, the representation (layer or token, feature map) with the highest validation mAP. Every SVM uses C = 10, fixed in advance as in Part 1 (Stage D studies C separately). Extraction: deep models = forward pass on CPU (see the protocol above); handcrafted = dense SIFT + encoding on CPU, timed on 5 images. Inference: feature map + classifier (or text similarity) per image, features already extracted.

| method                | representation                 | supervision                            | C   |   test mAP |   dim |   extract_ms_per_image |   infer_us_per_image |
|:----------------------|:-------------------------------|:---------------------------------------|:----|-----------:|------:|-----------------------:|---------------------:|
| BoVW + SVM            | bovw_K512_C0 (sqrt_l2)         | Task labels                            | 10  |      0.694 |  2048 |                239.594 |               19.460 |
| VLAD + SVM            | vlad_K32_C80 (ssr_intra_l2)    | Task labels                            | 10  |      0.746 | 10240 |                207.029 |              227.124 |
| Fisher Vector + SVM   | fv_K32_C80 (ssr_l2)            | Task labels                            | 10  |      0.784 | 20608 |                295.513 |              360.781 |
| CNN (ResNet-18) + SVM | resnet18_layer4 (l2)           | Pretrained (ImageNet) + task labels    | 10  |      0.964 |   512 |                 18.257 |                3.133 |
| ViT-B/32 + SVM        | vit_b_32 (l2)                  | Pretrained (ImageNet) + task labels    | 10  |      0.961 |   768 |                 35.005 |                5.757 |
| DINOv2 + SVM          | dinov2_vits14_patchmean (l2)   | Self-supervised + task labels          | 10  |      0.984 |   384 |                 52.278 |                2.242 |
| CLIP zero-shot        | clip_vit_b_32, prompts (photo) | Image–text pretraining, no task labels | –   |      0.826 |   512 |                 35.182 |                2.948 |

### Ranking metrics of the same methods

Test set, mean over the 5 categories. Positive class: the category; negatives: background. AP = scikit-learn `average_precision_score` (sum over the ranking of precision × recall step, no interpolation). P@K / R@K: precision / recall of the top K images. `precision_thr0` / `recall_thr0`: classification at the SVM threshold 0 (none for zero-shot, which has no threshold). Per-category values are in `part4_ranking.csv`.

| method                |    AP |   P@10 |   P@50 |   P@100 |   R@10 |   R@50 |   R@100 | precision_thr0   | recall_thr0   |
|:----------------------|------:|-------:|-------:|--------:|-------:|-------:|--------:|:-----------------|:--------------|
| BoVW + SVM            | 0.694 |  0.920 |  0.852 |   0.788 |  0.051 |  0.223 |   0.397 | 0.704            | 0.579         |
| VLAD + SVM            | 0.746 |  1.000 |  0.948 |   0.824 |  0.053 |  0.249 |   0.414 | 0.806            | 0.543         |
| Fisher Vector + SVM   | 0.784 |  1.000 |  0.960 |   0.840 |  0.053 |  0.256 |   0.426 | 0.833            | 0.577         |
| CNN (ResNet-18) + SVM | 0.964 |  1.000 |  1.000 |   0.990 |  0.053 |  0.267 |   0.526 | 0.937            | 0.876         |
| ViT-B/32 + SVM        | 0.961 |  1.000 |  1.000 |   0.996 |  0.053 |  0.267 |   0.531 | 0.930            | 0.870         |
| DINOv2 + SVM          | 0.984 |  1.000 |  1.000 |   0.998 |  0.053 |  0.267 |   0.532 | 0.964            | 0.937         |
| CLIP zero-shot        | 0.826 |  1.000 |  0.944 |   0.900 |  0.053 |  0.252 |   0.481 | –                | –             |

![part4_pr_curves](part4_pr_curves.png)

## Limitations and deviations from the protocol

- AP compares each category only with background images, so it does not measure confusions between categories; the confusion matrices (Stages B and E) do.
- Images with several categories (e.g. horse + person) are left out of the confusion matrices, which need one true class per image.
- CLIP zero-shot classification uses 6 prompts (5 categories + background) because the test set contains background images; the assignment mentions the 5 categories. The 5-way accuracy on object images is printed by `run_part4_clip`.
- Extraction times of deep and handcrafted methods are measured differently (all images incl. model loading vs. 5 images) and are approximate.
- All runs on CPU, one machine; see the protocol for hardware and versions.

