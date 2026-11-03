# Part 2 — Advanced encodings: VLAD and Fisher Vectors

Same linear SVM and splits as Part 1. Values are test AP / mAP.

## Normalization and spatial tiling per encoding

`ssr` = signed square root, `intra` = per-word L2 (VLAD).

| encoding     | spatial   | feature_map   |   mAP |
|:-------------|:----------|:--------------|------:|
| bovw_K512_C0 | 2x2 tiles | l2            | 0.593 |
| bovw_K512_C0 | 2x2 tiles | sqrt_l2       | 0.694 |
| bovw_K512_C0 | global    | l2            | 0.585 |
| bovw_K512_C0 | global    | sqrt_l2       | 0.669 |
| fv_K16_C80   | 2x2 tiles | none          | 0.685 |
| fv_K16_C80   | 2x2 tiles | l2            | 0.705 |
| fv_K16_C80   | 2x2 tiles | ssr_l2        | 0.775 |
| fv_K16_C80   | global    | none          | 0.643 |
| fv_K16_C80   | global    | l2            | 0.694 |
| fv_K16_C80   | global    | ssr_l2        | 0.733 |
| fv_K32_C80   | 2x2 tiles | none          | 0.690 |
| fv_K32_C80   | 2x2 tiles | l2            | 0.721 |
| fv_K32_C80   | 2x2 tiles | ssr_l2        | 0.784 |
| fv_K32_C80   | global    | none          | 0.638 |
| fv_K32_C80   | global    | l2            | 0.708 |
| fv_K32_C80   | global    | ssr_l2        | 0.740 |
| fv_K64_C80   | 2x2 tiles | none          | 0.704 |
| fv_K64_C80   | 2x2 tiles | l2            | 0.728 |
| fv_K64_C80   | 2x2 tiles | ssr_l2        | 0.803 |
| fv_K64_C80   | global    | none          | 0.669 |
| fv_K64_C80   | global    | l2            | 0.723 |
| fv_K64_C80   | global    | ssr_l2        | 0.780 |
| vlad_K16_C80 | 2x2 tiles | none          | 0.649 |
| vlad_K16_C80 | 2x2 tiles | l2            | 0.662 |
| vlad_K16_C80 | 2x2 tiles | ssr_l2        | 0.714 |
| vlad_K16_C80 | 2x2 tiles | intra_l2      | 0.675 |
| vlad_K16_C80 | 2x2 tiles | ssr_intra_l2  | 0.702 |
| vlad_K16_C80 | global    | none          | 0.669 |
| vlad_K16_C80 | global    | l2            | 0.672 |
| vlad_K16_C80 | global    | ssr_l2        | 0.661 |
| vlad_K16_C80 | global    | intra_l2      | 0.667 |
| vlad_K16_C80 | global    | ssr_intra_l2  | 0.653 |
| vlad_K32_C80 | 2x2 tiles | none          | 0.674 |
| vlad_K32_C80 | 2x2 tiles | l2            | 0.675 |
| vlad_K32_C80 | 2x2 tiles | ssr_l2        | 0.740 |
| vlad_K32_C80 | 2x2 tiles | intra_l2      | 0.719 |
| vlad_K32_C80 | 2x2 tiles | ssr_intra_l2  | 0.746 |
| vlad_K32_C80 | global    | none          | 0.675 |
| vlad_K32_C80 | global    | l2            | 0.668 |
| vlad_K32_C80 | global    | ssr_l2        | 0.695 |
| vlad_K32_C80 | global    | intra_l2      | 0.673 |
| vlad_K32_C80 | global    | ssr_intra_l2  | 0.681 |
| vlad_K64_C80 | 2x2 tiles | none          | 0.708 |
| vlad_K64_C80 | 2x2 tiles | l2            | 0.709 |
| vlad_K64_C80 | 2x2 tiles | ssr_l2        | 0.772 |
| vlad_K64_C80 | 2x2 tiles | intra_l2      | 0.747 |
| vlad_K64_C80 | 2x2 tiles | ssr_intra_l2  | 0.766 |
| vlad_K64_C80 | global    | none          | 0.702 |
| vlad_K64_C80 | global    | l2            | 0.692 |
| vlad_K64_C80 | global    | ssr_l2        | 0.725 |
| vlad_K64_C80 | global    | intra_l2      | 0.695 |
| vlad_K64_C80 | global    | ssr_intra_l2  | 0.712 |

## Selected setting per encoding (on mean validation AP): per-class test AP

| encoding     | spatial   | feature_map   |   motorbike |   aeroplane |   person |   horse |   car |   mAP |
|:-------------|:----------|:--------------|------------:|------------:|---------:|--------:|------:|------:|
| bovw_K512_C0 | 2x2 tiles | sqrt_l2       |       0.632 |       0.691 |    0.750 |   0.662 | 0.734 | 0.694 |
| fv_K16_C80   | 2x2 tiles | ssr_l2        |       0.776 |       0.729 |    0.800 |   0.769 | 0.798 | 0.775 |
| fv_K32_C80   | 2x2 tiles | ssr_l2        |       0.783 |       0.759 |    0.811 |   0.758 | 0.807 | 0.784 |
| fv_K64_C80   | 2x2 tiles | ssr_l2        |       0.810 |       0.788 |    0.820 |   0.770 | 0.828 | 0.803 |
| vlad_K16_C80 | 2x2 tiles | ssr_l2        |       0.680 |       0.694 |    0.761 |   0.670 | 0.764 | 0.714 |
| vlad_K32_C80 | 2x2 tiles | ssr_intra_l2  |       0.707 |       0.726 |    0.768 |   0.735 | 0.793 | 0.746 |
| vlad_K64_C80 | 2x2 tiles | ssr_intra_l2  |       0.745 |       0.765 |    0.798 |   0.717 | 0.805 | 0.766 |

## Dimension, storage and cost

`codebook_fit_s`: PCA + k-means/GMM (NaN for encoders trained before it was recorded). SIFT and encoding times are per image, on a few sample images.

| encoding     |   dim | dtype   |   kb_per_image |   disk_mb |   codebook_fit_s |   sift_ms_per_image |   encode_ms_per_image |   train_time_s |   infer_ms_per_image |   mAP |
|:-------------|------:|:--------|---------------:|----------:|-----------------:|--------------------:|----------------------:|---------------:|---------------------:|------:|
| bovw_K512_C0 |  2048 | int64   |         16.000 |    11.960 |           63.996 |             158.309 |               102.960 |          0.145 |                0.019 | 0.694 |
| fv_K16_C80   | 10304 | float64 |         80.500 |   413.949 |           14.686 |             158.309 |                99.488 |          1.095 |                0.197 | 0.775 |
| fv_K32_C80   | 20608 | float64 |        161.000 |   837.738 |           36.302 |             158.309 |               127.548 |          2.109 |                0.391 | 0.784 |
| fv_K64_C80   | 41216 | float64 |        322.000 |  1688.463 |           55.741 |             158.309 |               178.654 |          3.122 |                0.756 | 0.803 |
| vlad_K16_C80 |  5120 | float32 |         20.000 |   100.933 |            1.888 |             158.309 |                63.511 |          0.449 |                0.089 | 0.714 |
| vlad_K32_C80 | 10240 | float32 |         40.000 |   199.480 |            5.506 |             158.309 |                72.918 |          0.706 |                0.225 | 0.746 |
| vlad_K64_C80 | 20480 | float32 |         80.000 |   384.221 |            8.098 |             158.309 |                78.211 |          1.051 |                0.442 | 0.766 |

## Questions

**Stage H**

- For K visual words and D-dimensional local descriptors, compare the nominal dimensions of BoVW (K) and VLAD (K x D). What changes in representation capacity and storage?
  - *Answer:*

**Stage I**

- What additional information does a Fisher Vector encode compared with VLAD, and what computational costs does it introduce?
  - *Answer:*

