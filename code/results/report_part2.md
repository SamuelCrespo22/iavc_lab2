# Part 2 — Advanced encodings: VLAD and Fisher Vectors

## Stage H — VLAD

> Replace BoVW with VLAD (same descriptors and splits), normalization selected on validation, repeat the classification and retrieval experiments, note the gain over BoVW.

Validation mAP per layout and normalization:

| encoding     |   2x2 tiles, none |   2x2 tiles, l2 |   2x2 tiles, ssr_l2 |   2x2 tiles, intra_l2 |   2x2 tiles, ssr_intra_l2 |   global, none |   global, l2 |   global, ssr_l2 |   global, intra_l2 |   global, ssr_intra_l2 | selected                |
|:-------------|------------------:|----------------:|--------------------:|----------------------:|--------------------------:|---------------:|-------------:|-----------------:|-------------------:|-----------------------:|:------------------------|
| vlad_K16_C80 |             0.694 |           0.707 |               0.740 |                 0.716 |                     0.718 |          0.628 |        0.652 |            0.660 |              0.664 |                  0.659 | 2x2 tiles, ssr_l2       |
| vlad_K32_C80 |             0.715 |           0.715 |               0.790 |                 0.778 |                     0.793 |          0.616 |        0.655 |            0.664 |              0.613 |                  0.649 | 2x2 tiles, ssr_intra_l2 |
| vlad_K64_C80 |             0.752 |           0.751 |               0.812 |                 0.801 |                     0.822 |          0.702 |        0.708 |            0.710 |              0.690 |                  0.723 | 2x2 tiles, ssr_intra_l2 |

Test AP of the selected setting:

| encoding     |   motorbike |   aeroplane |   person |   horse |   car |   mAP |   gain vs. BoVW |   dim |   kb_per_image |
|:-------------|------------:|------------:|---------:|--------:|------:|------:|----------------:|------:|---------------:|
| bovw_K512_C0 |       0.632 |       0.691 |    0.750 |   0.662 | 0.734 | 0.694 |           0.000 |  2048 |         16.000 |
| vlad_K16_C80 |       0.680 |       0.694 |    0.761 |   0.670 | 0.764 | 0.714 |           0.020 |  5120 |         20.000 |
| vlad_K32_C80 |       0.707 |       0.726 |    0.768 |   0.735 | 0.793 | 0.746 |           0.052 | 10240 |         40.000 |
| vlad_K64_C80 |       0.745 |       0.765 |    0.798 |   0.717 | 0.805 | 0.766 |           0.072 | 20480 |         80.000 |

## Stage I — Fisher Vector

> GMM fitted on training descriptors only; Fisher Vectors with the same classifier and splits; compare FV, VLAD and BoVW: AP / mAP, feature dimension, extraction time and storage.

Validation mAP per layout and normalization:

| encoding   |   2x2 tiles, none |   2x2 tiles, l2 |   2x2 tiles, ssr_l2 |   global, none |   global, l2 |   global, ssr_l2 | selected          |
|:-----------|------------------:|----------------:|--------------------:|---------------:|-------------:|-----------------:|:------------------|
| fv_K16_C80 |             0.707 |           0.701 |               0.763 |          0.659 |        0.684 |            0.750 | 2x2 tiles, ssr_l2 |
| fv_K32_C80 |             0.737 |           0.710 |               0.784 |          0.727 |        0.695 |            0.748 | 2x2 tiles, ssr_l2 |
| fv_K64_C80 |             0.742 |           0.720 |               0.796 |          0.734 |        0.726 |            0.782 | 2x2 tiles, ssr_l2 |

Test AP of the selected setting:

| encoding   |   motorbike |   aeroplane |   person |   horse |   car |   mAP |
|:-----------|------------:|------------:|---------:|--------:|------:|------:|
| fv_K16_C80 |       0.776 |       0.729 |    0.800 |   0.769 | 0.798 | 0.775 |
| fv_K32_C80 |       0.783 |       0.759 |    0.811 |   0.758 | 0.807 | 0.784 |
| fv_K64_C80 |       0.810 |       0.788 |    0.820 |   0.770 | 0.828 | 0.803 |

Comparison of the three encodings (selected settings). Codebook fit = PCA + k-means / GMM, once; SIFT and encoding time per image on 5 test images (CPU); storage as stored on disk; SVM training and inference (per test image) averaged over the categories:

| encoding     |   mAP |   dim | dtype   |   kb_per_image |   disk_mb |   codebook_fit_s |   sift_ms_per_image |   encode_ms_per_image |   svm_train_s |   infer_us |
|:-------------|------:|------:|:--------|---------------:|----------:|-----------------:|--------------------:|----------------------:|--------------:|-----------:|
| bovw_K512_C0 | 0.694 |  2048 | int64   |         16.000 |    11.960 |           63.996 |             152.118 |               101.578 |         0.160 |     19.968 |
| vlad_K16_C80 | 0.714 |  5120 | float32 |         20.000 |   100.933 |            1.888 |             152.118 |                64.233 |         0.497 |     91.059 |
| vlad_K32_C80 | 0.746 | 10240 | float32 |         40.000 |   199.480 |            5.506 |             152.118 |                72.144 |         0.727 |    230.016 |
| vlad_K64_C80 | 0.766 | 20480 | float32 |         80.000 |   384.221 |            8.098 |             152.118 |                77.609 |         1.054 |    457.766 |
| fv_K16_C80   | 0.775 | 10304 | float64 |         80.500 |   413.949 |           14.686 |             152.118 |                98.920 |         1.154 |    205.103 |
| fv_K32_C80   | 0.784 | 20608 | float64 |        161.000 |   837.738 |           36.302 |             152.118 |               126.003 |         2.058 |    388.536 |
| fv_K64_C80   | 0.803 | 41216 | float64 |        322.000 |  1688.463 |           55.741 |             152.118 |               174.649 |         3.281 |    807.297 |

