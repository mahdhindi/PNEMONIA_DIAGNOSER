# Results summary

## Main comparison (test split)

| Rank | Model | AUROC | AUROC 95% CI (seed 1) | Accuracy | Macro-F1 | Recall (pneu.) | Specificity | Params (M) | Ckpt (MB) | Train time (min) | Infer (ms/img) | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-50 (ImageNet, fine-tuned) | 0.9717 | [0.956, 0.984] | 0.8382 | 0.8053 | 0.9974 | 0.5714 | 23.51 | 94.3 | 4.4 | 1.87 | 1 |
| 2 | Small CNN (scratch) | 0.9458 | [0.924, 0.963] | 0.8285 | 0.7916 | 0.9974 | 0.5455 | 0.39 | 1.6 | 3.4 | 1.55 | 1 |
| 3 | DenseNet-121 (ImageNet, fine-tuned) | 0.9233 | [0.893, 0.948] | 0.8026 | 0.7525 | 1.0000 | 0.4719 | 6.96 | 28.4 | 5.9 | 2.28 | 1 |
| 4 | Logistic regression (pixels) | 0.8989 | [0.872, 0.922] | 0.8042 | 0.7601 | 0.9845 | 0.5022 | 0.15 | 0.6 | 3.0 | 1.25 | 1 |
| 5 | Majority class | 0.5000 | [0.500, 0.500] | 0.6262 | 0.3851 | 1.0000 | 0.0000 | 0.00 | 0.0 | 0.0 | 1.27 | 1 |

![](figures/main_auroc.png)

## Ablation A - fixed pixel permutation

Prediction: the linear model is unaffected (permutation-invariant); models that won by exploiting local spatial texture lose their advantage and the ranking changes.

| Model | auroc_normal | auroc_permuted | delta | rank_normal | rank_permuted | acc_normal | acc_permuted |
|---|---|---|---|---|---|---|---|
| Logistic regression (pixels) | 0.8989 | 0.8987 | -0.0002 | 4 | 3 | 0.8042 | 0.8026 |
| Small CNN (scratch) | 0.9458 | 0.8562 | -0.0896 | 2 | 4 | 0.8285 | 0.7492 |
| ResNet-50 (ImageNet, fine-tuned) | 0.9717 | 0.9002 | -0.0715 | 1 | 2 | 0.8382 | 0.7621 |
| DenseNet-121 (ImageNet, fine-tuned) | 0.9233 | 0.9018 | -0.0215 | 3 | 1 | 0.8026 | 0.7476 |

![](figures/ablation_permute.png)

## Ablation B - training-set size

Training images per fraction: {'10% train': 445, '25% train': 1108, '100% train': 4424}. Prediction: the gap between ImageNet-pretrained models and models learned from scratch widens as data shrinks.

| model | 10% train | 25% train | 100% train | 10% train rank | 25% train rank | 100% train rank |
|---|---|---|---|---|---|---|
| Majority class |  |  | 0.5000 |  |  | 5 |
| Logistic regression (pixels) | 0.8518 | 0.9081 | 0.8989 | 4 | 3 | 4 |
| Small CNN (scratch) | 0.8727 | 0.9019 | 0.9458 | 2 | 4 | 2 |
| ResNet-50 (ImageNet, fine-tuned) | 0.8712 | 0.9756 | 0.9717 | 3 | 1 | 1 |
| DenseNet-121 (ImageNet, fine-tuned) | 0.9455 | 0.9571 | 0.9233 | 1 | 2 | 3 |

![](figures/ablation_fraction.png)

## Correct-classification rate by image subtype (test split)

Pneumonia is one label covering bacterial and viral cases; viral pneumonia is typically more diffuse and harder to see.

| model | bacteria (n=240) | normal (n=231) | virus (n=147) |
|---|---|---|---|
| Majority class | 1.0000 | 0.0000 | 1.0000 |
| Logistic regression (pixels) | 1.0000 | 0.5022 | 0.9592 |
| Small CNN (scratch) | 0.9958 | 0.5455 | 1.0000 |
| ResNet-50 (ImageNet, fine-tuned) | 0.9958 | 0.5714 | 1.0000 |
| DenseNet-121 (ImageNet, fine-tuned) | 1.0000 | 0.4719 | 1.0000 |


# Data audit

Prepared 2026-09-25T12:30:37+00:00 from `/content/repo/data/raw` (strategy: `original_test`).

## Size

- Files found: **11712** ({'PNEUMONIA': 8546, 'NORMAL': 3166})
- Original folders x class: {'NORMAL': {'test': 468, 'train': 2682, 'val': 16}, 'PNEUMONIA': {'test': 780, 'train': 7750, 'val': 16}}
- Corrupt: 0; exact-duplicate redundant files: 94
- Patients (grouping keys): 3288 ({'NORMAL': 1443, 'PNEUMONIA': 1845}); 1155 have >1 image (max 30)
- Resolution: width {'min': 384, 'median': 1281, 'max': 2916}, height {'min': 127, 'median': 888, 'max': 2713}, aspect {'min': 0.835, 'median': 1.416, 'max': 3.379}, 4803 distinct sizes; modes {'L': 11146, 'RGB': 566}
- Features: model input 224x224x3 = 150528 values (grayscale replicated to 3 channels); median raw image ~1,137,528 pixels

## Final splits (patient-grouped, seed-fixed)

- {'NORMAL': {'test': 231, 'train': 1145, 'val': 203}, 'PNEUMONIA': {'test': 387, 'train': 3279, 'val': 579}}
- Subtypes: {'bacteria': {'test': 240, 'train': 2145, 'val': 375}, 'normal': {'test': 231, 'train': 1145, 'val': 203}, 'unknown': {'test': 0, 'train': 0, 'val': 1}, 'virus': {'test': 147, 'train': 1134, 'val': 203}}
- Leakage checks after split: {'patient_keys_shared_train_val': 0, 'patient_keys_shared_train_test': 0, 'near_dup_pairs_train_test': 0, 'near_dup_pairs_train_val': 0}

## Problems found

1. 566 files are stored as 3-channel RGB although X-rays are grayscale; all images were converted to single-channel.
2. Image resolution is not standardised: 4803 distinct sizes, width 384-2916 px, aspect ratio 0.835-3.379. We pad to square before resizing so anatomy is not stretched.
3. 5794 files are byte-identical copies of a file with the same name (archive packaging, e.g. a nested folder); one copy is kept.
4. 94 byte-identical duplicate images stored under different names, in 30 groups (0 groups span two original splits, 0 carry conflicting labels). One copy is kept; conflicting-label groups are dropped entirely.
5. 1155 patients contribute more than one image (3691 images, up to 30 per patient). Splitting by image instead of by patient puts the same child on both sides of a split (leakage).
6. Class imbalance: 72.9% of images are PNEUMONIA. Accuracy alone is misleading; a majority-class predictor already scores that number.
7. Class prior shifts between the original train (74.2% pneumonia) and test (62.6% pneumonia) folders: the test folder is a separate collection, not an i.i.d. sample of the training distribution.
8. The dataset ships a validation folder of only 16 images - useless for model selection. We merge it into the training pool and carve a patient-grouped validation set of our own.
9. Simulated naive by-image random split (15% val): 479 of 781 validation images (61.3%) would share a patient with the training set. Our split assigns whole patients.

## Near-duplicate calibration

Nearest-neighbour pHash distance percentiles: {'p1': 44, 'p5': 48, 'p10': 52, 'p25': 56, 'p50': 62, 'p75': 68, 'p90': 74} (threshold 10 bits of 256).
