# Data audit

Prepared 2026-09-25T15:02:17+00:00 from `/content/repo/data/raw` (strategy: `original_test`).

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
10. Acquisition geometry leaks the label: image height alone separates the classes with AUROC 0.916 on train / 0.921 on test (median W x H, aspect: NORMAL (1636, 1318, 1.222), PNEUMONIA (1168, 784, 1.484) in train). Resizing removes absolute size; with pad-to-square the aspect ratio survives as padding (AUROC 0.865 train / 0.703 test), a shortcut that transfers poorly to the test folder.

## Near-duplicate calibration

Nearest-neighbour pHash distance percentiles: {'p1': 50, 'p5': 56, 'p10': 60, 'p25': 66, 'p50': 72, 'p75': 78, 'p90': 84} (threshold 10 bits of 256).
