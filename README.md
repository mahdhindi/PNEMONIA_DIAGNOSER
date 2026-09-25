# Pneumonia detection from pediatric chest X-rays — a fair comparison of five classifiers

**CSBP711 Advanced Artificial Intelligence, United Arab Emirates University, Fall 2026 — Assignment 1: Datasets and Algorithm Comparison**

Group: **Mahd Hindi, Khaled AlHassani, Abdullah AlKaabi** — College of Informatics & AI

Slides: `slides/CSBP711_A1_slides.pdf` · Repository: https://github.com/mahdhindi/PNEMONIA_DIAGNOSER

We take the most widely used public pediatric chest X-ray dataset, audit it properly, split it without leakage,
compare five classifiers under one identical protocol, explain why the winner wins in terms of a property of the
data, and test that explanation with an ablation. Every number in this README and in the slides is produced by the
code in this repository from a fresh download; nothing is copied from a paper.

---

## 1. Dataset

| | |
|---|---|
| **Name** | Chest X-Ray Images (Pneumonia) — pediatric, two classes (NORMAL / PNEUMONIA) |
| **Upstream source** | Kermany, Zhang & Goldbaum (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data **v2**, https://doi.org/10.17632/rscbjbr9sj.2 |
| **Mirror used for download** | Kaggle — https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| **Licence** | **CC BY 4.0** (research use permitted, attribution required) |
| **Version / download date** | Kaggle mirror, downloaded **2026-09-25** (5,863 files: 5,216 train / 16 val / 624 test; recorded automatically in `data/raw/DOWNLOAD_INFO.json`). After removing 5,794 archive packaging copies and 94 hidden duplicates: **5,824 unique images**. |
| **How it was collected** | Anterior–posterior chest radiographs of children aged 1–5 from Guangzhou Women and Children's Medical Center, taken during routine care; low-quality scans were removed; labels were graded by two physicians, with the evaluation set checked by a third (Kermany et al., *Cell* 172(5):1122–1131, 2018, https://doi.org/10.1016/j.cell.2018.02.010). |
| **Why this dataset** | It is the most cited pediatric pneumonia CXR benchmark, it is large enough that a linear baseline does not solve it on the held-out test folder, and it has documented but rarely audited data problems (patient leakage, duplicates, an acquisition shortcut) that make it a demanding test of data handling and fair comparison. _[Add one sentence on the link to the group's research or work.]_ |

Everything the audit found is written to `data/audit.md` by `prepare_data.py` and summarised in §3.

## 2. Reproduce everything

```bash
git clone https://github.com/mahdhindi/PNEMONIA_DIAGNOSER.git && cd PNEMONIA_DIAGNOSER
pip install -r requirements.txt            # PyTorch >= 2.2 with CUDA recommended (a Colab T4 is enough)

# 1) data — needs a Kaggle API token (KAGGLE_API_TOKEN env var, or ~/.kaggle/kaggle.json)
python scripts/download_data.py kaggle     # alternatives: `zip --path file.zip` (Mendeley archive) or `roboflow --version N --api_key ..`

# 2) audit + patient-grouped split + tensor cache, 3) five models, 4) ablations, 5) tables & figures
bash scripts/run_all.sh                                    # seed 42, main table + both ablations, ~1 h on a T4
SEEDS="43" FRACTIONS="" bash scripts/run_all.sh            # a second seed for mean ± std (~35 min)
RESIZE=stretch FRACTIONS="" ABLATIONS="" bash scripts/run_all.sh   # the padding-cue check (~25 min)
```

Outputs: `data/audit.md`, `data/index.csv` (the split), `results/*.json` (one per run, including per-image
probabilities), `results/main_table.md`, `results/ablation_*.md`, `results/check_stretch.csv`,
`results/figures/*.png`, `results/summary.md`. `colab_run.ipynb` runs exactly these commands on Google Colab.
No hard-coded local paths: every script takes `--data` / `--results` arguments. An interrupted run resumes; finished
runs are skipped.

Environment of the reported runs: Google Colab, Tesla T4, PyTorch 2.11.0+cu128, Python 3.13, mixed precision.

Offline end-to-end check without the real data (synthetic images, CPU, about one minute):

```bash
bash tests/smoke_test.sh
```

## 3. Data handling — what the audit found

Size and features: 5,824 unique images (1,583 NORMAL, 4,241 PNEUMONIA); median raw scan 1281×888 px (about
1.14 megapixels), 4,803 distinct resolutions; model input 224×224×3 = 150,528 values (grayscale replicated to three
channels). Splits: **train 4,424 / validation 782 / test 618**, assigned by patient; the test split is the dataset's own
test folder. Class balance: 72.9% pneumonia overall, 74.2% in the training folder, 62.6% in the test folder.

| Problem found | What we did |
|---|---|
| **Patient leakage.** 1,155 patients contribute more than one scan (up to 30). A simulated naive by-image random split shares a patient between training and validation for **479 of 781 (61.3%)** validation images. | Whole patients assigned to one split (union-find over patient ids and near-duplicate pairs). Patients shared across splits after: 0. |
| **Hidden duplicates.** 94 byte-identical images stored under different file names (30 groups); a further 5,794 files were byte-identical copies of the same file names (a nested folder in the archive). | One copy kept; no duplicate group crossed a split or carried conflicting labels. |
| **Unusable validation folder.** The dataset ships a validation folder of 16 images. | Merged into the training pool; a 782-image validation set carved by patient for model selection. |
| **Collection shift.** The test folder is a separate collection: 62.6% pneumonia vs 74.2% in training, and its normal scans are wider (median aspect 1.35 vs 1.22). | Kept as the held-out test set so results are comparable with the literature; reported and explained rather than re-split away. |
| **Geometry shortcut.** Pneumonia scans are systematically smaller and wider (median 1168×776 px, aspect 1.49) than normal scans (1640×1328 px, aspect 1.22). Image height alone separates the classes with AUROC 0.92 on train and test. Resizing removes absolute size, but pad-to-square keeps the aspect ratio visible as black bars (AUROC 0.87 on train, 0.70 on test). | Flagged as a shortcut and tested with a second preprocessing (§6, extra check). |
| **Encoding noise.** 566 files are stored as RGB although X-rays are grayscale; 0 corrupt files; 0 near-duplicates at pHash distance ≤ 10/256 (nearest-neighbour distance is ≥ 44 for every image). | All images converted to one channel; one preprocessing pipeline for every model. |

## 4. Comparison protocol — why the table is fair

* **Same data**: identical train/validation/test split (`data/index.csv`), fixed with seed 42, grouped by patient.
* **Same preprocessing**: grayscale → pad to square → 224×224 → ImageNet normalisation; the same light augmentation
  (random resized crop 0.8–1, ±7° rotation, ±15% brightness/contrast) for every trainable model. The logistic
  regression sees exactly the tensors the CNNs see; it only flattens them.
* **Same seeds**: `set_seed()` fixes Python, NumPy, PyTorch and DataLoader RNGs; `cudnn.deterministic` and
  `use_deterministic_algorithms(warn_only=True)` are on. Two CPU runs with the same seed are bit-identical; on a GPU a
  few kernels are non-deterministic, so we also report ± std over seeds.
* **Same optimiser and budget**: AdamW, weight decay 1e-4, one warm-up epoch then cosine decay, batch 32, 12 epochs,
  checkpoint selected by validation AUROC. The only per-model difference is the learning rate (from-scratch CNN
  1e-3, linear 1e-4, fine-tuned pretrained 1e-4), recorded in every results file.
* **Metrics**: AUROC (primary — threshold-free and robust to the 73/27 imbalance), accuracy, balanced accuracy,
  macro-F1, sensitivity, specificity, with 95% bootstrap confidence intervals over the test set; parameter count,
  checkpoint size, wall-clock training time, inference ms/image.

Models: majority class (reference), logistic regression on pixels (the simple baseline), a small CNN trained from
scratch (4 conv blocks), ResNet-50 and DenseNet-121 (both ImageNet-pretrained, fully fine-tuned).

## 5. Results

Test split = the dataset's own test folder, 618 images; mean ± std over seeds 42 and 43.

| Rank | Model | AUROC | Accuracy | Balanced acc. | Sensitivity | Specificity | Params | Checkpoint | Train (T4) | Inference |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-50, ImageNet, fine-tuned | **0.971 ± 0.002** | 0.863 ± 0.034 | 0.817 | 0.997 | 0.64 | 23.5 M | 94 MB | 4.4 min | 1.9 ms/img |
| 2 | Small CNN, from scratch | 0.942 ± 0.006 | 0.826 ± 0.003 | 0.769 | 0.996 | 0.54 | 0.39 M | 1.6 MB | 3.4 min | 1.5 ms/img |
| 3 | DenseNet-121, ImageNet, fine-tuned | 0.936 ± 0.017 | 0.822 ± 0.028 | 0.762 | 0.999 | 0.53 | 7.0 M | 28 MB | 5.9 min | 2.3 ms/img |
| 4 | Logistic regression on pixels (simple baseline) | 0.895 ± 0.006 | 0.774 ± 0.042 | 0.704 | 0.982 | 0.43 | 0.15 M | 0.6 MB | 3.0 min | 1.3 ms/img |
| 5 | Majority class (reference) | 0.500 | 0.626 | 0.500 | 1.000 | 0.00 | 0 | 0 | 0 | — |

The 95% bootstrap CI on AUROC for a single run is about ±0.015 (ResNet-50, seed 42: [0.956, 0.984]). ResNet-50's
lead is outside that interval; DenseNet-121 vs the small CNN is not (they swap places between seeds). On the
in-distribution validation split every model scores AUROC ≥ 0.99 — including logistic regression (0.992) — so the
test-folder ranking is a ranking of robustness to the collection shift described in §3.

Sensitivity is ≥ 0.98 for every model while specificity is 0.43–0.64: at the 0.5 threshold all models over-call
pneumonia on the test folder. §6 gives the cause.

Figures: `results/figures/main_auroc.png`, `results/figures/ablation_permute.png`. Full tables: `results/summary.md`.

## 6. Explanation and ablations

**Why ResNet-50 wins — a property of the data.** The dataset carries two kinds of class signal. A *global* one —
brightness, contrast and image geometry (§3: image height alone gives AUROC 0.92) — and a *local* one, the texture
of opacities in the lung fields. In-distribution the global signal is enough: a linear model on raw pixels reaches
AUROC 0.99 on validation. The test folder is a separate collection whose global cues are shifted (63% vs 74%
pneumonia; wider normal scans), so what still transfers is the local texture, and ResNet-50's ImageNet-pretrained
filters read it best.

**Ablation A (presented) — fixed pixel permutation.** `python -m cxr.train --model <m> --ablation permute` applies
one fixed random permutation of pixel positions to every image in every split. Pixel values and class balance are
unchanged; spatial locality is destroyed. Prediction: logistic regression (permutation-invariant by construction) is
unchanged, every CNN falls to its level, and the ranking changes. Result (mean of seeds 42 and 43, test AUROC):

| Model | Original | Permuted | Δ | Rank original → permuted |
|---|---|---|---|---|
| ResNet-50 | 0.971 | 0.893 | −0.078 | 1 → 3 |
| Small CNN | 0.942 | 0.846 | −0.095 | 2 → 4 |
| DenseNet-121 | 0.936 | 0.909 | −0.027 | 3 → 1 |
| Logistic regression | 0.895 | 0.895 | −0.0001 | 4 → 2 |

Both seeds agree individually. The winner loses first place and the linear control does not move, as predicted.

**Ablation B (run, not built on) — training-set size.** `--train_fraction 0.25 / 0.1` keeps a stratified fraction
of training *patients* with validation and test unchanged. At one seed the curves are not monotonic (DenseNet-121:
0.946 at 10%, 0.957 at 25%, 0.923 at 100%). That is informative — more data from the shifted collection does not help
on the test folder — but it does not test a clean prediction, so we report it and do not build on it
(`results/ablation_fraction.md`).

**Extra check — the padding cue.** The audit predicted that pad-to-square turns the aspect-ratio difference into a
black-bar cue that misfires on the wider test normals: an aspect-ratio rule alone gives specificity 0.50 on the
test folder, the same as the CNNs. Re-running seed 42 with images squashed to a square instead of padded
(`RESIZE=stretch`) raises specificity for every model (ResNet-50 0.57 → 0.81, DenseNet-121 0.47 → 0.73, small CNN
0.55 → 0.74, logistic regression 0.50 → 0.60) and ResNet-50's AUROC to 0.990 (accuracy 0.93). The ranking of the
winner is unchanged. The padded run remains the main comparison because it was our pre-registered preprocessing;
the squashed run checks the audit's hypothesis (`results/check_stretch.csv`).

**Why random re-splits of this dataset report about 97% accuracy.** A by-image random split puts the same patient
on both sides (61% of validation images in our simulation) and evaluates on in-distribution images. On the real
test folder the honest numbers are 86% accuracy / 0.97 AUROC (padded) and 93% / 0.99 (squashed).

**Limits.** Two seeds for the main table and one for the squash check; 618 test images; DenseNet-121 vs small CNN
not separable; the 0.5 threshold was never tuned for the test prior; GPU kernels are not all deterministic.

## 7. Contributions

| Member | Share | Contribution |
|---|---|---|
| Mahd Hindi | 40% | Pipeline code (`src/cxr/`, scripts, notebook), repository setup, seed-42 run and ablations, README |
| Khaled AlHassani | 30% | Data audit review and split validation, seed-43 run, results tables and figures, slides 2–3 |
| Abdullah AlKaabi | 30% | Squash-resize check, explanation and ablation analysis, slides 1, 4–5, dataset provenance and citations |

## 8. Use of AI assistants

Claude (Anthropic, claude.ai) was used as a coding and writing assistant: it drafted the pipeline code in
`src/cxr/`, the scripts, the Colab notebook and this README, proposed the ablation designs, and helped interpret
the results. All experiments were executed by the group on Google Colab from this repository; the group chose the
dataset, the models and the ablations, reviewed and edited the code, checked the audit findings against the raw
data, and wrote the slides. No number in the slides or this README was produced by an AI assistant without being
computed by the code here.

## 9. Third-party code, libraries and citations

* PyTorch and torchvision (ResNet-50 `IMAGENET1K_V2` and DenseNet-121 `IMAGENET1K_V1` weights) — Paszke et al., NeurIPS 2019
* scikit-learn (metrics), NumPy, pandas, Pillow, matplotlib, tqdm
* ImageHash (pHash near-duplicate detection) — J. Buchner, https://github.com/JohannesBuchner/imagehash
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 · Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017
* Kermany et al. 2018 (dataset, §1)

Code in this repository is released under the MIT licence (`LICENSE`). The dataset keeps its own CC BY 4.0 licence.

## 10. Repository layout

| Path | Role |
|---|---|
| `src/cxr/index.py` | scans any folder layout; recovers class, original split and patient id from file names |
| `src/cxr/prepare_data.py` | decodes once; audits problems; removes duplicates; patient-grouped split; tensor cache |
| `src/cxr/dataset.py` | the single preprocessing pipeline all models share, plus the ablations |
| `src/cxr/models.py` | the five models |
| `src/cxr/train.py` | one training/evaluation loop; writes a results JSON per run |
| `src/cxr/report.py` | aggregates runs into tables, figures and `results/summary.md` |
| `scripts/run_all.sh`, `scripts/download_data.py`, `scripts/lr_sweep.sh` | reproduction, data download, optional learning-rate check |
| `tests/smoke_test.sh` | offline end-to-end check on synthetic images |
| `data/audit.md`, `data/index.csv` | the audit and the exact split used |
| `results/` | every run's JSON, tables, figures, `summary.md` |
| `slides/` | the five slides |
