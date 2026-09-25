# Pneumonia detection from pediatric chest X-rays — a fair comparison of five classifiers

**CSBP711 Advanced Artificial Intelligence, UAEU, Fall 2026 — Assignment 1: Datasets and Algorithm Comparison**

Group: _<member 1>, <member 2>, <member 3>, <member 4>, <member 5>_ &nbsp;|&nbsp; Slides: `slides/CSBP711_A1_slides.pdf`

This repository takes the chest X-ray pneumonia classifier our group built in an earlier software-engineering
course (YOLOv11n vs ResNet-50V2 + Flask app) and does what that project never did: audit the data properly,
split it without leakage, compare five algorithms under one identical protocol, and test *why* the winner wins
with an ablation. Every number in the slides is produced by the code here; nothing is copied from a paper.

---

## 1. Dataset

| | |
|---|---|
| **Name** | Chest X-Ray Images (Pneumonia) — pediatric, 2 classes (NORMAL / PNEUMONIA) |
| **Upstream source** | Kermany, Zhang & Goldbaum (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, Mendeley Data **v2**, https://doi.org/10.17632/rscbjbr9sj.2 |
| **Mirror used for download** | Kaggle — https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (5,863 files: 5,216 train / 16 val / 624 test) |
| **Licence** | **CC BY 4.0** (research use permitted, attribution required) |
| **Version / download date** | see `data/raw/DOWNLOAD_INFO.json`, written automatically by `scripts/download_data.py` — _<fill in after download, e.g. "Kaggle mirror, downloaded 2026-09-27">_ |
| **How it was collected** | Anterior–posterior chest radiographs of children aged 1–5 from Guangzhou Women and Children's Medical Center, taken during routine care; low-quality scans were removed; labels were graded by two physicians, with the evaluation set checked by a third (Kermany et al., *Cell* 172(5):1122–1131, 2018, https://doi.org/10.1016/j.cell.2018.02.010). |
| **Relation to our previous project** | The Roboflow dataset we used before (https://universe.roboflow.com/mohamed-traore-2ekkp/chest-x-rays-qjmia, CC BY 4.0, 5,824 source images) is a re-split, re-export of this same data with 3× augmentation baked into the training folder. We go back to the un-augmented source so that preprocessing is under our control and identical for every model. |

Everything the audit found (size, resolution spread, duplicates, near-duplicates, multi-image patients, class
imbalance, class-prior shift between the original train and test folders, the 16-image validation folder, and
a simulation of how much leakage a naive by-image split would cause) is written to `data/audit.md` by
`prepare_data.py` and summarised on slide 2.

## 2. Reproduce everything

```bash
git clone <this repo> && cd <this repo>
pip install -r requirements.txt            # PyTorch >= 2.2 with CUDA recommended (Colab T4 is enough)

# 1) data  (Kaggle API key in ~/.kaggle/kaggle.json)   — alternatives: `roboflow --version N --api_key ..` or `zip --path file.zip`
python scripts/download_data.py kaggle

# 2) audit + patient-grouped split + tensor cache, 3) five models, 4) both ablations, 5) tables & figures
bash scripts/run_all.sh                    # one seed (42), ~1 h on a T4
SEEDS="42 43 44" bash scripts/run_all.sh   # three seeds -> mean ± std in the table
```

Outputs: `data/audit.md`, `results/*.json` (one per run), `results/main_table.md`, `results/ablation_*.md`,
`results/figures/*.png`, `results/summary.md`. A Colab notebook that runs exactly these commands is in
`colab_run.ipynb`. No hard-coded local paths; every script takes `--data/--results` arguments.

Offline end-to-end check without the real data (synthetic images, CPU, ~1 min):

```bash
bash tests/smoke_test.sh
```

## 3. What the code does

| File | Role |
|---|---|
| `src/cxr/index.py` | scans any folder layout, recovers class, original split and **patient id** from file names (`person123_bacteria_45.jpeg` → patient 123) |
| `src/cxr/prepare_data.py` | decodes once; audits problems; removes exact duplicates; finds near-duplicates (pHash-256); assigns **whole patients** (and near-duplicate pairs) to one split; keeps the dataset's own test folder as the held-out test set; caches 224×224 grayscale tensors |
| `src/cxr/dataset.py` | the **single preprocessing pipeline** all models share, plus the two ablations |
| `src/cxr/models.py` | majority class · logistic regression on pixels · small CNN from scratch · ResNet-50 (ImageNet) · DenseNet-121 (ImageNet) |
| `src/cxr/train.py` | one training/evaluation loop for every model; writes a results JSON with metrics, CIs, params, checkpoint size, training time, inference latency |
| `src/cxr/report.py` | aggregates runs into the comparison table, ablation tables, figures and `summary.md` |
| `scripts/run_all.sh` | the whole experiment in one command |

## 4. Comparison protocol (why the table is fair)

* **Same data**: identical train/val/test split (`data/index.csv`), fixed with seed 42, grouped by patient.
* **Same preprocessing**: grayscale → pad to square → 224×224 → ImageNet normalisation; the same light augmentation
  (random resized crop 0.8–1, ±7° rotation, ±15 % brightness/contrast) for every trainable model. The logistic
  regression sees exactly the tensors the CNNs see; it only flattens them.
* **Same seeds**: `set_seed()` fixes Python, NumPy, PyTorch and DataLoader RNGs; `cudnn.deterministic` and
  `use_deterministic_algorithms(warn_only=True)` are on. Two CPU runs with the same seed are bit-identical;
  on GPU a few kernels are non-deterministic, so we also report ± std over seeds when more than one is run.
* **Same optimiser and budget**: AdamW, weight decay 1e-4, 1 warm-up epoch + cosine decay, batch 32, 12 epochs,
  checkpoint selected by validation AUROC. The only per-model difference is the learning rate
  (from-scratch CNN 1e-3, linear 1e-4, fine-tuned pretrained 1e-4), recorded in every results file.
* **Metrics**: AUROC (primary; threshold-free and robust to the 73/27 class imbalance), accuracy, macro-F1,
  pneumonia recall and specificity, with 95 % bootstrap confidence intervals over the test set; parameter count,
  checkpoint size, wall-clock training time, inference ms/image.

## 5. Results

_Paste `results/main_table.md` here after the run and link the figures._

## 6. Explanation and ablations

_Filled in from the actual numbers — see slide 5. Ablation A (`--ablation permute`) applies one fixed random
permutation of pixel positions to every image in every split: pixel values and class balance are untouched,
spatial structure is destroyed; a permutation-invariant model (logistic regression) is unaffected by
construction. Ablation B (`--train_fraction 0.25 / 0.1`) keeps a stratified fraction of training **patients**
with val/test unchanged._

## 7. Contributions

_Required by the assignment — one line per member, matching the commit history._

| Member | Contribution |
|---|---|
| _<name>_ | _e.g. data audit and split (`prepare_data.py`), slide 2_ |
| _<name>_ | _e.g. logistic regression + small CNN runs, LR sanity check_ |
| _<name>_ | _e.g. ResNet-50 / DenseNet-121 runs, timing and model-size columns_ |
| _<name>_ | _e.g. ablations, figures, slide 5_ |
| _<name>_ | _e.g. README, dataset provenance, slide 1, repository hygiene_ |

## 8. Use of AI assistants

_Required by the assignment — state honestly. Suggested wording:_
Claude (Anthropic) was used to draft the pipeline code in `src/cxr/`, the scripts, and this README, and to
help structure the analysis. All experiments were executed by the group on Google Colab; the group reviewed
and edited the code, chose the dataset, models and ablations, interpreted the results and wrote the slides.

## 9. Third-party code, libraries and citations

* PyTorch and torchvision (ResNet-50 `IMAGENET1K_V2` and DenseNet-121 `IMAGENET1K_V1` weights) — Paszke et al. 2019
* scikit-learn (metrics), NumPy, pandas, Pillow, matplotlib, tqdm
* ImageHash (pHash near-duplicate detection) — J. Buchner, https://github.com/JohannesBuchner/imagehash
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 · Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017
* Kermany et al. 2018 (dataset, see §1)

Code in this repository is released under the MIT licence (`LICENSE`). The dataset keeps its own CC BY 4.0 licence.
