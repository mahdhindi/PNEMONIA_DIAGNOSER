#!/usr/bin/env bash
# End-to-end reproduction.  From the repo root:
#   bash scripts/run_all.sh                      # seed 42
#   SEEDS="42 43 44" bash scripts/run_all.sh     # three seeds for mean +- std
#   EPOCHS=12 RAW=data/raw bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-12}"
RAW="${RAW:-data/raw}"
MODELS="${MODELS:-majority logreg smallcnn resnet50 densenet121}"
TRAINABLE="${TRAINABLE:-logreg smallcnn resnet50 densenet121}"
FRACTIONS="${FRACTIONS:-0.25 0.1}"
IMG_SIZE="${IMG_SIZE:-224}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--no_pretrained --num_workers 0" for offline smoke tests

echo "== 1/4 prepare data (audit + patient-grouped split + cache)"
[ -f data/index.csv ] || python -m cxr.prepare_data --source "$RAW" --out data --seed 42 --img_size "$IMG_SIZE"

for SEED in $SEEDS; do
  echo "== 2/4 main comparison, seed $SEED"
  for M in $MODELS; do
    python -m cxr.train --model "$M" --seed "$SEED" --epochs "$EPOCHS" --img_size "$IMG_SIZE" $EXTRA
  done
  echo "== 3/4 ablation A (fixed pixel permutation), seed $SEED"
  for M in $TRAINABLE; do
    python -m cxr.train --model "$M" --seed "$SEED" --epochs "$EPOCHS" --img_size "$IMG_SIZE" --ablation permute $EXTRA
  done
  echo "== 3/4 ablation B (training-set size), seed $SEED"
  for F in $FRACTIONS; do
    for M in $TRAINABLE; do
      python -m cxr.train --model "$M" --seed "$SEED" --epochs "$EPOCHS" --img_size "$IMG_SIZE" --train_fraction "$F" $EXTRA
    done
  done
done

echo "== 4/4 report"
python -m cxr.report --results results --data data
