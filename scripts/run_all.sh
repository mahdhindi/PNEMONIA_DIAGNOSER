#!/usr/bin/env bash
# End-to-end reproduction.  From the repo root:
#   RESIZE=stretch FRACTIONS="" ABLATIONS="" bash scripts/run_all.sh   # extra check: remove the aspect-ratio (padding) cue
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
ABLATIONS="${ABLATIONS-permute}"          # set ABLATIONS="" to skip the permutation ablation
FRACTIONS="${FRACTIONS-0.25 0.1}"      # set FRACTIONS="" to skip the data-size ablation
IMG_SIZE="${IMG_SIZE:-224}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA="--no_pretrained --num_workers 0" for offline smoke tests

# run one configuration unless its results file already exists (lets an interrupted Colab session resume)
run() {  # run <model> <seed> [extra args...]
  local M="$1" SEED="$2"; shift 2
  local ABL="none" FRAC="1"
  local args=("$@"); for ((i=0; i<${#args[@]}; i++)); do
    [ "${args[i]}" = "--ablation" ] && ABL="${args[i+1]}"
    [ "${args[i]}" = "--train_fraction" ] && FRAC="$(python -c "print(f'{float(\"${args[i+1]}\"):g}')")"
  done
  local NAME="${M}__abl-${ABL}__frac-${FRAC}__seed-${SEED}"; [ "$RESIZE" != "pad" ] && NAME="${NAME}__${RESIZE}"
  if [ -f "results/${NAME}.json" ]; then echo "   skip ${NAME} (done)"; return; fi
  python -m cxr.train --model "$M" --seed "$SEED" --epochs "$EPOCHS" --img_size "$IMG_SIZE" --resize "$RESIZE" "$@" $EXTRA
}

echo "== 1/4 prepare data (audit + patient-grouped split + cache)"
RESIZE="${RESIZE:-pad}"
SUFFIX=""; [ "$RESIZE" != "pad" ] && SUFFIX="_$RESIZE"
[ -f "data/cache/images_${IMG_SIZE}${SUFFIX}.npy" ] || python -m cxr.prepare_data --source "$RAW" --out data --seed 42 --img_size "$IMG_SIZE" --resize "$RESIZE"

for SEED in $SEEDS; do
  echo "== 2/4 main comparison, seed $SEED"
  for M in $MODELS; do
    run "$M" "$SEED"
  done
  for A in $ABLATIONS; do
    echo "== 3/4 ablation A ($A), seed $SEED"
    for M in $TRAINABLE; do
      run "$M" "$SEED" --ablation "$A"
    done
  done
  echo "== 3/4 ablation B (training-set size), seed $SEED"
  for F in $FRACTIONS; do
    for M in $TRAINABLE; do
      run "$M" "$SEED" --train_fraction "$F"
    done
  done
done

echo "== 4/4 report"
python -m cxr.report --results results --data data
