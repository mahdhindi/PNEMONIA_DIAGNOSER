#!/usr/bin/env bash
# Offline end-to-end check on synthetic images (CPU, ~1 min). Numbers are meaningless; only exercises the code path.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
rm -rf data_smoke results_smoke runs_smoke
python tests/make_synthetic.py --out data_smoke/raw --n_patients 60
python -m cxr.prepare_data --source data_smoke/raw --out data_smoke --img_size 64 --seed 42
for M in majority logreg smallcnn resnet50 densenet121; do
  python -m cxr.train --model "$M" --seed 42 --data data_smoke --results results_smoke --runs runs_smoke \
    --img_size 64 --epochs 1 --num_workers 0 --no_pretrained
done
python -m cxr.train --model logreg --seed 42 --data data_smoke --results results_smoke --runs runs_smoke --img_size 64 --epochs 1 --num_workers 0 --ablation permute
python -m cxr.train --model logreg --seed 42 --data data_smoke --results results_smoke --runs runs_smoke --img_size 64 --epochs 1 --num_workers 0 --train_fraction 0.25
python -m cxr.report --results results_smoke --data data_smoke
echo "SMOKE TEST OK"
