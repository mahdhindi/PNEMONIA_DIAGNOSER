#!/usr/bin/env bash
# Optional: check that each model's default learning rate is sensible by validation AUROC
# (uses --tag so the sweep runs never mix with the main results).  Compare val AUROC in results/lr-*.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
for M in logreg smallcnn resnet50 densenet121; do
  for LR in 1e-3 1e-4 1e-5; do
    python -m cxr.train --model "$M" --seed 42 --epochs "${EPOCHS:-6}" --lr "$LR" --results results/lr_sweep --tag "lr$LR"
  done
done
python - << 'PY'
import json, glob
for f in sorted(glob.glob("results/lr_sweep/*.json")):
    r = json.load(open(f)); print(f"{r['model']:12s} lr={r['config']['lr']:<7g} val_auroc={r['val']['auroc']:.4f}")
PY
