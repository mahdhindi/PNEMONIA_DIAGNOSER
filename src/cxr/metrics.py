"""Metrics.  Primary task metric: AUROC (threshold-free, robust to the 73/27
class imbalance).  We also report accuracy, macro-F1, pneumonia recall
(sensitivity) and specificity at the 0.5 threshold, plus 95% bootstrap CIs so
that differences between models can be read against the noise of the test set."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y: np.ndarray, p: np.ndarray, subtype: np.ndarray | None = None,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    single_class = len(np.unique(y)) < 2
    out = dict(
        n=int(len(y)),
        accuracy=float(accuracy_score(y, yhat)),
        balanced_accuracy=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall=float(recall_score(y, yhat, zero_division=0)),
        specificity=float(tn / max(tn + fp, 1)),
        f1=float(f1_score(y, yhat, zero_division=0)),
        macro_f1=float(f1_score(y, yhat, average="macro", zero_division=0)),
        auroc=float(roc_auc_score(y, p)) if not single_class else float("nan"),
        auprc=float(average_precision_score(y, p)) if not single_class else float("nan"),
        confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    )
    # bootstrap CIs
    if n_boot and not single_class:
        rng = np.random.default_rng(seed)
        accs, aucs, mf1s = [], [], []
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if len(np.unique(y[i])) < 2:
                continue
            accs.append(accuracy_score(y[i], yhat[i]))
            aucs.append(roc_auc_score(y[i], p[i]))
            mf1s.append(f1_score(y[i], yhat[i], average="macro", zero_division=0))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out["ci95"] = dict(accuracy=ci(accs), auroc=ci(aucs), macro_f1=ci(mf1s))
    if subtype is not None:
        per = {}
        for s in sorted(set(map(str, subtype))):
            m = np.asarray(subtype).astype(str) == s
            if m.sum():
                per[s] = dict(n=int(m.sum()), recall_of_true_label=float((yhat[m] == y[m]).mean()))
        out["per_subtype_accuracy"] = per
    return out
