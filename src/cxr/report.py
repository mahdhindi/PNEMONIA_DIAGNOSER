"""Aggregate results/*.json into the deliverable tables and figures.

    python -m cxr.report [--results results] [--data data]

Writes results/main_table.{csv,md}, results/ablation_permute.{csv,md},
results/ablation_fraction.{csv,md}, results/figures/*.png, results/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["majority", "logreg", "smallcnn", "resnet50", "densenet121"]
LABEL = {"majority": "Majority class", "logreg": "Logistic regression (pixels)", "smallcnn": "Small CNN (scratch)",
         "resnet50": "ResNet-50 (ImageNet, fine-tuned)", "densenet121": "DenseNet-121 (ImageNet, fine-tuned)"}


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_runs(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results.glob("*.json")):
        r = json.load(open(f))
        t = r["test"]
        rows.append(dict(model=r["model"], seed=r["seed"], ablation=r["ablation"], frac=float(r["train_fraction"]),
                         resize=r.get("resize", "pad"), tag=r.get("tag", ""), balanced_acc=t["balanced_accuracy"],
                         params_M=r["params_M"], ckpt_mb=r["checkpoint_mb"], train_time_s=r["train_time_s"],
                         infer_ms=r["infer_ms_per_img"], best_epoch=r["best_epoch"], n_train=r["n_train"],
                         auroc=t["auroc"], accuracy=t["accuracy"], macro_f1=t["macro_f1"], recall=t["recall"],
                         specificity=t["specificity"], precision=t["precision"],
                         auroc_lo=t.get("ci95", {}).get("auroc", [np.nan, np.nan])[0],
                         auroc_hi=t.get("ci95", {}).get("auroc", [np.nan, np.nan])[1],
                         val_auroc=r["val"]["auroc"], file=f.name))
    if not rows:
        raise SystemExit(f"no results JSON files in {results}")
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, keys) -> pd.DataFrame:
    metrics = ["auroc", "accuracy", "balanced_acc", "macro_f1", "recall", "specificity", "train_time_s", "infer_ms"]
    g = df.groupby(keys)
    out = g[metrics].mean()
    std = g[metrics].std().fillna(0.0)
    for m in ["auroc", "accuracy", "macro_f1"]:
        out[m + "_std"] = std[m]
    out["params_M"] = g["params_M"].first()
    out["ckpt_mb"] = g["ckpt_mb"].first()
    out["n_seeds"] = g["seed"].nunique()
    out["n_train"] = g["n_train"].first()
    out["auroc_lo"] = g["auroc_lo"].first()
    out["auroc_hi"] = g["auroc_hi"].first()
    return out.reset_index()


def rank(df: pd.DataFrame, col="auroc") -> pd.Series:
    return df[col].rank(ascending=False, method="min").astype(int)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    R = Path(a.results); F = R / "figures"; F.mkdir(exist_ok=True, parents=True)
    df_all = load_runs(R)
    df = df_all[(df_all.resize == "pad") & (df_all.tag == "")]
    order = [m for m in ORDER if m in set(df.model)]

    # ------------------------------------------------------------ main table
    main = df[(df.ablation == "none") & (df.frac == 1.0)]
    mt = agg(main, ["model"]).set_index("model").loc[order].reset_index()
    mt["rank"] = rank(mt)
    pm = lambda r, m: f"{r[m]:.4f}" + (f" ± {r[m + '_std']:.4f}" if r["n_seeds"] > 1 else "")
    show = pd.DataFrame({
        "Rank": mt["rank"], "Model": mt["model"].map(LABEL),
        "AUROC": [pm(r, "auroc") for _, r in mt.iterrows()],
        "AUROC 95% CI (seed 1)": [f"[{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]" if not pd.isna(r.auroc_lo) else "" for _, r in mt.iterrows()],
        "Accuracy": [pm(r, "accuracy") for _, r in mt.iterrows()],
        "Balanced acc.": mt["balanced_acc"].map("{:.4f}".format),
        "Macro-F1": [pm(r, "macro_f1") for _, r in mt.iterrows()],
        "Recall (pneu.)": mt["recall"].map("{:.4f}".format),
        "Specificity": mt["specificity"].map("{:.4f}".format),
        "Params (M)": mt["params_M"].map("{:.2f}".format),
        "Ckpt (MB)": mt["ckpt_mb"].map("{:.1f}".format),
        "Train time (min)": (mt["train_time_s"] / 60).map("{:.1f}".format),
        "Infer (ms/img)": mt["infer_ms"].map("{:.2f}".format),
        "Seeds": mt["n_seeds"],
    }).sort_values("Rank")
    mt.to_csv(R / "main_table.csv", index=False)
    (R / "main_table.md").write_text(md_table(show))

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(mt)); vals = mt["auroc"].to_numpy()
    err = np.vstack([np.nan_to_num(vals - mt["auroc_lo"].to_numpy(), nan=0), np.nan_to_num(mt["auroc_hi"].to_numpy() - vals, nan=0)])
    if mt["n_seeds"].max() > 1:
        err = np.vstack([mt["auroc_std"], mt["auroc_std"]])
    ax.barh(y, vals, xerr=err, color=["#9aa0a6", "#8ab4f8", "#fbbc04", "#34a853", "#137333"][:len(mt)], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[m]}\n{p:.1f}M params · {t / 60:.1f} min train"
                        for m, p, t in zip(mt["model"], mt["params_M"], mt["train_time_s"])], fontsize=8)
    ax.invert_yaxis()
    lo = max(0.4, np.nanmin(vals) - 0.1)
    ax.set_xlim(lo, 1.0 + (1.0 - lo) * 0.12)
    ax.set_xlabel("Test AUROC (error bars: 95% bootstrap CI or ± std over seeds)")
    for i, v in enumerate(vals):
        ax.text(v + err[1][i] + (1.0 - lo) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_title("Test AUROC per model (identical data, preprocessing, seeds)")
    fig.tight_layout(); fig.savefig(F / "main_auroc.png", dpi=200); plt.close(fig)

    parts = ["# Results summary", "", "## Main comparison (test split)", "", md_table(show), "",
             "![](figures/main_auroc.png)", ""]

    # ------------------------------------------------------ ablation A: permute
    perm = df[(df.ablation == "permute") & (df.frac == 1.0)]
    if len(perm):
        pa = agg(perm, ["model"]).set_index("model")
        ma = mt.set_index("model")
        rows = []
        for m in order:
            if m in pa.index and m in ma.index:
                rows.append(dict(model=m, auroc_normal=ma.at[m, "auroc"], auroc_permuted=pa.at[m, "auroc"],
                                 delta=pa.at[m, "auroc"] - ma.at[m, "auroc"],
                                 acc_normal=ma.at[m, "accuracy"], acc_permuted=pa.at[m, "accuracy"]))
        ab = pd.DataFrame(rows)
        ab["rank_normal"] = rank(ab, "auroc_normal"); ab["rank_permuted"] = rank(ab, "auroc_permuted")
        ab.to_csv(R / "ablation_permute.csv", index=False)
        showa = ab.assign(Model=ab["model"].map(LABEL))[["Model", "auroc_normal", "auroc_permuted", "delta",
                                                          "rank_normal", "rank_permuted", "acc_normal", "acc_permuted"]]
        (R / "ablation_permute.md").write_text(md_table(showa))
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(ab)); w = 0.38
        ax.bar(x - w / 2, ab["auroc_normal"], w, label="original images", color="#34a853")
        ax.bar(x + w / 2, ab["auroc_permuted"], w, label="fixed pixel permutation", color="#ea4335")
        ax.set_xticks(x); ax.set_xticklabels(ab["model"].map(LABEL), rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.0); ax.set_ylabel("Test AUROC"); ax.legend()
        ax.set_title("Ablation A: destroy spatial structure, keep every pixel value")
        fig.tight_layout(); fig.savefig(F / "ablation_permute.png", dpi=200); plt.close(fig)
        parts += ["## Ablation A - fixed pixel permutation", "",
                  "Prediction: the linear model is unaffected (permutation-invariant); models that won by exploiting "
                  "local spatial texture lose their advantage and the ranking changes.", "", md_table(showa), "",
                  "![](figures/ablation_permute.png)", ""]

    # ----------------------------------------------------- ablation B: fraction
    frac = df[df.ablation == "none"]
    if frac["frac"].nunique() > 1:
        fa = agg(frac, ["model", "frac"])
        piv = fa.pivot(index="model", columns="frac", values="auroc").loc[[m for m in order if m in set(fa.model)]]
        piv.columns = [f"{int(c * 100)}% train" for c in piv.columns]
        ranks = piv.rank(ascending=False, method="min").astype("Int64")
        ranks.columns = [c + " rank" for c in ranks.columns]
        n_by = fa.groupby("frac")["n_train"].max().fillna(0)
        fb = pd.concat([piv, ranks], axis=1).reset_index()
        fb["model"] = fb["model"].map(LABEL)
        fb.to_csv(R / "ablation_fraction.csv", index=False)
        (R / "ablation_fraction.md").write_text(md_table(fb))
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in piv.index:
            xs = [float(c.split("%")[0]) for c in piv.columns]
            ax.plot(xs, piv.loc[m].to_numpy(), marker="o", label=LABEL[m])
        ax.set_xlabel("fraction of training patients kept (%)"); ax.set_ylabel("Test AUROC"); ax.set_xscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([f"{int(v)}%" for v in xs]); ax.legend(fontsize=8)
        ax.set_title("Ablation B: shrink the training set (val/test unchanged)")
        fig.tight_layout(); fig.savefig(F / "ablation_fraction.png", dpi=200); plt.close(fig)
        parts += ["## Ablation B - training-set size", "",
                  f"Training images per fraction: {dict(zip(piv.columns, n_by.astype(int).tolist()))}. Prediction: the gap "
                  "between ImageNet-pretrained models and models learned from scratch widens as data shrinks.", "",
                  md_table(fb), "", "![](figures/ablation_fraction.png)", ""]

    # ------------------------------------------ extra check: stretch resize
    st = df_all[(df_all.resize == "stretch") & (df_all.ablation == "none") & (df_all.frac == 1.0)]
    if len(st):
        sa = agg(st, ["model"]).set_index("model"); ma = mt.set_index("model")
        rows = [dict(Model=LABEL[m], auroc_pad=ma.at[m, "auroc"], auroc_stretch=sa.at[m, "auroc"],
                     spec_pad=ma.at[m, "specificity"], spec_stretch=sa.at[m, "specificity"],
                     bal_acc_pad=ma.at[m, "balanced_acc"], bal_acc_stretch=sa.at[m, "balanced_acc"])
                for m in order if m in sa.index and m in ma.index]
        sc = pd.DataFrame(rows)
        sc.to_csv(R / "check_stretch.csv", index=False)
        parts += ["## Extra check - removing the aspect-ratio (padding) cue", "",
                  "Same split, same seeds, images squashed to a square instead of padded, so the original aspect ratio is no "
                  "longer visible. Tests whether the false positives on test NORMAL images come from the geometry shortcut "
                  "found in the audit.", "", md_table(sc), ""]

    # ------------------------------------------------- errors by subtype
    sub_rows = []
    for f in sorted(R.glob("*__abl-none__frac-1__seed-*.json")):
        r = json.load(open(f)); per = r["test"].get("per_subtype_accuracy", {})
        if per:
            sub_rows.append(dict(model=LABEL.get(r["model"], r["model"]), seed=r["seed"],
                                 **{f"{k} (n={v['n']})": v["recall_of_true_label"] for k, v in per.items()}))
    if sub_rows:
        sub = pd.DataFrame(sub_rows).groupby("model").mean(numeric_only=True).drop(columns="seed", errors="ignore")
        sub = sub.reindex([LABEL[m] for m in order if LABEL[m] in sub.index]).reset_index()
        sub.to_csv(R / "per_subtype.csv", index=False)
        parts += ["## Correct-classification rate by image subtype (test split)", "",
                  "Pneumonia is one label covering bacterial and viral cases; viral pneumonia is typically more diffuse "
                  "and harder to see.", "", md_table(sub), ""]

    # ------------------------------------------------------------- audit
    audit = Path(a.data) / "audit.md"
    if audit.exists():
        parts += ["", audit.read_text()]
    (R / "summary.md").write_text("\n".join(parts))
    print("\n".join(parts[:6]))
    print(f"\n[report] wrote {R}/summary.md, main_table.md/csv, figures/")


if __name__ == "__main__":
    main(sys.argv[1:])
