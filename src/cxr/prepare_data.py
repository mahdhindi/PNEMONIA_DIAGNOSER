"""Prepare the dataset in one pass: decode -> audit -> split -> cache.

    python -m cxr.prepare_data --source data/raw --out data

Outputs (all under --out):
  index.csv              one row per image: labels, patient key, split, problems
  cache/images_<S>.npy   uint8 [N, S, S] grayscale, padded-to-square, resized
  audit.json / audit.md  dataset statistics and every problem we found

The preprocessing here (pad to square, resize to S x S, grayscale) is the
single preprocessing shared by every model in the comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .index import scan

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------
def load_gray_square(path: str, size: int):
    """Grayscale, pad to square with black, resize to size x size."""
    with Image.open(path) as im:
        mode, (w, h) = im.mode, im.size
        im = im.convert("L")
        s = max(w, h)
        canvas = Image.new("L", (s, s), 0)
        canvas.paste(im, ((s - w) // 2, (s - h) // 2))
        canvas = canvas.resize((size, size), Image.LANCZOS)
        arr = np.asarray(canvas, dtype=np.uint8)
    return arr, mode, w, h


def _phash_bits(arr: np.ndarray, hash_size: int = 16) -> np.ndarray:
    h = imagehash.phash(Image.fromarray(arr), hash_size=hash_size)
    return np.packbits(h.hash.flatten())


def near_duplicate_pairs(bits: np.ndarray, thr: int, chunk: int = 256):
    """All (i, j, dist) with i<j and Hamming(bits_i, bits_j) <= thr, plus each
    row's nearest-neighbour distance (used to justify the threshold)."""
    n = bits.shape[0]
    nn = np.full(n, 10**6, dtype=np.int32)
    pairs = []
    for i in range(0, n, chunk):
        x = bits[i:i + chunk]
        d = _POPCOUNT[np.bitwise_xor(x[:, None, :], bits[None, :, :])].sum(axis=2, dtype=np.int32)
        for k in range(x.shape[0]):
            d[k, i + k] = 10**6
        nn[i:i + chunk] = d.min(axis=1)
        ii, jj = np.nonzero(d <= thr)
        for a, b in zip(ii, jj):
            if i + a < b:
                pairs.append((int(i + a), int(b), int(d[a, b])))
    return pairs, nn


# ----------------------------------------------------------------------------
# union-find for grouping (patient key + near-duplicate edges)
# ----------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def stratified_group_split(df: pd.DataFrame, group: np.ndarray, fracs: dict, seed: int) -> pd.Series:
    """Assign whole groups to splits so that each split gets ~frac of the
    images of each class.  Remaining groups go to 'train'."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"g": group, "label": df["label"].to_numpy()})
    g_label = g.groupby("g")["label"].mean().round().astype(int)
    g_size = g.groupby("g").size()
    assign = {}
    for lab in sorted(g_label.unique()):
        gids = g_label[g_label == lab].index.to_numpy().copy()
        rng.shuffle(gids)
        total = int(g_size[gids].sum())
        target = {k: v * total for k, v in fracs.items()}
        cum = {k: 0 for k in fracs}
        for gid in gids:
            for name in fracs:
                if cum[name] < target[name]:
                    assign[gid] = name
                    cum[name] += int(g_size[gid])
                    break
            else:
                assign[gid] = "train"
    return pd.Series([assign[x] for x in group], index=df.index)


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder containing NORMAL/PNEUMONIA class folders (any depth)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15, help="only used with --split_strategy pooled")
    ap.add_argument("--split_strategy", choices=["auto", "original_test", "pooled"], default="auto",
                    help="original_test: keep the dataset's own test folder as the held-out test set "
                         "(patient-disjoint by construction) and carve val out of train by patient. "
                         "pooled: ignore original folders and re-split everything by patient.")
    ap.add_argument("--near_dup_bits", type=int, default=10,
                    help="pHash-256 Hamming distance at or below which two images count as near-duplicates")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = scan(args.source)
    print(f"[scan] {len(df)} image files under {args.source}")

    # ---------------------------------------------------------------- decode
    S = args.img_size
    images = np.zeros((len(df), S, S), dtype=np.uint8)
    md5s, hashes, modes, ws, hs, bad = [], [], [], [], [], []
    for i, p in enumerate(tqdm(df["path"], desc="[decode]", unit="img")):
        try:
            with open(p, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            arr, mode, w, h = load_gray_square(p, S)
            images[i] = arr
            md5s.append(md5); hashes.append(_phash_bits(arr)); modes.append(mode); ws.append(w); hs.append(h)
        except Exception as e:  # corrupt / unreadable file
            bad.append((p, repr(e)))
            md5s.append(None); hashes.append(np.zeros(32, np.uint8)); modes.append(None); ws.append(0); hs.append(0)
    df["md5"], df["mode"], df["width"], df["height"] = md5s, modes, ws, hs
    df["corrupt"] = df["md5"].isna()
    bits = np.stack(hashes)

    # ------------------------------------------------------------------ audit
    problems: list[str] = []
    audit: dict = {"n_files": int(len(df)), "img_size": S, "resize": "grayscale, pad-to-square (black), LANCZOS resize"}
    audit["per_class"] = df["class_name"].value_counts().to_dict()
    audit["per_split_class"] = df.pivot_table(index="orig_split", columns="class_name", values="path",
                                              aggfunc="count", fill_value=0).astype(int).to_dict()
    audit["n_corrupt"] = int(df["corrupt"].sum())
    if bad:
        problems.append(f"{len(bad)} unreadable/corrupt files were dropped (listed in index.csv).")

    ok = ~df["corrupt"]
    audit["image_modes"] = df.loc[ok, "mode"].value_counts().to_dict()
    if audit["image_modes"].get("RGB", 0):
        problems.append(f"{audit['image_modes']['RGB']} files are stored as 3-channel RGB although X-rays are "
                        f"grayscale; all images were converted to single-channel.")
    wh = df.loc[ok, ["width", "height"]]
    ar = (wh["width"] / wh["height"])
    audit["width"] = dict(min=int(wh["width"].min()), median=int(wh["width"].median()), max=int(wh["width"].max()))
    audit["height"] = dict(min=int(wh["height"].min()), median=int(wh["height"].median()), max=int(wh["height"].max()))
    audit["aspect_ratio"] = dict(min=round(float(ar.min()), 3), median=round(float(ar.median()), 3), max=round(float(ar.max()), 3))
    audit["n_unique_resolutions"] = int(wh.drop_duplicates().shape[0])
    problems.append(f"Image resolution is not standardised: {audit['n_unique_resolutions']} distinct sizes, "
                    f"width {audit['width']['min']}-{audit['width']['max']} px, aspect ratio "
                    f"{audit['aspect_ratio']['min']}-{audit['aspect_ratio']['max']}. We pad to square before resizing "
                    f"so anatomy is not stretched.")
    audit["name_parse_rate"] = round(float(df["name_parsed"].mean()), 4)
    audit["roboflow_export"] = bool(df["roboflow_export"].any())

    # exact duplicates
    df["drop_reason"] = ""
    dup_groups = df.loc[ok].groupby("md5")
    exact_dup_groups = {k: g for k, g in dup_groups if len(g) > 1}
    # packaging copies (same bytes AND same file name, e.g. a nested folder in the archive)
    # are not a dataset problem; same bytes under different names are.
    pack = {k: g for k, g in exact_dup_groups.items() if g["base_name"].nunique() == 1}
    real = {k: g for k, g in exact_dup_groups.items() if g["base_name"].nunique() > 1}
    n_pack = sum(len(g) - 1 for g in pack.values())
    n_extra = sum(len(g) - 1 for g in real.values())
    cross_class = sum(1 for g in real.values() if g["label"].nunique() > 1)
    cross_split = sum(1 for g in real.values() if g["orig_split"].nunique() > 1)
    audit["exact_duplicates"] = dict(groups=len(real), redundant_files=int(n_extra), packaging_copies=int(n_pack),
                                     groups_spanning_classes=int(cross_class), groups_spanning_splits=int(cross_split))
    if n_pack:
        problems.append(f"{n_pack} files are byte-identical copies of a file with the same name (archive packaging, "
                        f"e.g. a nested folder); one copy is kept.")
    if real:
        problems.append(f"{n_extra} byte-identical duplicate images stored under different names, in {len(real)} groups "
                        f"({cross_split} groups span two original splits, {cross_class} carry conflicting labels). "
                        f"One copy is kept; conflicting-label groups are dropped entirely.")
    for k, g in exact_dup_groups.items():
        if g["label"].nunique() > 1:
            df.loc[g.index, "drop_reason"] = "label_conflict_duplicate"
        else:
            df.loc[g.index[1:], "drop_reason"] = "exact_duplicate"
    df.loc[df["corrupt"], "drop_reason"] = "corrupt"
    keep = df["drop_reason"] == ""

    # near duplicates (perceptual hash) among kept images
    kidx = np.flatnonzero(keep.to_numpy())
    pairs, nn = near_duplicate_pairs(bits[kidx], args.near_dup_bits)
    pairs = [(int(kidx[a]), int(kidx[b]), d) for a, b, d in pairs]
    audit["near_duplicates"] = dict(
        threshold_bits=args.near_dup_bits, hash="pHash 16x16 (256 bit)", pairs=len(pairs),
        nn_distance_percentiles={f"p{q}": int(np.percentile(nn, q)) for q in (1, 5, 10, 25, 50, 75, 90)},
        pairs_spanning_splits=int(sum(df.at[a, "orig_split"] != df.at[b, "orig_split"] for a, b, _ in pairs)),
        pairs_spanning_classes=int(sum(df.at[a, "label"] != df.at[b, "label"] for a, b, _ in pairs)),
        pairs_spanning_patients=int(sum(df.at[a, "patient_key"] != df.at[b, "patient_key"] for a, b, _ in pairs)),
    )
    if pairs:
        problems.append(f"{len(pairs)} near-duplicate image pairs (pHash distance <= {args.near_dup_bits}/256): "
                        f"{audit['near_duplicates']['pairs_spanning_splits']} span the original train/test folders, "
                        f"{audit['near_duplicates']['pairs_spanning_classes']} have conflicting labels. Near-duplicates "
                        f"are forced into the same split so they cannot leak.")

    # patients
    kd = df[keep]
    pat = kd.groupby("patient_key").agg(n=("path", "size"), label=("label", "first"), split=("orig_split", "first"))
    audit["patients"] = dict(
        n_patient_keys=int(len(pat)),
        per_class={"NORMAL": int((pat["label"] == 0).sum()), "PNEUMONIA": int((pat["label"] == 1).sum())},
        with_multiple_images=int((pat["n"] > 1).sum()),
        images_in_multi_image_patients=int(pat.loc[pat["n"] > 1, "n"].sum()),
        max_images_per_patient=int(pat["n"].max()),
    )
    if audit["patients"]["with_multiple_images"] > 0:
        problems.append(f"{audit['patients']['with_multiple_images']} patients contribute more than one image "
                        f"({audit['patients']['images_in_multi_image_patients']} images, up to "
                        f"{audit['patients']['max_images_per_patient']} per patient). Splitting by image instead of by "
                        f"patient puts the same child on both sides of a split (leakage).")
    if audit["roboflow_export"]:
        problems.append("The source is a Roboflow re-export: the original train/test folders were pooled and "
                        "re-split at random by image, and patient ids from different original splits can no longer "
                        "be told apart. Patient grouping below is therefore conservative (merges by bare id).")

    # class prior per original split
    prior = kd.groupby("orig_split")["label"].mean().round(4).to_dict()
    audit["pneumonia_prior_per_orig_split"] = prior
    audit["class_balance_overall"] = round(float(kd["label"].mean()), 4)
    problems.append(f"Class imbalance: {100 * kd['label'].mean():.1f}% of images are PNEUMONIA. Accuracy alone is "
                    f"misleading; a majority-class predictor already scores that number.")
    if "train" in prior and "test" in prior and abs(prior["train"] - prior["test"]) > 0.03:
        problems.append(f"Class prior shifts between the original train ({100 * prior['train']:.1f}% pneumonia) and "
                        f"test ({100 * prior['test']:.1f}% pneumonia) folders: the test folder is a separate collection, "
                        f"not an i.i.d. sample of the training distribution.")
    if "val" in prior:
        n_val = int((kd["orig_split"] == "val").sum())
        if n_val < 100:
            problems.append(f"The dataset ships a validation folder of only {n_val} images - useless for model "
                            f"selection. We merge it into the training pool and carve a patient-grouped validation "
                            f"set of our own.")

    # --------------------------------------------------------------- split
    strategy = args.split_strategy
    if strategy == "auto":
        has_test = (kd["orig_split"] == "test").any() and (kd["orig_split"] == "train").any()
        strategy = "original_test" if has_test and not audit["roboflow_export"] else "pooled"
    audit["split_strategy"] = strategy

    df["split"] = "dropped"
    if strategy == "original_test":
        pool = kd[kd["orig_split"] != "test"]
        df.loc[kd.index[kd["orig_split"] == "test"], "split"] = "test"
        fracs = {"val": args.val_frac}
    else:
        pool = kd
        fracs = {"test": args.test_frac, "val": args.val_frac}

    pos = {idx: k for k, idx in enumerate(pool.index)}
    uf = UnionFind(len(pool))
    first_of_key = {}
    for k, idx in enumerate(pool.index):
        key = pool.at[idx, "patient_key"]
        if key in first_of_key:
            uf.union(first_of_key[key], k)
        else:
            first_of_key[key] = k
    n_neardup_unions = 0
    for a, b, _ in pairs:
        if a in pos and b in pos:
            uf.union(pos[a], pos[b]); n_neardup_unions += 1
    groups = np.array([uf.find(k) for k in range(len(pool))])
    split_assign = stratified_group_split(pool, groups, fracs, args.seed)
    df.loc[pool.index, "split"] = split_assign.values
    audit["n_groups_in_pool"] = int(len(np.unique(groups)))
    audit["near_dup_edges_used_for_grouping"] = int(n_neardup_unions)

    # leakage checks on the final split
    tr = df[df["split"] == "train"]; te = df[df["split"] == "test"]; va = df[df["split"] == "val"]
    leak = {
        "patient_keys_shared_train_val": int(len(set(tr["patient_key"]) & set(va["patient_key"]))),
        "patient_keys_shared_train_test": int(len(set(tr["patient_key"]) & set(te["patient_key"]))),
        "near_dup_pairs_train_test": int(sum((df.at[a, "split"], df.at[b, "split"]) in {("train", "test"), ("test", "train")} for a, b, _ in pairs)),
        "near_dup_pairs_train_val": int(sum((df.at[a, "split"], df.at[b, "split"]) in {("train", "val"), ("val", "train")} for a, b, _ in pairs)),
    }
    audit["leakage_after_our_split"] = leak
    if strategy == "original_test" and leak["near_dup_pairs_train_test"]:
        problems.append(f"{leak['near_dup_pairs_train_test']} near-duplicate pairs connect the original test folder to "
                        f"our training data. We keep the community-standard test folder unchanged (so results are "
                        f"comparable with the literature) but report this; the affected test images are listed in "
                        f"index.csv (column near_dup_of_train).")
        te_ids = {b if df.at[a, "split"] == "train" else a for a, b, _ in pairs
                  if {df.at[a, "split"], df.at[b, "split"]} == {"train", "test"}}
        df["near_dup_of_train"] = False
        df.loc[list(te_ids), "near_dup_of_train"] = True

    # what a naive by-image split would have done
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pool))
    n_val = int(round(args.val_frac * len(pool)))
    naive_val = pool.iloc[perm[:n_val]]; naive_tr = pool.iloc[perm[n_val:]]
    shared = set(naive_val["patient_key"]) & set(naive_tr["patient_key"])
    audit["naive_random_split_simulation"] = dict(
        val_frac=args.val_frac, patients_on_both_sides=len(shared),
        val_images_with_patient_in_train=int(naive_val["patient_key"].isin(shared).sum()),
        val_images_total=int(n_val),
    )
    problems.append(f"Simulated naive by-image random split ({int(100 * args.val_frac)}% val): "
                    f"{audit['naive_random_split_simulation']['val_images_with_patient_in_train']} of {n_val} validation "
                    f"images ({100 * audit['naive_random_split_simulation']['val_images_with_patient_in_train'] / max(n_val, 1):.1f}%) "
                    f"would share a patient with the training set. Our split assigns whole patients.")

    # subtype breakdown
    final = df[df["split"] != "dropped"]
    audit["subtype_counts"] = final.pivot_table(index="split", columns="subtype", values="path",
                                                aggfunc="count", fill_value=0).astype(int).to_dict()
    audit["final_splits"] = df[df["split"] != "dropped"].pivot_table(index="split", columns="class_name", values="path",
                                                                     aggfunc="count", fill_value=0).astype(int).to_dict()
    audit["final_split_sizes"] = df["split"].value_counts().to_dict()
    audit["n_features"] = dict(raw_pixels_median=int(audit["width"]["median"] * audit["height"]["median"]),
                               model_input=f"{S}x{S}x3 = {S * S * 3} values (grayscale replicated to 3 channels)")
    audit["problems"] = problems
    audit["prepared_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit["python"] = platform.python_version()
    audit["source_dir"] = str(Path(args.source).resolve())
    audit["seconds"] = round(time.time() - t0, 1)

    # -------------------------------------------------------------- cache
    keep_idx = np.flatnonzero((df["split"] != "dropped").to_numpy())
    df["cache_idx"] = -1
    df.loc[df.index[keep_idx], "cache_idx"] = np.arange(len(keep_idx))
    np.save(out / "cache" / f"images_{S}.npy", images[keep_idx])
    df.to_csv(out / "index.csv", index=False)
    with open(out / "audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)
    with open(out / "audit.md", "w") as f:
        f.write(render_audit_md(audit))
    print("\n".join(f" - {p}" for p in problems))
    print(f"\n[done] splits: {audit['final_split_sizes']}  ->  {out}/index.csv, cache/images_{S}.npy, audit.md "
          f"({audit['seconds']} s)")


def render_audit_md(a: dict) -> str:
    L = ["# Data audit", "",
         f"Prepared {a['prepared_at']} from `{a['source_dir']}` (strategy: `{a['split_strategy']}`).", "",
         "## Size", "",
         f"- Files found: **{a['n_files']}** ({a['per_class']})",
         f"- Original folders x class: {a['per_split_class']}",
         f"- Corrupt: {a['n_corrupt']}; exact-duplicate redundant files: {a['exact_duplicates']['redundant_files']}",
         f"- Patients (grouping keys): {a['patients']['n_patient_keys']} ({a['patients']['per_class']}); "
         f"{a['patients']['with_multiple_images']} have >1 image (max {a['patients']['max_images_per_patient']})",
         f"- Resolution: width {a['width']}, height {a['height']}, aspect {a['aspect_ratio']}, "
         f"{a['n_unique_resolutions']} distinct sizes; modes {a['image_modes']}",
         f"- Features: model input {a['n_features']['model_input']}; median raw image ~{a['n_features']['raw_pixels_median']:,} pixels",
         "", "## Final splits (patient-grouped, seed-fixed)", "",
         f"- {a['final_splits']}", f"- Subtypes: {a['subtype_counts']}",
         f"- Leakage checks after split: {a['leakage_after_our_split']}",
         "", "## Problems found", ""]
    L += [f"{i + 1}. {p}" for i, p in enumerate(a["problems"])]
    L += ["", "## Near-duplicate calibration", "",
          f"Nearest-neighbour pHash distance percentiles: {a['near_duplicates']['nn_distance_percentiles']} "
          f"(threshold {a['near_duplicates']['threshold_bits']} bits of 256).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main(sys.argv[1:])
