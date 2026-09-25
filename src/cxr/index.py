"""Scan a raw dataset folder and build an index of every image.

Supports the three common layouts of the Kermany et al. (2018) pediatric CXR
dataset without any hard-coded paths:

  * Mendeley / Kaggle original:  <root>/**/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
  * Roboflow classification export: <root>/{train,valid,test}/{NORMAL,PNEUMONIA}/*.jpg
    (file names keep the original stem plus a "_jpeg.rf.<hash>" suffix)

From each file name we recover:
  * patient key  - "person123_bacteria_45.jpeg" -> patient p123 (several images
                   per patient exist, which is the leakage hazard we audit)
  * subtype      - bacteria / virus / normal
  * orig_split   - the split folder the file was found in ("unknown" if none)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}

_RF_SUFFIX = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-f]{32}$", re.I)
_PNEU_RE = re.compile(r"^person(\d+)_(bacteria|virus)_(\d+)$", re.I)
_NORM_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)-(\d+)(?:-(\d+))?$", re.I)


def parse_stem(stem: str) -> dict:
    """Recover patient id and subtype from a Kermany-style file stem."""
    base = _RF_SUFFIX.sub("", stem)
    roboflow = base != stem
    m = _PNEU_RE.match(base)
    if m:
        return dict(base=base, pid=f"p{int(m.group(1))}", subtype=m.group(2).lower(),
                    roboflow=roboflow, parsed=True)
    m = _NORM_RE.match(base)
    if m:
        prefix = "n2" if m.group(1) else "n1"
        return dict(base=base, pid=f"{prefix}-{int(m.group(2))}", subtype="normal",
                    roboflow=roboflow, parsed=True)
    return dict(base=base, pid=None, subtype=None, roboflow=roboflow, parsed=False)


def scan(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        rel_parts = p.relative_to(root).parts[:-1]
        cls = None
        for part in reversed(rel_parts):
            if part.upper() in CLASS_TO_LABEL:
                cls = part.upper()
                break
        if cls is None:
            continue  # not inside a class folder
        split = "unknown"
        for part in reversed(rel_parts):
            if part.lower() in SPLIT_NAMES:
                split = SPLIT_NAMES[part.lower()]
                break
        info = parse_stem(p.stem)
        if info["parsed"]:
            # Patient numbering restarts in each original split of the Kermany
            # release, so the split is part of the key.  A Roboflow re-split has
            # lost that information -> fall back to the bare id (more conservative:
            # it may merge two different patients, never separate one).
            pkey = info["pid"] if (info["roboflow"] or split == "unknown") else f"{split}:{info['pid']}"
        else:
            pkey = f"file:{info['base']}"
        rows.append(dict(
            path=str(p), rel_path=str(p.relative_to(root)), class_name=cls,
            label=CLASS_TO_LABEL[cls], orig_split=split, base_name=info["base"],
            patient_key=pkey, subtype=info["subtype"] or ("normal" if cls == "NORMAL" else "unknown"),
            name_parsed=info["parsed"], roboflow_export=info["roboflow"],
        ))
    if not rows:
        raise SystemExit(f"No images found under {root} inside NORMAL/PNEUMONIA folders.")
    return pd.DataFrame(rows)
