"""Generate a tiny synthetic dataset with the Kermany folder/file-name layout so
the full pipeline (audit -> split -> train -> report) can be smoke-tested
without the real data.  Not for any scientific use."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def make_img(rng, pneumonia: bool, w: int, h: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    base = 60 + 40 * np.exp(-(((xx - w / 2) / (w / 3)) ** 2 + ((yy - h / 2) / (h / 2.5)) ** 2))
    img = base + rng.normal(0, 12, (h, w))
    if pneumonia:  # local bright blobs = "opacities"
        for _ in range(rng.integers(2, 5)):
            cx, cy, r = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4), rng.integers(w // 12, w // 6)
            img += 70 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_synthetic")
    ap.add_argument("--n_patients", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    rng = np.random.default_rng(0)
    splits = {"train": 0.75, "val": 0.03, "test": 0.22}
    pid = {"train": 0, "test": 0, "val": 0}
    files = []
    for split, frac in splits.items():
        n = max(int(a.n_patients * frac), 2)
        for k in range(n):
            pneu = rng.random() < 0.72
            n_img = rng.integers(1, 4) if pneu else 1
            pid[split] += 1
            for j in range(n_img):
                w, h = int(rng.integers(300, 700)), int(rng.integers(250, 600))
                img = make_img(rng, pneu, w, h)
                cls = "PNEUMONIA" if pneu else "NORMAL"
                if pneu:
                    name = f"person{pid[split]}_{'bacteria' if rng.random() < 0.6 else 'virus'}_{rng.integers(1, 9999)}.jpeg"
                else:
                    name = f"{'NORMAL2-' if rng.random() < 0.3 else ''}IM-{pid[split]:04d}-000{j + 1}.jpeg"
                p = out / split / cls / name
                p.parent.mkdir(parents=True, exist_ok=True)
                if rng.random() < 0.05:
                    img = img.convert("RGB")  # a few RGB-encoded files, like the real data
                img.save(p, quality=90)
                files.append(p)
    # inject problems: an exact duplicate across splits and a near duplicate
    src = files[3]; dst = out / "test" / src.parent.name / ("dup_" + src.name)
    shutil.copy(src, dst)
    im = Image.open(files[5]).convert("L"); im = Image.fromarray(np.clip(np.asarray(im) + 3, 0, 255).astype(np.uint8))
    im.save(out / "train" / files[5].parent.name / ("near_" + files[5].name), quality=90)
    (out / "train" / "NORMAL" / "corrupt.jpeg").write_bytes(b"not an image")
    print(f"synthetic dataset with {len(files) + 3} files at {out}")


if __name__ == "__main__":
    main()
