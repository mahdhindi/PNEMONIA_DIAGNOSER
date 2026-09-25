"""One Dataset, one preprocessing pipeline, shared by every model.

Cached images are uint8 [S, S] grayscale (see prepare_data).  For every model:
  train:  RandomResizedCrop(scale 0.8-1) -> RandomRotation(+-7 deg) -> brightness/contrast jitter
  eval:   identity
then (both): [optional ablation: fixed pixel permutation] -> replicate to 3
channels -> float [0,1] -> ImageNet mean/std normalisation.

The linear baseline sees exactly the same tensors as the CNNs; it just
flattens them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FixedPixelPermutation(torch.nn.Module):
    """Ablation A: apply ONE fixed permutation of spatial positions to every
    image (train/val/test alike).  Pixel values and class balance are
    untouched; only spatial structure is destroyed.  A permutation-invariant
    model (logistic regression on pixels) is unaffected by construction."""

    def __init__(self, size: int, perm_seed: int = 1234):
        super().__init__()
        g = torch.Generator().manual_seed(perm_seed)
        self.register_buffer("perm", torch.randperm(size * size, generator=g))
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, h, w = x.shape
        return x.reshape(c, h * w)[:, self.perm].reshape(c, h, w)


def build_transform(img_size: int, train: bool, ablation: str | None = None, hflip: bool = False):
    ops = []
    if train:
        ops += [
            v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomRotation(7),
            v2.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        if hflip:
            ops.append(v2.RandomHorizontalFlip())
    if ablation == "permute":
        ops.append(FixedPixelPermutation(img_size))
    ops += [
        v2.Lambda(lambda x: x.expand(3, -1, -1).contiguous()),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(ops)


class CachedCXR(Dataset):
    def __init__(self, images: np.ndarray, rows: pd.DataFrame, transform):
        self.images = images                       # memmap or array [N, S, S]
        self.idx = rows["cache_idx"].to_numpy()
        self.labels = rows["label"].to_numpy().astype(np.float32)
        self.subtype = rows["subtype"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[self.idx[i]])).unsqueeze(0)  # [1, S, S] uint8
        return self.transform(x), self.labels[i]


def load_index(data_dir: str | Path, img_size: int, resize: str = "pad"):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "index.csv")
    suffix = "" if resize == "pad" else f"_{resize}"
    images = np.load(data_dir / "cache" / f"images_{img_size}{suffix}.npy", mmap_mode="r")
    return df, images


def subsample_train(train_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Ablation B: keep a fraction of the training *patients* (stratified by
    class, seed-fixed).  Validation and test are never touched."""
    if fraction >= 1.0:
        return train_df
    rng = np.random.default_rng(seed)
    keep = []
    for lab, part in train_df.groupby("label"):
        keys = part["patient_key"].unique().copy()
        rng.shuffle(keys)
        target = fraction * len(part)
        n = 0
        for k in keys:
            if n >= target:
                break
            keep.append(k); n += int((part["patient_key"] == k).sum())
    return train_df[train_df["patient_key"].isin(keep)]
