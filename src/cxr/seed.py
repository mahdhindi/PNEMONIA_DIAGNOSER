"""Seeding / determinism helpers.

Every model in the comparison is trained through the same functions here, so
"same random seeds for all algorithms" is enforced by construction rather than
by convention.
"""
from __future__ import annotations

import os
import random

import numpy as np

# Must be set before CUDA context creation for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a handful of CUDA kernels have no deterministic variant;
    # we log rather than crash, and note it in the results JSON.
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own, reproducible RNG stream."""
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
