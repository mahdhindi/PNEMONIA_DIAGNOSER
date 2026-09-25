"""Train and evaluate ONE model with the shared protocol.

    python -m cxr.train --model resnet50 --seed 42
    python -m cxr.train --model logreg   --seed 42 --ablation permute
    python -m cxr.train --model smallcnn --seed 42 --train_fraction 0.25

Protocol (identical for every trainable model): AdamW, weight decay 1e-4,
1 warm-up epoch then cosine decay, batch 32, fixed epoch budget, model
selection by best validation AUROC, final evaluation on the untouched test
split.  Only the learning rate differs (from-scratch 1e-3, pretrained 1e-4),
and it is recorded in the results file.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CachedCXR, build_transform, load_index, subsample_train
from .metrics import compute_metrics
from .models import ZOO, build_model, count_params
from .seed import make_generator, set_seed, worker_init_fn


def run_name(a) -> str:
    extra = ("" if a.resize == "pad" else f"__{a.resize}") + (("__" + a.tag) if a.tag else "")
    return f"{a.model}__abl-{a.ablation}__frac-{a.train_fraction:g}__seed-{a.seed}{extra}"


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    ps, ys = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logit = model(x)
        ps.append(torch.sigmoid(logit.float()).squeeze(1).cpu().numpy()); ys.append(y.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    p, y = np.concatenate(ps), np.concatenate(ys)
    return p, y, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(ZOO))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--resize", choices=["pad", "stretch"], default="pad", help="which prepared cache to use")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="override the model's default learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ablation", choices=["none", "permute"], default="none",
                    help="permute = fixed random pixel permutation applied to all splits")
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="keep this fraction of training patients (data-size ablation)")
    ap.add_argument("--no_pretrained", action="store_true", help="random init for resnet50/densenet121")
    ap.add_argument("--hflip", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not a.no_amp
    name = run_name(a)
    Path(a.results).mkdir(parents=True, exist_ok=True)
    run_dir = Path(a.runs) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    df, images = load_index(a.data, a.img_size, a.resize)
    tr = subsample_train(df[df["split"] == "train"], a.train_fraction, a.seed)
    va, te = df[df["split"] == "val"], df[df["split"] == "test"]
    abl = None if a.ablation == "none" else a.ablation
    ds_tr = CachedCXR(images, tr, build_transform(a.img_size, True, abl, a.hflip))
    ds_va = CachedCXR(images, va, build_transform(a.img_size, False, abl))
    ds_te = CachedCXR(images, te, build_transform(a.img_size, False, abl))
    g = make_generator(a.seed)
    dl_kw = dict(num_workers=a.num_workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn)
    dl_tr = DataLoader(ds_tr, a.batch_size, shuffle=True, drop_last=False, generator=g, **dl_kw)
    dl_va = DataLoader(ds_va, a.batch_size * 2, shuffle=False, **dl_kw)
    dl_te = DataLoader(ds_te, a.batch_size * 2, shuffle=False, **dl_kw)
    print(f"[{name}] device={device} train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} "
          f"train-prior={tr['label'].mean():.3f}")

    # ----------------------------------------------------------------- model
    model = build_model(a.model, a.img_size, pretrained=not a.no_pretrained).to(device)
    n_params = count_params(model)
    spec = ZOO[a.model]
    lr = a.lr if a.lr is not None else spec["lr"]
    history, train_time, best_auc, best_epoch, best_state = [], 0.0, -1.0, -1, None
    t_wall = time.perf_counter()

    if not spec["trainable"]:
        model.fit(float(tr["label"].mean()))
        best_state = copy.deepcopy(model.state_dict())
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.weight_decay)
        steps_per_epoch = max(len(dl_tr), 1)
        total = a.epochs * steps_per_epoch
        warm = steps_per_epoch

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            prog = (step - warm) / max(total - warm, 1)
            return 0.5 * (1 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = torch.amp.GradScaler(enabled=amp)
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(1, a.epochs + 1):
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            tot, n = 0.0, 0
            for x, y in tqdm(dl_tr, desc=f"epoch {epoch}/{a.epochs}", leave=False):
                x = x.to(device, non_blocking=True); y = y.to(device).unsqueeze(1)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = loss_fn(model(x).float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * x.shape[0]; n += x.shape[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            ep_time = time.perf_counter() - t0
            train_time += ep_time
            p, y, _ = predict(model, dl_va, device, amp)
            m = compute_metrics(y, p, n_boot=0)
            history.append(dict(epoch=epoch, train_loss=tot / max(n, 1), val_auroc=m["auroc"],
                                val_acc=m["accuracy"], val_macro_f1=m["macro_f1"], epoch_time_s=ep_time,
                                lr=sched.get_last_lr()[0]))
            print(f"  ep{epoch:02d} loss={tot / max(n, 1):.4f} val_auroc={m['auroc']:.4f} "
                  f"val_acc={m['accuracy']:.4f} ({ep_time:.0f}s)")
            score = m["auroc"] if not math.isnan(m["auroc"]) else m["accuracy"]
            if score > best_auc:
                best_auc, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    wall_time = time.perf_counter() - t_wall

    # ------------------------------------------------------------- evaluate
    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best.pt")
    ckpt_mb = (run_dir / "best.pt").stat().st_size / 1e6
    p_va, y_va, _ = predict(model, dl_va, device, amp)
    p_te, y_te, secs = predict(model, dl_te, device, amp)
    val_m = compute_metrics(y_va, p_va, subtype=ds_va.subtype)
    test_m = compute_metrics(y_te, p_te, subtype=ds_te.subtype)
    np.save(run_dir / "test_probs.npy", np.stack([y_te, p_te], 1))

    res = dict(
        run=name, model=a.model, family=spec["family"], seed=a.seed, ablation=a.ablation,
        train_fraction=a.train_fraction, pretrained=(not a.no_pretrained) and a.model in ("resnet50", "densenet121"),
        n_train=len(ds_tr), n_val=len(ds_va), n_test=len(ds_te),
        params=n_params, params_M=round(n_params / 1e6, 3), checkpoint_mb=round(ckpt_mb, 2),
        train_time_s=round(train_time, 1), wall_time_s=round(wall_time, 1), epochs=a.epochs if spec["trainable"] else 0,
        best_epoch=best_epoch, infer_ms_per_img=round(1000 * secs / max(len(ds_te), 1), 3),
        peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else None,
        resize=a.resize, tag=a.tag, val=val_m, test=test_m, history=history,
        probs=dict(val_y=y_va.astype(int).tolist(), val_p=[round(float(x), 5) for x in p_va],
                   test_y=y_te.astype(int).tolist(), test_p=[round(float(x), 5) for x in p_te]),
        config=dict(img_size=a.img_size, resize=a.resize, batch_size=a.batch_size, lr=lr, weight_decay=a.weight_decay,
                    optimizer="AdamW", schedule="1 warm-up epoch + cosine", loss="BCEWithLogits",
                    selection="best val AUROC", amp=amp, hflip=a.hflip,
                    augmentation="RandomResizedCrop(0.8-1) + RandomRotation(7) + brightness/contrast 0.15",
                    normalisation="ImageNet mean/std on grayscale replicated to 3 channels"),
        env=dict(torch=torch.__version__, device=torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu",
                 python=platform.python_version(), deterministic="use_deterministic_algorithms(warn_only=True), cudnn.deterministic"),
    )
    with open(Path(a.results) / f"{name}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{name}] TEST auroc={test_m['auroc']:.4f} acc={test_m['accuracy']:.4f} macroF1={test_m['macro_f1']:.4f} "
          f"recall={test_m['recall']:.4f} spec={test_m['specificity']:.4f} | params={n_params / 1e6:.2f}M "
          f"train={train_time:.0f}s -> {a.results}/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
