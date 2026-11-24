"""Train the GRU mask model.

Example (Kaggle/Colab):
    python src/train.py --noisy DATA/noisy_trainset_28spk_wav --clean DATA/clean_trainset_28spk_wav \
        --epochs 30 --out checkpoints
"""
import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PairDataset, list_pairs
from model import GRUMaskNet, loss_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy", required=True)
    p.add_argument("--clean", required=True)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--max_files", type=int, default=0, help="debug: use only N files")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    pairs = list_pairs(a.noisy, a.clean)
    random.shuffle(pairs)
    if a.max_files:
        pairs = pairs[:a.max_files]
    n_val = max(1, int(len(pairs) * a.val_fraction))
    train_ds, val_ds = PairDataset(pairs[n_val:]), PairDataset(pairs[:n_val])
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=0)

    model = GRUMaskNet().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train(); tr = 0.0
        for noisy, clean in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            wav, spec = model(noisy)
            loss = loss_fn(model, spec, clean, wav)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tr += loss.item()
        sched.step()

        model.eval(); va = 0.0
        with torch.no_grad():
            for noisy, clean in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                wav, spec = model(noisy)
                va += loss_fn(model, spec, clean, wav).item()
        tr /= max(1, len(train_dl)); va /= max(1, len(val_dl))
        print(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}  ({time.time()-t0:.0f}s)")
        torch.save(model.state_dict(), out / "last.pt")
        if va < best:
            best = va
            torch.save(model.state_dict(), out / "best.pt")
    print("best val loss:", best)


if __name__ == "__main__":
    main()
