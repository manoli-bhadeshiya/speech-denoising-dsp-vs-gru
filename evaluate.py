"""Score noisy input, spectral subtraction, Wiener filter and the trained model on the test set.

    python src/evaluate.py --noisy DATA/noisy_testset_wav --clean DATA/clean_testset_wav \
        --ckpt checkpoints/best.pt --out results

Writes results/per_file.csv, results/summary.md and a few audio examples in results/examples/.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

from baselines import spectral_subtraction, wiener_filter
from data import SR, list_pairs, load_wav
from model import GRUMaskNet


def si_sdr(est, ref, eps=1e-8):
    ref = ref - ref.mean(); est = est - est.mean()
    s = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    return 10 * np.log10((np.sum(s ** 2) + eps) / (np.sum((est - s) ** 2) + eps))


def metrics(est, clean):
    try:
        p = pesq(SR, clean, est, "wb")
    except Exception:  # PESQ can fail on silent/very short files
        p = np.nan
    return {"pesq": p, "stoi": stoi(clean, est, SR, extended=False), "si_sdr": si_sdr(est, clean)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); (out / "examples").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if a.ckpt:
        model = GRUMaskNet().to(device)
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        model.eval()

    pairs = list_pairs(a.noisy, a.clean)
    if a.max_files:
        pairs = pairs[:a.max_files]
    rows, model_time, audio_time = [], 0.0, 0.0
    for i, (n, c) in enumerate(pairs):
        noisy, clean = load_wav(n), load_wav(c)
        L = min(len(noisy), len(clean)); noisy, clean = noisy[:L], clean[:L]
        outputs = {"noisy": noisy,
                   "spectral_sub": spectral_subtraction(noisy),
                   "wiener": wiener_filter(noisy)}
        if model is not None:
            t0 = time.time()
            with torch.no_grad():
                wav, _ = model(torch.from_numpy(noisy)[None].to(device))
            model_time += time.time() - t0; audio_time += L / SR
            outputs["gru_model"] = wav[0].cpu().numpy()
        for method, est in outputs.items():
            rows.append({"file": n.name, "method": method, **metrics(est, clean)})
            if i < a.n_examples:
                sf.write(out / "examples" / f"{n.stem}_{method}.wav", est, SR)
        if i < a.n_examples:
            sf.write(out / "examples" / f"{n.stem}_clean.wav", clean, SR)
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(pairs)} files")

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_file.csv", index=False)
    summary = df.groupby("method")[["pesq", "stoi", "si_sdr"]].mean().round(3)
    order = [m for m in ["noisy", "spectral_sub", "wiener", "gru_model"] if m in summary.index]
    summary = summary.loc[order]
    text = summary.to_markdown()
    if audio_time:
        text += f"\n\nModel real-time factor on {device}: {model_time / audio_time:.4f} (lower is faster; <1 = faster than real time)"
    (out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
