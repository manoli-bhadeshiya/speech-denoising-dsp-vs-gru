"""Data loading for VoiceBank-DEMAND (noisy/clean pairs with identical file names)."""
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

SR = 16000


def load_wav(path, sr=SR):
    """Read a wav file as mono float32 and resample it to `sr` (VoiceBank is 48 kHz)."""
    x, file_sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def list_pairs(noisy_dir, clean_dir):
    noisy_dir, clean_dir = Path(noisy_dir), Path(clean_dir)
    pairs = []
    for n in sorted(noisy_dir.glob("*.wav")):
        c = clean_dir / n.name
        if c.exists():
            pairs.append((n, c))
    if not pairs:
        raise FileNotFoundError(f"No matching wav pairs in {noisy_dir} and {clean_dir}")
    return pairs


class PairDataset(Dataset):
    """Returns random fixed-length crops (training) of noisy/clean pairs.

    Files are resampled once and cached in RAM (~10 h of 16 kHz float32 is ~2.3 GB).
    """

    def __init__(self, pairs, seg_seconds=2.0, sr=SR, cache=True):
        self.pairs = pairs
        self.seg = int(seg_seconds * sr)
        self.sr = sr
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.pairs)

    def _get(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        n, c = self.pairs[i]
        noisy, clean = load_wav(n, self.sr), load_wav(c, self.sr)
        L = min(len(noisy), len(clean))
        item = (noisy[:L], clean[:L])
        if self.cache is not None:
            self.cache[i] = item
        return item

    def __getitem__(self, i):
        noisy, clean = self._get(i)
        L = len(noisy)
        if L >= self.seg:
            start = np.random.randint(0, L - self.seg + 1)
            noisy, clean = noisy[start:start + self.seg], clean[start:start + self.seg]
        else:
            pad = self.seg - L
            noisy, clean = np.pad(noisy, (0, pad)), np.pad(clean, (0, pad))
        return torch.from_numpy(noisy), torch.from_numpy(clean)
