"""Classic DSP baselines: spectral subtraction and a decision-directed Wiener filter.

Both estimate the noise spectrum from the first frames of the file (assumed speech-free),
which is the textbook assumption. Discussing where that assumption breaks is a good point
for your README.
"""
import numpy as np
from scipy.signal import stft, istft

N_FFT, HOP = 512, 128


def _stft(x, sr):
    _, _, X = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    return X


def _istft(X, sr, length):
    _, x = istft(X, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")
    x = x[:length]
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    return x.astype(np.float32)


def _noise_psd(P, noise_frames):
    return P[:, :noise_frames].mean(axis=1, keepdims=True) + 1e-10


def spectral_subtraction(x, sr=16000, noise_frames=20, alpha=2.0, beta=0.02):
    """Power spectral subtraction with over-subtraction factor `alpha` and spectral floor `beta`."""
    X = _stft(x, sr)
    P = np.abs(X) ** 2
    N = _noise_psd(P, noise_frames)
    S = np.maximum(P - alpha * N, beta * P)
    Y = np.sqrt(S) * np.exp(1j * np.angle(X))  # keep the noisy phase
    return _istft(Y, sr, len(x))


def wiener_filter(x, sr=16000, noise_frames=20, dd_alpha=0.98, gain_floor=0.05):
    """Wiener filter with the decision-directed a-priori SNR estimate (Ephraim & Malah, 1984)."""
    X = _stft(x, sr)
    P = np.abs(X) ** 2
    N = _noise_psd(P, noise_frames)[:, 0]
    G = np.zeros_like(P)
    prev_S = np.zeros(P.shape[0])
    for t in range(P.shape[1]):
        post_snr = P[:, t] / N
        prio_snr = dd_alpha * prev_S / N + (1 - dd_alpha) * np.maximum(post_snr - 1, 0)
        g = np.maximum(prio_snr / (1 + prio_snr), gain_floor)
        G[:, t] = g
        prev_S = (g ** 2) * P[:, t]
    return _istft(G * X, sr, len(x))
