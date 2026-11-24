"""A small real-time-capable mask-based denoiser (STFT -> GRU -> mask -> iSTFT).

Causal by design (unidirectional GRU, no look-ahead beyond one STFT frame), which matters
for real-time voice products like ai-coustics' SDK. ~0.9 M parameters.
"""
import torch
import torch.nn as nn

N_FFT, HOP = 512, 128
N_BINS = N_FFT // 2 + 1


class GRUMaskNet(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.inp = nn.Sequential(nn.Linear(N_BINS, hidden), nn.LayerNorm(hidden), nn.PReLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, N_BINS), nn.Sigmoid())

    def stft(self, x):
        return torch.stft(x, N_FFT, HOP, window=self.window, return_complex=True)

    def istft(self, X, length):
        return torch.istft(X, N_FFT, HOP, window=self.window, length=length)

    def forward(self, noisy):
        """noisy: (batch, samples) -> enhanced waveform, enhanced STFT."""
        X = self.stft(noisy)                      # (B, F, T)
        feat = torch.log1p(X.abs()).transpose(1, 2)  # (B, T, F) log-magnitude features
        h, _ = self.gru(self.inp(feat))
        mask = self.out(h).transpose(1, 2)        # (B, F, T), values in [0, 1]
        Y = mask * X                              # apply mask, keep noisy phase
        return self.istft(Y, noisy.shape[-1]), Y


def safe_mag(X, eps=1e-8):
    """|X| with a finite gradient at X = 0 (silence/zero padding would otherwise give NaN)."""
    return torch.sqrt(X.real ** 2 + X.imag ** 2 + eps)


def loss_fn(model, enhanced_spec, clean, enhanced_wav):
    """Compressed-magnitude L1 (perceptually motivated) + small waveform L1 term."""
    C = model.stft(clean)
    mag_loss = (safe_mag(enhanced_spec) ** 0.3 - safe_mag(C) ** 0.3).abs().mean()
    wav_loss = (enhanced_wav - clean).abs().mean()
    return mag_loss + 0.1 * wav_loss
