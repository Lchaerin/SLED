"""
Binaural audio preprocessor: raw waveform → 5-channel feature map.

Input : [B, 2, T_samples]   stereo waveform at 48 kHz
Output: [B, 5, 64, T_frames] feature tensor

Channels (per SLED_MODEL.md note on IPD wrapping fix):
  0  L mel-spectrogram   log( mel_bank @ |STFT_L| )
  1  R mel-spectrogram   log( mel_bank @ |STFT_R| )
  2  cos(IPD)            mel-weighted cosine of interaural phase difference
  3  sin(IPD)            mel-weighted sine of interaural phase difference
  4  ILD                 mel-weighted interaural level difference

STFT parameters (matching annotation frame hop = 20 ms @ 48 kHz):
  n_fft      = 1024        → F = 513 frequency bins
  win_length = 960         → 20 ms analysis window
  hop_length = 960         → 20 ms hop (= annotation hop)
  center     = False       → T_frames = (T - n_fft) // hop + 1
  For T = N*hop : T_frames = N (exact match to annotation count)

IPD/ILD are mel-averaged (divided by mel filter sums) so they remain
in physically meaningful units (radians and dB-like).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class BinauralPreprocessor(nn.Module):

    def __init__(
        self,
        sample_rate: int = 48_000,
        n_fft: int = 1024,
        hop_length: int = 960,    # 20 ms @ 48 kHz  (= annotation hop)
        win_length: int = 960,    # 20 ms window
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: float = 22_050.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Hann window — kept as buffer for device portability
        self.register_buffer("window", torch.hann_window(win_length))

        # Mel filter bank: (F, n_mels)  F = n_fft//2+1
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="htk",
        )  # (F, n_mels)
        self.register_buffer("mel_fb", mel_fb)

        # Per-mel-band sum of filter coefficients (for averaging IPD/ILD)
        mel_sums = mel_fb.sum(dim=0).clamp(min=1e-6)  # (n_mels,)
        self.register_buffer("mel_sums", mel_sums)

    # ------------------------------------------------------------------
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, 2, T)  float32, stereo waveform
        Returns:
            feat:  (B, 5, n_mels, T_frames) float32
                   T_frames = T // hop_length  (exact, trimmed from STFT output)
        """
        B, _, T_in = audio.shape
        T_out = T_in // self.hop_length    # expected output frames
        L, R = audio[:, 0], audio[:, 1]   # each (B, T)

        stft_l = self._stft(L)   # (B, F, T_f) complex  T_f may be T_out+1 with center=True
        stft_r = self._stft(R)
        # Trim to exact T_out frames
        stft_l = stft_l[..., :T_out]
        stft_r = stft_r[..., :T_out]

        mag_l = stft_l.abs()     # (B, F, T_f)
        mag_r = stft_r.abs()
        EPS = 1e-7

        # --- mel spectrograms ---
        # mel_fb: (F, n_mels) → einsum: (B,F,T)×(F,M) = (B,M,T)
        mel_l = torch.einsum("bft,fm->bmt", mag_l, self.mel_fb)
        mel_r = torch.einsum("bft,fm->bmt", mag_r, self.mel_fb)
        log_mel_l = torch.log(mel_l.clamp(min=EPS))
        log_mel_r = torch.log(mel_r.clamp(min=EPS))

        # --- IPD (cos + sin, mel-averaged) ---
        cross = stft_l * stft_r.conj()                     # (B, F, T_f) complex
        ipd   = cross.angle()                              # (B, F, T_f) ∈ [-π, π]
        # cos_ipd_mel[b, m, t] = Σ_f mel_fb[f,m] * cos(ipd[b,f,t]) / mel_sums[m]
        cos_ipd_mel = torch.einsum("bft,fm->bmt", torch.cos(ipd), self.mel_fb) / self.mel_sums[None, :, None]
        sin_ipd_mel = torch.einsum("bft,fm->bmt", torch.sin(ipd), self.mel_fb) / self.mel_sums[None, :, None]

        # --- ILD (mel-averaged) ---
        ild_full = torch.log(mag_l.clamp(min=EPS)) - torch.log(mag_r.clamp(min=EPS))  # (B,F,T)
        ild_mel  = torch.einsum("bft,fm->bmt", ild_full, self.mel_fb) / self.mel_sums[None, :, None]

        # Stack → (B, 5, n_mels, T_frames)
        feat = torch.stack([log_mel_l, log_mel_r, cos_ipd_mel, sin_ipd_mel, ild_mel], dim=1)
        return feat

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """STFT on batched audio (B, T) → (B, F, T_f) complex.
        Uses center=True so T_f = T//hop + 1; caller trims to T//hop."""
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
