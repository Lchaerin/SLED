"""
Building blocks for the SLED encoder.

ConvBlock        : Conv2d → BN → ReLU → MaxPool(freq-only)
SEBlock          : Squeeze-and-Excitation channel attention
BiFPNLayer       : One top-down + bottom-up BiFPN fusion pass
BiFPNNeck        : 2-layer BiFPN over three ConvBlock scales
TemporalAttPool  : Attention-weighted temporal pooling [B,T,d] → [B,d]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
# ConvBlock
# ──────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU → MaxPool(freq axis only)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))   # freq ÷2, time unchanged
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        return self.pool(self.drop(F.relu(self.bn(self.conv(x)))))


# ──────────────────────────────────────────────────────────
# SEBlock
# ──────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        s = x.mean(dim=(2, 3))               # (B, C)  global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))       # (B, C)
        return x * s[:, :, None, None]


# ──────────────────────────────────────────────────────────
# BiFPN
# ──────────────────────────────────────────────────────────

class _BiFPNLayer(nn.Module):
    """
    Single BiFPN layer over 3 feature maps P3, P4, P5 of equal shape (B, d, T).

    Fast-normalized weighted fusion (Tan et al., 2020):
        fused = Σ_i (w_i / (Σ_j w_j + ε)) * P_i
    followed by a depthwise-separable Conv1d for feature mixing.
    """

    def __init__(self, d: int = 128, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # Learnable fusion weights (always ≥ 0 after relu)
        # Top-down: P4_td needs 2 inputs, P3_td needs 2 inputs
        # Bottom-up: P4_bu needs 3 inputs, P5_bu needs 3 inputs
        self.w_td4  = nn.Parameter(torch.ones(2))   # (P4, P5)
        self.w_td3  = nn.Parameter(torch.ones(2))   # (P3, P4_td)
        self.w_bu4  = nn.Parameter(torch.ones(3))   # (P4, P4_td, P3_bu)
        self.w_bu5  = nn.Parameter(torch.ones(3))   # (P5, P5_td, P4_bu)

        # Depthwise-separable conv after each fusion (causal: left-only padding)
        def _dw_sep(d):
            return nn.Sequential(
                nn.ConstantPad1d((2, 0), 0),                           # causal left pad
                nn.Conv1d(d, d, kernel_size=3, padding=0, groups=d, bias=False),
                nn.Conv1d(d, d, kernel_size=1, bias=False),
                nn.BatchNorm1d(d),
                nn.SiLU(),
            )
        self.conv_td4 = _dw_sep(d)
        self.conv_td3 = _dw_sep(d)
        self.conv_bu4 = _dw_sep(d)
        self.conv_bu5 = _dw_sep(d)

    @staticmethod
    def _fuse(feats: list[torch.Tensor], weights: torch.Tensor, eps: float) -> torch.Tensor:
        w = F.relu(weights)
        w = w / (w.sum() + eps)
        return sum(w[i] * feats[i] for i in range(len(feats)))

    def forward(
        self,
        p3: torch.Tensor,   # (B, d, T)
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- top-down ---
        p5_td = p5                                                  # identity
        p4_td = self.conv_td4(self._fuse([p4, p5_td], self.w_td4, self.eps))
        p3_td = self.conv_td3(self._fuse([p3, p4_td], self.w_td3, self.eps))

        # --- bottom-up ---
        p3_bu = p3_td                                               # bottom unchanged
        p4_bu = self.conv_bu4(self._fuse([p4, p4_td, p3_bu], self.w_bu4, self.eps))
        p5_bu = self.conv_bu5(self._fuse([p5, p5_td, p4_bu], self.w_bu5, self.eps))

        return p3_bu, p4_bu, p5_bu


class BiFPNNeck(nn.Module):
    """
    2-layer BiFPN over the 3 ConvBlock intermediate outputs.

    Inputs:  P3 (B, c3, F3, T), P4 (B, c4, F4, T), P5 (B, c5, F5, T)
             (F3=32, F4=16, F5=8 after 3 MaxPool(freq))
    Process:
      1. Project each Pk to (B, d, T) via freq-flatten + Linear
      2. Apply 2× _BiFPNLayer
      3. Sum all three outputs → (B, d, T)
    """

    def __init__(self, d: int = 256, n_layers: int = 2, in_channels: tuple = (64, 128, 256)):
        super().__init__()
        # Projection: frequency-average then pointwise conv (channel projection)
        # Frequency averaging eliminates the F dimension cheaply.
        # P3: (B,c3,32,T) → avg_freq → (B,c3,T) → Conv1d(c3,d,1)
        # P4: (B,c4,16,T) → avg_freq → (B,c4,T) → Conv1d(c4,d,1)
        # P5: (B,c5,8,T)  → avg_freq → (B,c5,T) → Conv1d(c5,d,1)
        self.proj3 = nn.Conv1d(in_channels[0], d, kernel_size=1, bias=False)
        self.proj4 = nn.Conv1d(in_channels[1], d, kernel_size=1, bias=False)
        self.proj5 = nn.Conv1d(in_channels[2], d, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm1d(d)
        self.bn4   = nn.BatchNorm1d(d)
        self.bn5   = nn.BatchNorm1d(d)

        self.layers = nn.ModuleList([_BiFPNLayer(d) for _ in range(n_layers)])

    def forward(
        self,
        p3: torch.Tensor,   # (B, c3, F3, T)
        p4: torch.Tensor,   # (B, c4, F4, T)
        p5: torch.Tensor,   # (B, c5, F5, T)
    ) -> torch.Tensor:

        def _proj(x, proj, bn):
            x = x.mean(dim=2)          # freq-average: (B, C, T)
            return F.silu(bn(proj(x))) # channel project: (B, d, T)

        f3 = _proj(p3, self.proj3, self.bn3)
        f4 = _proj(p4, self.proj4, self.bn4)
        f5 = _proj(p5, self.proj5, self.bn5)

        for layer in self.layers:
            f3, f4, f5 = layer(f3, f4, f5)

        return f3 + f4 + f5   # (B, d, T)


# ──────────────────────────────────────────────────────────
# Temporal Attention Pooling
# ──────────────────────────────────────────────────────────

class TemporalAttPool(nn.Module):
    """
    Attention-weighted mean over the time axis.
    Input : (B, T, d)
    Output: (B, d)
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        w = self.score(x).softmax(dim=1)    # (B, T, 1)
        return (x * w).sum(dim=1)            # (B, d)
