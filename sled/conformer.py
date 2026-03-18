"""
Conformer block (Gulati et al., 2020) — Macaron-style FFN sandwiching.

Structure per block:
    x = x + 0.5 * FFN1(LayerNorm(x))
    x = x + MHSA(LayerNorm(x))
    x = x + ConvModule(LayerNorm(x))
    x = x + 0.5 * FFN2(LayerNorm(x))

Input/output shape: (B, T, d_model)

ConvModule:
    LayerNorm → Pointwise(d→2d) → GLU → DepthwiseConv(k=31) → BN → SiLU
              → Pointwise(d→d) → Dropout
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, ffn_dim)
        self.fc2  = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc2(self.drop(F.silu(self.fc1(x))))
        return self.drop(x)


class ConvModule(nn.Module):
    """Conformer convolution sub-module."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.norm   = nn.LayerNorm(d_model)
        self.pw1    = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)   # pointwise expand
        # GLU halves channels back to d_model
        self.dw     = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.bn     = nn.BatchNorm1d(d_model)
        self.pw2    = nn.Conv1d(d_model, d_model, kernel_size=1)       # pointwise project
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        residual = x
        x = self.norm(x).transpose(1, 2)     # (B, d, T)
        x = F.glu(self.pw1(x), dim=1)        # (B, d, T)  GLU halves channels
        x = F.silu(self.bn(self.dw(x)))       # (B, d, T)
        x = self.drop(self.pw2(x))
        return x.transpose(1, 2)             # (B, T, d)


class ConformerBlock(nn.Module):
    """
    Full Conformer block (Macaron-style).
    Params ≈ 2×FFN + MHSA + ConvModule + 4×LayerNorm
           ≈ 2×(d×ffn + ffn×d) + 4×d² + d×k + 4×(2d) ≈ 4dF + 4d² + dk
    For d=128, F=512, k=31: ≈ 262K + 65K + 4K ≈ 331K per block
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 512,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1  = FeedForward(d_model, ffn_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mhsa  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv  = ConvModule(d_model, conv_kernel, dropout)
        self.ffn2  = FeedForward(d_model, ffn_dim, dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, d)
        x = x + 0.5 * self.ffn1(x)

        x_norm = self.norm2(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.norm_out(x)
