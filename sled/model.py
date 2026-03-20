"""
SLED — Sound Localization & Event Detection model (DETR-style).

Architecture (~9M parameters):
  Input  : (B, 5, 64, T)   5-channel feature from BinauralPreprocessor
  Output : per-frame per-slot detections + source_embed (B, S, 256)

Forward path:
  ConvBlock×3  (5→64→128→256 ch, freq-MaxPool×3 → freq 64→32→16→8)
    ↓  saves P3, P4, P5 for BiFPN
  SEBlock      (channel attention on 256-ch output)
  Flatten+Proj (B, 256×8, T) → Conv1d(2048, 256) → (B, 256, T)
  Conformer×4  (B, T, 256) → (B, T, 256)
  BiFPN Neck   (P3, P4, P5) → (B, 256, T) — added to Conformer output
  ↓ enc_out (B, T, 256)
  SinusoidalPE + SlotDecoder  (4 DETR decoder layers)
    ↓  list of (B, S, 256) per layer
  Per-frame fusion: enc_out.unsqueeze(2) + slot_feat.unsqueeze(1)  → (B, T, S, 256)
  DetectionHeads (2-layer MLP per head)
    ┌ class_head  → (B, T, S, 301)
    ├ doa_head    → (B, T, S, 3)   L2-norm unit vector
    ├ loud_head   → (B, T, S)      dB regression
    └ conf_head   → (B, T, S)      confidence logit

source_embed = last-layer slot features (B, S, 256)  — for VLA / CLAP
Auxiliary losses from intermediate decoder layers (training only).

Parameter budget: ~9M
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .blocks    import ConvBlock, SEBlock, BiFPNNeck, TemporalAttPool
from .conformer import ConformerBlock


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SLEDConfig:
    n_classes:       int   = 300        # foreground classes (class 300 = empty)
    n_slots:         int   = 5          # max simultaneous sources
    d_model:         int   = 256
    n_mels:          int   = 64
    in_channels:     int   = 5          # L-mel, R-mel, cos-IPD, sin-IPD, ILD
    # ConvBlocks
    conv_channels:   tuple = (64, 128, 256)
    # Conformer
    n_conformer:     int   = 4
    n_heads:         int   = 8
    ffn_dim:         int   = 512
    conv_kernel:     int   = 31
    dropout:         float = 0.1
    # BiFPN
    n_bifpn:         int   = 2
    # Slot decoder (DETR-style)
    n_decoder_layers: int  = 4
    # CLAP projection (optional)
    clap_dim:        int   = 512


DEFAULT_CFG = SLEDConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """
    Standard sinusoidal positional encoding added to (B, T, d) sequences.

    Supports arbitrary T at inference time.
    Encoding is registered as a buffer (not learned).
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                    # (max_len, d)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)                        # (max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        T = x.size(1)
        return self.dropout(x + self.pe[:T].unsqueeze(0))     # (B, T, d)


# ──────────────────────────────────────────────────────────────────────────────
# DETR-style Slot Decoder
# ──────────────────────────────────────────────────────────────────────────────

class SlotDecoderLayer(nn.Module):
    """
    One DETR decoder layer with pre-norm:
      1. Self-attention:  slots attend to each other  (B, S, d)
      2. Cross-attention: slots (query) attend to encoder memory (B, T, d)
      3. FFN:             Linear→GELU→Dropout→Linear→Dropout
    All sub-layers use pre-norm + residual.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        slots:  torch.Tensor,   # (B, S, d)
        memory: torch.Tensor,   # (B, T, d)  encoder output with PE
    ) -> torch.Tensor:
        # 1. Pre-norm self-attention
        q = k = self.norm1(slots)
        slots = slots + self.self_attn(q, k, slots, need_weights=False)[0]

        # 2. Pre-norm cross-attention
        q2 = self.norm2(slots)
        slots = slots + self.cross_attn(q2, memory, memory, need_weights=False)[0]

        # 3. Pre-norm FFN
        slots = slots + self.ffn(self.norm3(slots))

        return slots  # (B, S, d)


class SlotDecoder(nn.Module):
    """
    Frame-local slot decoder with learnable query embeddings.

    Each frame's slots cross-attend ONLY to that frame's encoder output,
    making DOA predictions audio-driven and the model strictly causal.

    Input:
        memory  (B, T, d)   encoder output (PE already added externally)
    Output:
        list of (B, T, S, d) tensors — one per decoder layer (including last)
        Used to compute auxiliary losses on intermediate layers.
    """

    def __init__(self, n_slots: int, d_model: int, n_layers: int,
                 n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.slot_queries = nn.Embedding(n_slots, d_model)
        self.layers = nn.ModuleList([
            SlotDecoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, memory: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            memory: (B, T, d) encoder memory with PE
        Returns:
            list of (B, T, S, d), length = n_layers

        Each frame t cross-attends to memory[:, t:t+1, :] only, so DOA is
        driven by the current frame's binaural features (ILD/IPD), not the
        averaged clip-level context.
        """
        B, T, d = memory.shape
        S = self.slot_queries.num_embeddings

        # Reshape memory: (B, T, d) → (B*T, 1, d)  — each frame is its own sequence
        mem_flat = memory.reshape(B * T, 1, d)

        # Expand learnable slot queries to (B*T, S, d)
        slots = self.slot_queries.weight            # (S, d)
        slots = slots.unsqueeze(0).expand(B * T, -1, -1).clone()  # (B*T, S, d)

        # Flash Attention requires seq_len >= block_size (typically ≥ 16).
        # With S=5 slots, use the math backend to avoid cudaErrorInvalidConfiguration.
        all_slots = []
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                slots = layer(slots, mem_flat)                # (B*T, S, d)
                all_slots.append(slots.view(B, T, S, d))

        return all_slots   # list of (B, T, S, d)


# ──────────────────────────────────────────────────────────────────────────────
# CLAP Projection Head  (SLED → CLAP audio space)
# ──────────────────────────────────────────────────────────────────────────────

class CLAPProjectionHead(nn.Module):
    """Project SLED 256d source embeddings to CLAP 512d space."""

    def __init__(self, sled_dim: int = 256, clap_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sled_dim, 256),
            nn.GELU(),
            nn.Linear(256, clap_dim),
        )

    def forward(self, source_embeds: torch.Tensor) -> torch.Tensor:
        # source_embeds: (B, S, 256) → (B, S, 512) L2-normalized
        return F.normalize(self.proj(source_embeds), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# SLED Encoder
# ──────────────────────────────────────────────────────────────────────────────

class SLEDEncoder(nn.Module):
    """
    Encoder: feature extraction (Conv+SE+BiFPN) + temporal modeling (Conformer).
    Returns (B, T, d_model) for use by the slot decoder and detection heads.
    """

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        c1, c2, c3 = cfg.conv_channels          # 64, 128, 256

        # Conv blocks  (5→64→128→256 channels)
        self.cb1 = ConvBlock(cfg.in_channels, c1, dropout=0.0)
        self.cb2 = ConvBlock(c1, c2, dropout=0.0)
        self.cb3 = ConvBlock(c2, c3, dropout=0.0)

        # Channel attention
        self.se  = SEBlock(c3, reduction=16)

        # Freq flatten + projection:
        # After 3 freq-MaxPool on 64 mels → freq = 64 / 8 = 8
        # Flatten: (B, c3 * 8, T) = (B, 256*8, T) = (B, 2048, T) → (B, 256, T)
        freq_after_pool = cfg.n_mels // (2 ** 3)           # 64 → 8
        self.freq_proj = nn.Conv1d(c3 * freq_after_pool, cfg.d_model, kernel_size=1, bias=False)
        self.freq_bn   = nn.BatchNorm1d(cfg.d_model)

        # Conformer stack (causal: no future-frame leakage)
        self.conformers = nn.ModuleList([
            ConformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim,
                conv_kernel=cfg.conv_kernel,
                dropout=cfg.dropout,
                causal=True,
            )
            for _ in range(cfg.n_conformer)
        ])

        # BiFPN neck — receives (c1, c2, c3) = (64, 128, 256) channel maps
        self.bifpn = BiFPNNeck(d=cfg.d_model, n_layers=cfg.n_bifpn,
                               in_channels=cfg.conv_channels)

        # Post-fusion norm
        self.post_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        feat: torch.Tensor,                          # (B, 5, 64, T)
    ) -> torch.Tensor:
        """
        Returns:
            enc_out  (B, T, d)   full temporal feature sequence
        """
        # ── Conv blocks + save intermediates for BiFPN
        p3 = self.cb1(feat)     # (B, 64, 32, T)
        p4 = self.cb2(p3)       # (B, 128, 16, T)
        p5 = self.cb3(p4)       # (B, 256,  8, T)

        # ── Channel attention
        p5_se = self.se(p5)     # (B, 256, 8, T)

        # ── Flatten freq → temporal sequence
        B, C, Freq, T = p5_se.shape
        x = p5_se.reshape(B, C * Freq, T)              # (B, 2048, T)
        x = F.silu(self.freq_bn(self.freq_proj(x)))    # (B, 256, T)
        x = x.transpose(1, 2)                          # (B, T, 256)

        # ── Conformer
        for block in self.conformers:
            x = block(x)                               # (B, T, 256)

        # ── BiFPN: fuse multi-scale conv features
        bifpn_feat = self.bifpn(p3, p4, p5)            # (B, 256, T)
        bifpn_feat = bifpn_feat.transpose(1, 2)        # (B, T, 256)

        enc_out = self.post_norm(x + bifpn_feat)       # (B, T, 256)
        return enc_out


# ──────────────────────────────────────────────────────────────────────────────
# Detection Heads  (2-layer MLP per head)
# ──────────────────────────────────────────────────────────────────────────────

class DetectionHeads(nn.Module):
    """
    Per-frame per-slot 2-layer MLP detection heads.

    Input:  (B, T, S, d)  — fused per-frame slot features
    Output: dict of tensors with shapes (B, T, S, *)
    """

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        d     = cfg.d_model          # 256
        n_cls = cfg.n_classes + 1    # 301

        self.class_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, n_cls),
        )
        self.doa_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 3),
        )
        self.loud_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, T, S, d)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict:
            class_logits  (B, T, S, 301)
            doa_vec       (B, T, S, 3)    unit vector
            loudness      (B, T, S)       dB
            confidence    (B, T, S)       raw logit
        """
        cls_logits = self.class_head(x)                              # (B,T,S,301)
        doa_raw    = self.doa_head(x)                                # (B,T,S,3)
        doa_vec    = F.normalize(doa_raw, p=2, dim=-1, eps=1e-6)    # unit vector
        loudness   = self.loud_head(x).squeeze(-1)                   # (B,T,S)
        confidence = self.conf_head(x).squeeze(-1)                   # (B,T,S) raw logit

        return {
            "class_logits": cls_logits,
            "doa_vec":      doa_vec,
            "loudness":     loudness,
            "confidence":   confidence,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Full SLED Model
# ──────────────────────────────────────────────────────────────────────────────

class SLED(nn.Module):
    """
    Full SLED model with DETR-style slot decoder.

    The critical fix vs the previous version: each slot receives a *distinct*
    input via learnable slot query embeddings that cross-attend to the encoder
    memory, eliminating the bug where all slots received identical input.

    forward(feat) accepts pre-computed 5-channel features (B, 5, 64, T).
    Use BinauralPreprocessor to convert raw audio to features.

    Training output:
        {
            "class_logits": (B, T, S, 301),
            "doa_vec":      (B, T, S, 3),
            "loudness":     (B, T, S),
            "confidence":   (B, T, S),
            "source_embed": (B, S, 256),   # last decoder layer slot features
            "sce_vec":      (B, T, S, 3),  # SCE auxiliary (training only)
            "aux": [                        # intermediate decoder layers (training only)
                {"class_logits": ..., "doa_vec": ..., "loudness": ..., "confidence": ...},
                ...
            ]
        }

    Per-frame decoding:
        Slot features (B, S, d) from each decoder layer are broadcast over the
        time axis and fused with the encoder output (B, T, d) to give
        (B, T, S, d) — detection heads then operate per-frame per-slot.

    Notes:
        - In training mode: SCE auxiliary loss branch and aux decoder outputs active.
        - In eval mode: only the final decoder layer predictions are returned.
    """

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model   # 256

        self.encoder     = SLEDEncoder(cfg)
        self.pos_enc     = SinusoidalPE(d, dropout=cfg.dropout)
        self.slot_decoder = SlotDecoder(
            n_slots  = cfg.n_slots,
            d_model  = d,
            n_layers = cfg.n_decoder_layers,
            n_heads  = cfg.n_heads,
            ffn_dim  = cfg.ffn_dim,
            dropout  = cfg.dropout,
        )

        # Norm after per-frame + slot feature fusion
        self.fuse_norm = nn.LayerNorm(d)

        self.heads       = DetectionHeads(cfg)
        self.clap_head   = CLAPProjectionHead(d, cfg.clap_dim)

        # SCE auxiliary (training only): predicted source coordinate vector
        # (no extra parameters needed — computed from doa_vec + loudness)

    def _apply_heads(
        self,
        enc_out:   torch.Tensor,   # (B, T, d)
        slot_feat: torch.Tensor,   # (B, T, S, d)  — frame-local slot features
    ) -> dict[str, torch.Tensor]:
        """Fuse encoder output with slot features and apply detection heads."""
        # (B, T, 1, d) + (B, T, S, d) → (B, T, S, d)
        per_frame = enc_out.unsqueeze(2) + slot_feat
        per_frame = self.fuse_norm(per_frame)             # (B, T, S, d)
        return self.heads(per_frame)

    def forward(
        self,
        feat: torch.Tensor,      # (B, 5, 64, T)
    ) -> dict[str, torch.Tensor]:
        """
        Process a window of T frames.

        Returns per-frame predictions:
            class_logits  (B, T, S, 301)
            doa_vec       (B, T, S, 3)
            loudness      (B, T, S)
            confidence    (B, T, S)
            source_embed  (B, S, 256)   last decoder layer slot features
            sce_vec       (B, T, S, 3)  only during training (SCE aux)
            aux           list of dicts only during training
        """
        # ── Encoder ─────────────────────────────────────────────────────
        enc_out = self.encoder(feat)               # (B, T, d)

        # ── Add sinusoidal PE for slot cross-attention ───────────────────
        memory = self.pos_enc(enc_out)             # (B, T, d)

        # ── Frame-local slot decoder ─────────────────────────────────────
        # Returns one (B, T, S, d) tensor per decoder layer
        all_slot_feats = self.slot_decoder(memory) # list of (B, T, S, d)

        # ── Final prediction (last decoder layer) ────────────────────────
        final_slot = all_slot_feats[-1]            # (B, T, S, d)
        final_pred = self._apply_heads(enc_out, final_slot)

        # source_embed: mean over T → (B, S, d) for clip-level identity
        source_embed = final_slot.mean(dim=1)      # (B, S, d)

        out = {
            "class_logits": final_pred["class_logits"],
            "doa_vec":      final_pred["doa_vec"],
            "loudness":     final_pred["loudness"],
            "confidence":   final_pred["confidence"],
            "source_embed": source_embed,          # (B, S, d) — clip-averaged
        }

        # ── Training-only outputs ────────────────────────────────────────
        if self.training:
            # SCE: unit_doa scaled by linear loudness amplitude
            loudness  = final_pred["loudness"]
            doa_vec   = final_pred["doa_vec"]
            loud_lin  = (loudness / 20.0).exp()                 # (B, T, S)
            out["sce_vec"] = doa_vec * loud_lin.unsqueeze(-1)   # (B, T, S, 3)

            # Auxiliary predictions from intermediate decoder layers
            aux = []
            for slot_feat in all_slot_feats[:-1]:    # all but last, each (B,T,S,d)
                aux.append(self._apply_heads(enc_out, slot_feat))
            out["aux"] = aux

        return out

    # ── CLAP embed (optional, call explicitly) ───────────────────────
    def get_clap_embeds(self, source_embed: torch.Tensor) -> torch.Tensor:
        """(B, S, 256) → (B, S, 512) CLAP-aligned embeddings."""
        return self.clap_head(source_embed)

    # ── Inference interface ───────────────────────────────────────────
    @torch.inference_mode()
    def predict(
        self,
        feat: torch.Tensor,                     # (B, 5, 64, T)
        conf_threshold: float = 0.5,
        class_names: dict[int, str] | None = None,
    ) -> list[list[dict]]:
        """
        Run inference and return filtered detections per batch item.

        Returns list of lists of dicts:
            [
              [  # batch item 0
                {class_id, class_name, azimuth_rad, elevation_rad,
                 loudness_db, confidence, source_embed},
                ...
              ],
              ...
            ]
        """
        out = self(feat)

        # Aggregate over T: mean prediction (or use last frame for streaming)
        cls_prob  = out["class_logits"].softmax(-1).mean(dim=1)    # (B, S, 301)
        doa_mean  = F.normalize(out["doa_vec"].mean(dim=1), dim=-1) # (B, S, 3)
        loud_mean = out["loudness"].mean(dim=1)                     # (B, S)
        conf_mean = out["confidence"].sigmoid().mean(dim=1)         # (B, S)
        src_emb   = out["source_embed"]                             # (B, S, 256)

        B, S = conf_mean.shape
        results = []
        for b in range(B):
            dets = []
            for s in range(S):
                if float(conf_mean[b, s]) < conf_threshold:
                    continue
                class_id = int(cls_prob[b, s, :300].argmax())
                x, y, z  = doa_mean[b, s].tolist()
                az  = float(torch.atan2(torch.tensor(y), torch.tensor(x)))
                el  = float(torch.asin(torch.tensor(z).clamp(-1, 1)))
                dets.append({
                    "class_id":      class_id,
                    "class_name":    (class_names or {}).get(class_id, str(class_id)),
                    "azimuth_rad":   az,
                    "elevation_rad": el,
                    "loudness_db":   float(loud_mean[b, s]),
                    "confidence":    float(conf_mean[b, s]),
                    "source_embed":  src_emb[b, s],
                })
            results.append(dets)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Factory + param count utility
# ──────────────────────────────────────────────────────────────────────────────

def build_sled(cfg: SLEDConfig = DEFAULT_CFG) -> SLED:
    return SLED(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_summary(model: nn.Module) -> str:
    total = count_parameters(model)
    lines = [f"{'Module':<40} {'Params':>10}"]
    lines.append("-" * 52)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lines.append(f"  {name:<38} {n:>10,}")
    lines.append("-" * 52)
    lines.append(f"  {'Total':<38} {total:>10,}")
    return "\n".join(lines)
