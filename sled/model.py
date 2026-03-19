"""
SLED — Sound Localization & Event Detection model.

Architecture (per SLED_MODEL.md):
  Input  : (B, 5, 64, T)   5-channel feature from BinauralPreprocessor
  Output : detections list + source_embed (B, 5, 128) + clap_embed (B, 5, 512)

Forward path:
  ConvBlock×3  (5→32→64→128 ch, freq-MaxPool×3 → freq 64→32→16→8)
    ↓  saves P3, P4, P5 for BiFPN
  SEBlock      (channel attention on 128-ch output)
  Flatten+Proj (B,128×8,T) → Linear(1024,128) → (B,128,T)
  Conformer×4  (B,T,128) → (B,T,128)
  BiFPN Neck   (P3,P4,P5) → (B,128,T) — added to Conformer output
  TemporalAttPool  (B,T,128) → (B,128)
  Broadcast → (B,5,128)
  ┌ Class Head      → (B,5,301)  softmax
  ├ DOA Head        → (B,5,3)    L2-norm unit vector
  ├ Loudness Head   → (B,5,1)    dB regression
  └ Confidence Head → (B,5,1)    sigmoid

SlotAttentionPool  → source_embed (B,5,128)   VLA 전달용
CLAPProjectionHead → clap_embed   (B,5,512)   (optional)

Parameter budget: ~2.1M
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks    import ConvBlock, SEBlock, BiFPNNeck, TemporalAttPool
from .conformer import ConformerBlock


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SLEDConfig:
    n_classes:    int   = 300        # foreground classes (class 300 = empty)
    n_slots:      int   = 5          # max simultaneous sources
    d_model:      int   = 128
    n_mels:       int   = 64
    in_channels:  int   = 5          # L-mel, R-mel, cos-IPD, sin-IPD, ILD
    # ConvBlocks
    conv_channels: tuple = (32, 64, 128)
    # Conformer
    n_conformer:  int   = 4
    n_heads:      int   = 4
    ffn_dim:      int   = 512
    conv_kernel:  int   = 31
    dropout:      float = 0.1
    # BiFPN
    n_bifpn:      int   = 2
    # CLAP projection (optional)
    clap_dim:     int   = 512


DEFAULT_CFG = SLEDConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Slot Attention Pool  (VLA 전달용 source embedding)
# ──────────────────────────────────────────────────────────────────────────────

class SlotAttentionPool(nn.Module):
    """
    Attention-pool Conformer temporal features into per-slot embeddings.

    Input:
        feat         (B, T, d)    Conformer output
        slot_feat    (B, S, d)    slot-level broadcast features
    Output:
        source_embed (B, S, d)
    """

    def __init__(self, d: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.scale   = d ** -0.5

    def forward(
        self,
        feat:      torch.Tensor,   # (B, T, d)
        slot_feat: torch.Tensor,   # (B, S, d)
    ) -> torch.Tensor:
        q    = self.q_proj(slot_feat)                  # (B, S, d)
        k    = self.k_proj(feat)                       # (B, T, d)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, S, T)
        attn = attn.softmax(dim=-1)
        return torch.bmm(attn, feat)                   # (B, S, d)


# ──────────────────────────────────────────────────────────────────────────────
# CLAP Projection Head  (SLED → CLAP audio space)
# ──────────────────────────────────────────────────────────────────────────────

class CLAPProjectionHead(nn.Module):
    """Project SLED 128d source embeddings to CLAP 512d space."""

    def __init__(self, sled_dim: int = 128, clap_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sled_dim, 256),
            nn.GELU(),
            nn.Linear(256, clap_dim),
        )

    def forward(self, source_embeds: torch.Tensor) -> torch.Tensor:
        # source_embeds: (B, S, 128) → (B, S, 512) L2-normalized
        return F.normalize(self.proj(source_embeds), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# SLED Encoder
# ──────────────────────────────────────────────────────────────────────────────

class SLEDEncoder(nn.Module):
    """
    Encoder: feature extraction (Conv+SE+BiFPN) + temporal modeling (Conformer).
    Returns (B, T, d_model) for use by the detection heads.

    Also returns intermediate conv features for BiFPN and
    the pre-temporal-pool features for SlotAttentionPool.
    """

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        c1, c2, c3 = cfg.conv_channels          # 32, 64, 128

        # Conv blocks  (5→32→64→128 channels)
        self.cb1 = ConvBlock(cfg.in_channels, c1, dropout=0.0)
        self.cb2 = ConvBlock(c1, c2, dropout=0.0)
        self.cb3 = ConvBlock(c2, c3, dropout=0.0)

        # Channel attention
        self.se  = SEBlock(c3, reduction=16)

        # Freq flatten + projection:  (B, 128*8, T) → (B, 128, T)
        freq_after_pool = cfg.n_mels // (2 ** 3)           # 64 → 8
        self.freq_proj = nn.Conv1d(c3 * freq_after_pool, cfg.d_model, kernel_size=1, bias=False)
        self.freq_bn   = nn.BatchNorm1d(cfg.d_model)

        # Conformer stack
        self.conformers = nn.ModuleList([
            ConformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim,
                conv_kernel=cfg.conv_kernel,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_conformer)
        ])

        # BiFPN neck
        self.bifpn = BiFPNNeck(d=cfg.d_model, n_layers=cfg.n_bifpn)

        # Post-fusion norm
        self.post_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        feat: torch.Tensor,                          # (B, 5, 64, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            enc_out  (B, T, d)   full temporal feature sequence (for SlotAttPool)
            pooled   (B, d)      temporally pooled feature (for detection heads)
        """
        # ── Conv blocks + save intermediates for BiFPN
        p3 = self.cb1(feat)     # (B, 32, 32, T)
        p4 = self.cb2(p3)       # (B, 64, 16, T)
        p5 = self.cb3(p4)       # (B, 128,  8, T)

        # ── Channel attention
        p5_se = self.se(p5)     # (B, 128, 8, T)

        # ── Flatten freq → temporal sequence
        B, C, Freq, T = p5_se.shape
        x = p5_se.reshape(B, C * Freq, T)              # (B, 1024, T)
        x = F.silu(self.freq_bn(self.freq_proj(x)))    # (B, 128, T)
        x = x.transpose(1, 2)                          # (B, T, 128)

        # ── Conformer
        for block in self.conformers:
            x = block(x)                               # (B, T, 128)

        # ── BiFPN: fuse multi-scale conv features
        bifpn_feat = self.bifpn(p3, p4, p5)            # (B, 128, T)
        bifpn_feat = bifpn_feat.transpose(1, 2)        # (B, T, 128)

        enc_out = self.post_norm(x + bifpn_feat)       # (B, T, 128)
        return enc_out


# ──────────────────────────────────────────────────────────────────────────────
# Detection Heads
# ──────────────────────────────────────────────────────────────────────────────

class DetectionHeads(nn.Module):
    """Fixed-slot detection heads (per slot, shared weights)."""

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        d = cfg.d_model
        n_cls = cfg.n_classes + 1   # 301

        # Shared projection per slot (applied identically to each of the 5 slots)
        self.slot_proj = nn.Linear(d, d)
        self.norm      = nn.LayerNorm(d)

        self.class_head = nn.Linear(d, n_cls)
        self.doa_head   = nn.Linear(d, 3)
        self.loud_head  = nn.Linear(d, 1)
        self.conf_head  = nn.Linear(d, 1)

    def forward(
        self,
        slot_feat: torch.Tensor,   # (B, S, d)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict:
            class_logits  (B, S, 301)
            doa_vec       (B, S, 3)    unit vector
            loudness      (B, S)       dB
            confidence    (B, S)       [0, 1]
        """
        x = self.norm(F.gelu(self.slot_proj(slot_feat)))  # (B, S, d)

        cls_logits = self.class_head(x)                     # (B, S, 301)
        doa_raw    = self.doa_head(x)                       # (B, S, 3)
        doa_vec    = F.normalize(doa_raw, p=2, dim=-1, eps=1e-6)  # unit vector
        loudness   = self.loud_head(x).squeeze(-1)          # (B, S)
        conf_logit = self.conf_head(x).squeeze(-1)            # (B, S) raw logit

        return {
            "class_logits": cls_logits,
            "doa_vec":      doa_vec,
            "loudness":     loudness,
            "confidence":   conf_logit,   # raw logit; apply sigmoid for inference
        }


# ──────────────────────────────────────────────────────────────────────────────
# Full SLED Model
# ──────────────────────────────────────────────────────────────────────────────

class SLED(nn.Module):
    """
    Full SLED model.

    forward(feat) accepts pre-computed 5-channel features (B, 5, 64, T).
    Use BinauralPreprocessor to convert raw audio to features.

    Training output:
        {
            "class_logits": (B, T, S, 301),
            "doa_vec":      (B, T, S, 3),
            "loudness":     (B, T, S),
            "confidence":   (B, T, S),
            "source_embed": (B, S, 128),   # last frame slot features
        }

    Per-frame decoding:
        The model processes windows of T frames. Outputs are per-frame
        so that the loss is computed frame-by-frame with the annotation.

    Notes:
        - In training mode: SCE auxiliary loss branch is active.
        - In eval mode: only filtered detections are returned.
    """

    def __init__(self, cfg: SLEDConfig = DEFAULT_CFG):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        S = cfg.n_slots

        self.encoder      = SLEDEncoder(cfg)
        self.temp_pool    = TemporalAttPool(d)
        self.slot_pool    = SlotAttentionPool(d)
        self.heads        = DetectionHeads(cfg)

        # Slot query embeddings — broadcasted from temporally-pooled feature
        # (no additional slot-specific parameters needed; we just replicate)

        self.clap_head    = CLAPProjectionHead(d, cfg.clap_dim)

        # SCE auxiliary (training only): linear on (unit_doa * loudness)
        self.sce_proj     = nn.Linear(3, d // 4)

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
            source_embed  (B, S, 128)  from last-frame attention pool
            sce_vec       (B, T, S, 3) only during training (SCE aux)
        """
        B, _, _, T = feat.shape
        S = self.cfg.n_slots
        d = self.cfg.d_model

        enc_out = self.encoder(feat)    # (B, T, d)

        # ── Per-frame detection ──────────────────────────────────────────
        # Apply heads at every time step:
        # (B, T, d) → (B*T, d) → heads → reshape
        x_flat = enc_out.reshape(B * T, d)

        # Broadcast to S slots: (B*T, S, d)
        slot_feat = x_flat.unsqueeze(1).expand(-1, S, -1)   # (B*T, S, d)

        head_out = self.heads(slot_feat)   # each value: (B*T, S, ...)

        class_logits = head_out["class_logits"].reshape(B, T, S, -1)  # (B,T,S,301)
        doa_vec      = head_out["doa_vec"].reshape(B, T, S, 3)         # (B,T,S,3)
        loudness     = head_out["loudness"].reshape(B, T, S)            # (B,T,S)
        confidence   = head_out["confidence"].reshape(B, T, S)          # (B,T,S)

        # ── Source embed (slot-attention pool over full sequence) ────────
        # Use the global temporal-pooled feature as slot query seed
        pooled = self.temp_pool(enc_out)                    # (B, d)
        slot_queries = pooled.unsqueeze(1).expand(-1, S, -1)  # (B, S, d)
        source_embed = self.slot_pool(enc_out, slot_queries)   # (B, S, d)

        out = {
            "class_logits": class_logits,
            "doa_vec":      doa_vec,
            "loudness":     loudness,
            "confidence":   confidence,
            "source_embed": source_embed,
        }

        # ── SCE auxiliary (training only) ─────────────────────────────
        if self.training:
            # Predicted source coordinate: unit_doa scaled by loudness
            # loudness is in dB, convert to linear amplitude for scaling
            loud_lin = (loudness / 20.0).exp()                # (B, T, S)
            sce_vec = doa_vec * loud_lin.unsqueeze(-1)         # (B, T, S, 3)
            out["sce_vec"] = sce_vec

        return out

    # ── CLAP embed (optional, call explicitly) ───────────────────────
    def get_clap_embeds(self, source_embed: torch.Tensor) -> torch.Tensor:
        """(B, S, 128) → (B, S, 512) CLAP-aligned embeddings."""
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
        cls_prob = out["class_logits"].softmax(-1).mean(dim=1)   # (B, S, 301)
        doa_mean = F.normalize(out["doa_vec"].mean(dim=1), dim=-1)# (B, S, 3)
        loud_mean = out["loudness"].mean(dim=1)                   # (B, S)
        conf_mean = out["confidence"].sigmoid().mean(dim=1)          # (B, S)
        src_emb   = out["source_embed"]                           # (B, S, 128)

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
                    "class_id":    class_id,
                    "class_name":  (class_names or {}).get(class_id, str(class_id)),
                    "azimuth_rad": az,
                    "elevation_rad": el,
                    "loudness_db": float(loud_mean[b, s]),
                    "confidence":  float(conf_mean[b, s]),
                    "source_embed": src_emb[b, s],
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
