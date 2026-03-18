"""
SLED loss functions with Hungarian matching.

Per-frame Hungarian matching aligns predicted slots with GT sources,
then computes:
  - Focal loss (classification, α=0.25, γ=2.0)
  - Cosine distance loss (DOA, = 1 - cos_sim)
  - Smooth L1 loss (loudness)
  - BCE loss (confidence)
  - SCE auxiliary loss [training only] (DOA × loudness MSE)

Cost matrix for Hungarian matching:
  cost = λ_cls * cls_cost + λ_doa * doa_cost + λ_conf * conf_cost

Where:
  cls_cost  = -p(class_id)               (negative predicted probability)
  doa_cost  = 1 - cos(pred_doa, gt_doa)  (angular distance)
  conf_cost = 1 - confidence             (penalize low-confidence predictions)

Inputs are per-frame tensors; loss is computed by averaging over B×T.

GT tensors (from torch_dataset.py):
    cls   (B, T, S) int64   class_id, -1 = inactive
    doa   (B, T, S, 3) float32  unit vector
    loud  (B, T, S) float32  dBFS
    mask  (B, T, S) bool     active slot
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────────────
# Component losses
# ──────────────────────────────────────────────────────────────────────────────

def focal_loss(
    logits:   torch.Tensor,   # (N, n_cls)
    targets:  torch.Tensor,   # (N,) int64
    alpha:    float = 0.25,
    gamma:    float = 2.0,
    reduction: str  = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss for multi-class (one-vs-all interpretation)."""
    ce = F.cross_entropy(logits, targets, reduction="none")    # (N,)
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def cosine_distance_loss(
    pred: torch.Tensor,    # (N, 3) unit vectors
    tgt:  torch.Tensor,    # (N, 3) unit vectors
) -> torch.Tensor:
    """DOA loss: 1 - cosine_similarity, averaged over N."""
    cos_sim = F.cosine_similarity(pred, tgt, dim=-1)   # (N,)
    return (1.0 - cos_sim).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Hungarian matching (per frame)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def hungarian_match(
    pred_cls_prob: torch.Tensor,   # (P, n_cls) probabilities
    pred_doa:      torch.Tensor,   # (P, 3)
    pred_conf:     torch.Tensor,   # (P,)
    gt_cls:        torch.Tensor,   # (G,) int64
    gt_doa:        torch.Tensor,   # (G, 3)
    λ_cls:  float = 1.0,
    λ_doa:  float = 1.0,
    λ_conf: float = 0.5,
) -> tuple[list[int], list[int]]:
    """
    Match G GT sources to P predicted slots.
    Returns (pred_indices, gt_indices) for matched pairs.
    """
    P, G = pred_cls_prob.shape[0], gt_cls.shape[0]
    if G == 0 or P == 0:
        return [], []

    # Classification cost: -prob(gt_class)
    cls_cost = -pred_cls_prob[:, gt_cls]         # (P, G)

    # DOA cost: 1 - cos_sim
    pred_norm = F.normalize(pred_doa, dim=-1)
    gt_norm   = F.normalize(gt_doa,   dim=-1)
    doa_cost  = 1.0 - torch.mm(pred_norm, gt_norm.T)  # (P, G)

    # Confidence cost: penalize assigning low-confidence to GT
    conf_prob = pred_conf.sigmoid()
    conf_cost = (1.0 - conf_prob).unsqueeze(1).expand(-1, G)  # (P, G)

    cost = (λ_cls * cls_cost + λ_doa * doa_cost + λ_conf * conf_cost).cpu().numpy()

    row_idx, col_idx = linear_sum_assignment(cost)
    return row_idx.tolist(), col_idx.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Full SLED loss
# ──────────────────────────────────────────────────────────────────────────────

class SLEDLoss(nn.Module):
    """
    Computes SLED training loss over a batch of frames.

    Applies per-frame Hungarian matching then averages losses.

    Weights (from SLED_MODEL.md):
        classification : 1.0  (focal loss)
        DOA            : 1.0  (cosine distance)
        loudness       : 0.5  (smooth L1)
        confidence     : 1.0  (BCE)
        SCE auxiliary  : 0.3  (MSE on unit_doa × loudness)
    """

    def __init__(
        self,
        n_classes:       int   = 300,
        focal_alpha:     float = 0.25,
        focal_gamma:     float = 2.0,
        w_cls:           float = 1.0,
        w_doa:           float = 1.0,
        w_loud:          float = 0.5,
        w_conf:          float = 1.0,
        w_sce:           float = 0.3,
    ):
        super().__init__()
        self.n_classes  = n_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.w_cls       = w_cls
        self.w_doa       = w_doa
        self.w_loud      = w_loud
        self.w_conf      = w_conf
        self.w_sce       = w_sce
        self.empty_class = n_classes   # class 300

    def forward(
        self,
        pred:  dict[str, torch.Tensor],
        gt:    dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: {
                class_logits  (B, T, S, n_cls+1),
                doa_vec       (B, T, S, 3),
                loudness      (B, T, S),
                confidence    (B, T, S),
                sce_vec       (B, T, S, 3)  [training only],
            }
            gt: {
                cls   (B, T, S) int64   -1 = inactive
                doa   (B, T, S, 3) float32
                loud  (B, T, S) float32
                mask  (B, T, S) bool
            }

        Returns: dict of scalar loss tensors + "total"
        """
        B, T, S_pred  = pred["class_logits"].shape[:3]
        device = pred["class_logits"].device

        # Accumulate matched losses
        loss_cls   = torch.tensor(0.0, device=device)
        loss_doa   = torch.tensor(0.0, device=device)
        loss_loud  = torch.tensor(0.0, device=device)
        loss_conf  = torch.tensor(0.0, device=device)
        loss_sce   = torch.tensor(0.0, device=device)
        n_matched  = 0

        gt_cls  = gt["cls"]    # (B, T, S_gt)
        gt_doa  = gt["doa"]    # (B, T, S_gt, 3)
        gt_loud = gt["loud"]   # (B, T, S_gt)
        gt_mask = gt["mask"]   # (B, T, S_gt) bool

        has_sce = "sce_vec" in pred

        for b in range(B):
            for t in range(T):
                active = gt_mask[b, t]                     # (S_gt,) bool
                gt_idx = active.nonzero(as_tuple=True)[0]  # active GT source indices
                G = len(gt_idx)

                cls_logits_t = pred["class_logits"][b, t]  # (S_pred, n_cls+1)
                doa_t        = pred["doa_vec"][b, t]        # (S_pred, 3)
                loud_t       = pred["loudness"][b, t]       # (S_pred,)
                conf_t       = pred["confidence"][b, t]     # (S_pred,)

                # ── Confidence loss: all inactive predictions should be 0 ──
                if G == 0:
                    # No GT → all slots predict empty; confidence → 0
                    loss_conf = loss_conf + F.binary_cross_entropy_with_logits(
                        conf_t,
                        torch.zeros_like(conf_t),
                        reduction="mean",
                    )
                    # Classification: all slots → empty class
                    tgt_empty = torch.full((S_pred,), self.empty_class,
                                          dtype=torch.long, device=device)
                    loss_cls = loss_cls + focal_loss(
                        cls_logits_t, tgt_empty,
                        self.focal_alpha, self.focal_gamma,
                    )
                    continue

                # ── Hungarian matching ───────────────────────────────────
                cls_prob_t = cls_logits_t.softmax(dim=-1)   # (S_pred, n_cls+1)
                gt_cls_t   = gt_cls[b, t][gt_idx]           # (G,)
                gt_doa_t   = gt_doa[b, t][gt_idx]           # (G, 3)
                gt_loud_t  = gt_loud[b, t][gt_idx]          # (G,)

                pred_idx, match_idx = hungarian_match(
                    cls_prob_t[:, :self.n_classes],
                    doa_t, conf_t,
                    gt_cls_t, gt_doa_t,
                )

                if not pred_idx:
                    continue

                pi = torch.tensor(pred_idx,  device=device)
                gi = torch.tensor(match_idx, device=device)

                n_matched += len(pi)

                # ── Classification loss (matched + unmatched) ────────────
                # Matched: GT class
                tgt_cls_matched = gt_cls_t[gi]             # (M,)
                loss_cls = loss_cls + focal_loss(
                    cls_logits_t[pi], tgt_cls_matched,
                    self.focal_alpha, self.focal_gamma,
                )
                # Unmatched predictions → empty class
                unmatched_mask = torch.ones(S_pred, dtype=torch.bool, device=device)
                unmatched_mask[pi] = False
                if unmatched_mask.any():
                    tgt_empty = torch.full(
                        (unmatched_mask.sum(),), self.empty_class,
                        dtype=torch.long, device=device,
                    )
                    loss_cls = loss_cls + focal_loss(
                        cls_logits_t[unmatched_mask], tgt_empty,
                        self.focal_alpha, self.focal_gamma,
                    )

                # ── DOA loss ─────────────────────────────────────────────
                loss_doa = loss_doa + cosine_distance_loss(
                    doa_t[pi], gt_doa_t[gi]
                )

                # ── Loudness loss ─────────────────────────────────────────
                loss_loud = loss_loud + F.smooth_l1_loss(
                    loud_t[pi], gt_loud_t[gi]
                )

                # ── Confidence loss ───────────────────────────────────────
                tgt_conf = torch.zeros(S_pred, device=device)
                tgt_conf[pi] = 1.0
                loss_conf = loss_conf + F.binary_cross_entropy_with_logits(conf_t, tgt_conf)

                # ── SCE auxiliary ─────────────────────────────────────────
                if has_sce:
                    sce_t    = pred["sce_vec"][b, t]              # (S_pred, 3)
                    # GT SCE: unit_doa × linear(loudness)
                    gt_loud_lin = (gt_loud_t[gi] / 20.0).exp()   # (M,)
                    gt_sce  = gt_doa_t[gi] * gt_loud_lin.unsqueeze(-1)  # (M, 3)
                    loss_sce = loss_sce + F.mse_loss(sce_t[pi], gt_sce)

        # Normalise by batch×time
        norm = float(B * T)
        losses = {
            "cls":  self.w_cls  * loss_cls  / norm,
            "doa":  self.w_doa  * loss_doa  / norm,
            "loud": self.w_loud * loss_loud / norm,
            "conf": self.w_conf * loss_conf / norm,
        }
        if has_sce:
            losses["sce"] = self.w_sce * loss_sce / norm

        losses["total"] = sum(losses.values())
        return losses
