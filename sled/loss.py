"""
SLED loss functions with clip-level Hungarian matching.

Clip-level Hungarian matching aligns predicted slots with GT sources once
per clip (not per frame), then computes per-frame losses with the fixed assignment:
  - Focal loss (classification, α=0.25, γ=2.0)
  - Cosine distance loss (DOA, = 1 - cos_sim)
  - Smooth L1 loss (loudness)
  - BCE loss (confidence)
  - SCE auxiliary loss [training only] (DOA × loudness MSE)

Cost matrix for Hungarian matching (averaged over T frames):
  cost = cls_cost + doa_cost + 0.5 * conf_cost

Where:
  cls_cost  = -mean_t p(class_id)              (negative predicted probability)
  doa_cost  = mean_t 1 - cos(pred_doa, gt_doa) (angular distance)
  conf_cost = mean_t 1 - confidence            (penalize low-confidence predictions)

Inputs are per-clip tensors; loss is computed by averaging over B.

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
# Full SLED loss
# ──────────────────────────────────────────────────────────────────────────────

class SLEDLoss(nn.Module):
    """
    Computes SLED training loss over a batch of clips.

    Applies clip-level Hungarian matching (once per clip, not per frame),
    then computes per-frame losses with the fixed slot assignment.

    Weights:
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
        w_doa:           float = 2.0,
        w_loud:          float = 0.5,
        w_conf:          float = 0.5,
        w_sce:           float = 0.1,
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.w_cls       = w_cls
        self.w_doa       = w_doa
        self.w_loud      = w_loud
        self.w_conf      = w_conf
        self.w_sce       = w_sce
        self.empty_class = n_classes   # class index 300

    @torch.no_grad()
    def _clip_hungarian(
        self,
        cls_prob:  torch.Tensor,   # (T, S_pred, n_cls+1)
        doa_pred:  torch.Tensor,   # (T, S_pred, 3)
        conf:      torch.Tensor,   # (T, S_pred)
        gt_cls:    torch.Tensor,   # (T, S_gt) int64
        gt_doa:    torch.Tensor,   # (T, S_gt, 3)
        gt_mask:   torch.Tensor,   # (T, S_gt) bool
    ) -> tuple[list[int], list[int], torch.Tensor]:
        """
        Clip-level Hungarian matching.

        Builds a cost matrix (S_pred × G) by averaging costs over the T frames
        where each GT slot is active, then runs linear_sum_assignment once.

        Returns:
            pred_indices  : matched prediction slot indices  (length M)
            gt_slot_indices: matched GT slot indices into original S_gt dim (length M)
            active_any    : (S_gt,) bool — GT slots active in any frame
        """
        S_pred = cls_prob.shape[1]
        active_any   = gt_mask.any(dim=0)              # (S_gt,)
        gt_slot_idx  = active_any.nonzero(as_tuple=True)[0]  # (G,)
        G = len(gt_slot_idx)

        if G == 0 or S_pred == 0:
            return [], [], active_any

        cost = torch.zeros(S_pred, G, device=cls_prob.device)

        for gi, g_slot in enumerate(gt_slot_idx):
            active_t = gt_mask[:, g_slot]               # (T,) bool
            if not active_t.any():
                continue

            # cls cost: -mean_t prob(gt_class) over active frames
            # GT class is constant per source slot — use first active frame
            gt_c = gt_cls[active_t, g_slot][0].clamp(0, self.n_classes - 1)
            cost[:, gi] += -cls_prob[active_t, :, gt_c].mean(dim=0)   # (S_pred,)

            # doa cost: mean_t (1 - cos_sim) over active frames
            p_doa = F.normalize(doa_pred[active_t], dim=-1)   # (n, S_pred, 3)
            g_doa = F.normalize(gt_doa[active_t, g_slot], dim=-1)  # (n, 3)
            cos_sim = (p_doa * g_doa.unsqueeze(1)).sum(-1)    # (n, S_pred)
            cost[:, gi] += (1.0 - cos_sim).mean(dim=0)        # (S_pred,)

            # conf cost: mean_t (1 - sigmoid(conf)) over active frames
            cost[:, gi] += (1.0 - conf[active_t].sigmoid()).mean(dim=0)  # (S_pred,)

        row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
        # col_idx indexes into gt_slot_idx; map back to original S_gt indices
        matched_gt_slots = gt_slot_idx[col_idx].tolist()
        return row_idx.tolist(), matched_gt_slots, active_any

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
        B, T, S_pred = pred["class_logits"].shape[:3]
        device = pred["class_logits"].device

        loss_cls  = torch.tensor(0.0, device=device)
        loss_doa  = torch.tensor(0.0, device=device)
        loss_loud = torch.tensor(0.0, device=device)
        loss_conf = torch.tensor(0.0, device=device)
        loss_sce  = torch.tensor(0.0, device=device)

        gt_cls  = gt["cls"]    # (B, T, S_gt)
        gt_doa  = gt["doa"]    # (B, T, S_gt, 3)
        gt_loud = gt["loud"]   # (B, T, S_gt)
        gt_mask = gt["mask"]   # (B, T, S_gt) bool
        has_sce = "sce_vec" in pred

        for b in range(B):
            cls_logits_b = pred["class_logits"][b]   # (T, S_pred, n_cls+1)
            doa_b        = pred["doa_vec"][b]         # (T, S_pred, 3)
            loud_b       = pred["loudness"][b]        # (T, S_pred)
            conf_b       = pred["confidence"][b]      # (T, S_pred)
            cls_prob_b   = cls_logits_b.softmax(dim=-1)

            # ── Clip-level Hungarian matching ────────────────────────────
            pred_idx, gt_slots_matched, active_any = self._clip_hungarian(
                cls_prob_b, doa_b, conf_b,
                gt_cls[b], gt_doa[b], gt_mask[b],
            )

            # ── Classification loss ──────────────────────────────────────
            # Build target tensor: (T, S_pred), default = empty_class
            tgt_cls = torch.full(
                (T, S_pred), self.empty_class, dtype=torch.long, device=device,
            )
            for p, g_slot in zip(pred_idx, gt_slots_matched):
                active_t = gt_mask[b, :, g_slot]
                tgt_cls[active_t, p] = gt_cls[b, active_t, g_slot]

            loss_cls = loss_cls + focal_loss(
                cls_logits_b.reshape(T * S_pred, -1),
                tgt_cls.reshape(-1),
                self.focal_alpha, self.focal_gamma,
            )

            # ── Confidence loss ──────────────────────────────────────────
            # Matched slots → 1 on active frames; all others → 0
            tgt_conf = torch.zeros(T, S_pred, device=device)
            for p, g_slot in zip(pred_idx, gt_slots_matched):
                tgt_conf[gt_mask[b, :, g_slot], p] = 1.0
            loss_conf = loss_conf + F.binary_cross_entropy_with_logits(
                conf_b, tgt_conf,
            )

            # ── DOA / Loudness / SCE losses (matched pairs only) ─────────
            for p, g_slot in zip(pred_idx, gt_slots_matched):
                active_t = gt_mask[b, :, g_slot]
                if not active_t.any():
                    continue

                loss_doa = loss_doa + cosine_distance_loss(
                    doa_b[active_t, p], gt_doa[b, active_t, g_slot],
                )
                loss_loud = loss_loud + F.smooth_l1_loss(
                    loud_b[active_t, p] / 20.0,
                    gt_loud[b, active_t, g_slot] / 20.0,
                )
                if has_sce:
                    gt_loud_lin = (gt_loud[b, active_t, g_slot] / 20.0).exp()
                    gt_sce = gt_doa[b, active_t, g_slot] * gt_loud_lin.unsqueeze(-1)
                    loss_sce = loss_sce + F.mse_loss(
                        pred["sce_vec"][b, active_t, p], gt_sce,
                    )

        # Normalise by batch size
        norm = float(B)
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
