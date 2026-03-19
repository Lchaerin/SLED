"""
SLED inference visualization script.

GT and predicted values are shown in separate side-by-side panels,
with quantitative metrics summarized at the top of each figure.

Usage:
    python -m sled.visualize \
        --dataset  /path/to/dataset \
        --ckpt     /path/to/checkpoints/best.pt \
        --output   /path/to/vis_output \
        --n-scenes 8 \
        --conf-thr 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sled.model      import SLEDConfig, build_sled
from sled.preprocess import BinauralPreprocessor
from dataset.torch_dataset import SLEDDataset

HOP_SEC     = 0.02
SLOT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def vec_to_azel(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.degrees(np.arctan2(y, x)), np.degrees(np.arcsin(np.clip(z, -1, 1)))


@torch.no_grad()
def run_inference(model, preproc, audio, device):
    audio = audio.unsqueeze(0).to(device)
    with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
        out = model(preproc(audio))
    return {
        "doa":      out["doa_vec"].squeeze(0).cpu().float().numpy(),
        "conf":     out["confidence"].squeeze(0).sigmoid().cpu().float().numpy(),
        "loudness": out["loudness"].squeeze(0).cpu().float().numpy(),
        "cls_id":   out["class_logits"].squeeze(0)[..., :300]
                        .softmax(-1).argmax(-1).cpu().numpy(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics  (greedy nearest-neighbor matching per frame)
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, conf_thr):
    """
    Per-frame greedy matching: each active GT slot is matched to the
    confident predicted slot with the smallest angular distance.

    Returns dict of scalar metrics.
    """
    T, S_pred = pred["conf"].shape
    gt_mask  = gt["mask"]    # (T, S_gt)
    gt_doa   = gt["doa"]     # (T, S_gt, 3)
    gt_loud  = gt["loud"]    # (T, S_gt)
    gt_cls   = gt["cls"]     # (T, S_gt)

    doa_errors  = []   # degrees
    loud_errors = []   # |dB|
    cls_correct = []   # bool

    # Detection stats
    n_gt_active  = int(gt_mask.sum())
    n_pred_conf  = int((pred["conf"] >= conf_thr).sum())
    n_tp         = 0

    for t in range(T):
        gt_idx   = np.where(gt_mask[t])[0]           # active GT slots
        pred_idx = np.where(pred["conf"][t] >= conf_thr)[0]  # confident pred slots
        if len(gt_idx) == 0:
            continue

        if len(pred_idx) == 0:
            continue

        # Cosine similarity matrix (gt × pred)
        gt_vecs   = gt_doa[t, gt_idx]       # (G, 3)
        pred_vecs = pred["doa"][t, pred_idx] # (P, 3)
        # normalize
        gt_n   = gt_vecs   / (np.linalg.norm(gt_vecs,   axis=-1, keepdims=True) + 1e-8)
        pred_n = pred_vecs / (np.linalg.norm(pred_vecs, axis=-1, keepdims=True) + 1e-8)
        cos_sim = gt_n @ pred_n.T            # (G, P)
        ang_err = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))  # (G, P) degrees

        used_pred = set()
        for gi, g_slot in enumerate(gt_idx):
            # Find closest unused pred slot
            sorted_p = np.argsort(ang_err[gi])
            for pi_local in sorted_p:
                if pi_local not in used_pred:
                    p_slot = pred_idx[pi_local]
                    used_pred.add(pi_local)

                    doa_errors.append(ang_err[gi, pi_local])
                    loud_errors.append(abs(pred["loudness"][t, p_slot] - gt_loud[t, g_slot]))
                    cls_correct.append(int(pred["cls_id"][t, p_slot] == gt_cls[t, g_slot]))
                    n_tp += 1
                    break

    precision = n_tp / n_pred_conf if n_pred_conf > 0 else 0.0
    recall    = n_tp / n_gt_active  if n_gt_active  > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "doa_mean_err_deg":  float(np.mean(doa_errors))  if doa_errors  else float("nan"),
        "doa_median_err_deg":float(np.median(doa_errors))if doa_errors  else float("nan"),
        "loud_mae_db":       float(np.mean(loud_errors)) if loud_errors else float("nan"),
        "cls_acc_pct":       float(np.mean(cls_correct)) * 100 if cls_correct else float("nan"),
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "n_gt_frames":       n_gt_active,
        "n_pred_frames":     n_pred_conf,
        "n_matched":         n_tp,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_azel(ax_gt, ax_pred, time, gt, pred, conf_thr, row_label):
    """Draw azimuth or elevation on a GT axis and a Pred axis."""
    is_az = (row_label == "Azimuth")
    gt_vals, pred_vals = (gt[0], pred[0]) if is_az else (gt[1], pred[1])

    gt_mask   = gt[2]   # (T, S)
    conf_mask = pred["conf"] >= conf_thr

    ylim = (-195, 195) if is_az else (-100, 100)
    for ax in (ax_gt, ax_pred):
        ax.set_ylim(*ylim)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{row_label} (°)", fontsize=8)

    for s in range(gt_mask.shape[1]):
        c      = SLOT_COLORS[s]
        active = gt_mask[:, s]
        # GT
        if active.any():
            ax_gt.plot(time, np.where(active, gt_vals[:, s], np.nan),
                       color=c, linewidth=1.2, label=f"Slot {s}")
        # Pred
        if conf_mask[:, s].any():
            ax_pred.plot(time, np.where(conf_mask[:, s], pred_vals[:, s], np.nan),
                         color=c, linewidth=1.2, label=f"Slot {s}")


def _plot_loudness(ax_gt, ax_pred, time, gt_loud, gt_mask, pred_loud, pred_conf, conf_thr):
    for ax in (ax_gt, ax_pred):
        ax.set_ylabel("dBFS", fontsize=8)

    for s in range(gt_mask.shape[1]):
        c      = SLOT_COLORS[s]
        active = gt_mask[:, s]
        if active.any():
            ax_gt.plot(time, np.where(active, gt_loud[:, s], np.nan),
                       color=c, linewidth=1.2)
        cm = pred_conf[:, s] >= conf_thr
        if cm.any():
            ax_pred.plot(time, np.where(cm, pred_loud[:, s], np.nan),
                         color=c, linewidth=1.2)


def _plot_class(ax_gt, ax_pred, time, gt_cls, gt_mask, pred_cls, pred_conf, conf_thr):
    for ax in (ax_gt, ax_pred):
        ax.set_ylabel("Class ID", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)

    for s in range(gt_mask.shape[1]):
        c      = SLOT_COLORS[s]
        active = gt_mask[:, s]
        if active.any():
            vals = gt_cls[:, s].astype(float)
            vals[~active] = np.nan
            ax_gt.scatter(time[active], vals[active],
                          color=c, s=8, zorder=3, alpha=0.8)
        cm = pred_conf[:, s] >= conf_thr
        if cm.any():
            ax_pred.scatter(time[cm], pred_cls[cm, s],
                            color=c, s=8, zorder=3, alpha=0.8)


def _plot_confidence(ax, time, pred_conf, gt_mask, conf_thr):
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Confidence", fontsize=8)
    ax.axhline(conf_thr, color="black", linewidth=0.8, linestyle=":", alpha=0.6,
               label=f"threshold={conf_thr}")
    for s in range(pred_conf.shape[1]):
        ax.plot(time, pred_conf[:, s], color=SLOT_COLORS[s],
                linewidth=1.0, alpha=0.9, label=f"Slot {s}")
        # GT active shading
        active = gt_mask[:, s]
        if active.any():
            ax.fill_between(time, 0, 1, where=active,
                            color=SLOT_COLORS[s], alpha=0.08, zorder=0)
    ax.legend(fontsize=7, loc="upper right", ncol=6)


# ──────────────────────────────────────────────────────────────────────────────
# Main plot function
# ──────────────────────────────────────────────────────────────────────────────

def plot_scene(pred, gt, metrics, scene_id, conf_thr, save_path):
    T, S = pred["conf"].shape
    time = np.arange(T) * HOP_SEC

    gt_az,   gt_el   = vec_to_azel(gt["doa"])
    pred_az, pred_el = vec_to_azel(pred["doa"])

    # Layout: 5 rows × 3 cols  [GT | divider | Pred]
    # Row 5 (confidence) spans both columns
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#f9f9f9")

    outer = gridspec.GridSpec(
        6, 1, figure=fig,
        height_ratios=[0.18, 1, 1, 1, 1, 1.1],
        hspace=0.55,
    )

    # ── Metrics banner ───────────────────────────────────────────────────
    ax_info = fig.add_subplot(outer[0])
    ax_info.axis("off")
    m = metrics
    info = (
        f"Scene: {scene_id}     "
        f"│  DOA mean err: {m['doa_mean_err_deg']:.1f}°  "
        f"│  DOA median err: {m['doa_median_err_deg']:.1f}°  "
        f"│  Loudness MAE: {m['loud_mae_db']:.2f} dB  "
        f"│  Class acc: {m['cls_acc_pct']:.1f}%  "
        f"│  Precision: {m['precision']:.2f}  "
        f"│  Recall: {m['recall']:.2f}  "
        f"│  F1: {m['f1']:.2f}  "
        f"│  Matched: {m['n_matched']} / GT: {m['n_gt_frames']} frames"
    )
    ax_info.text(0.5, 0.5, info, ha="center", va="center",
                 fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    row_labels  = ["Azimuth", "Elevation", "Loudness (dBFS)", "Class ID"]
    data_rows   = [1, 2, 3, 4]

    axes_gt   = []
    axes_pred = []

    for ri, (label, row) in enumerate(zip(row_labels, data_rows)):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[row],
            width_ratios=[1, 0.02, 1], wspace=0.08,
        )
        ax_gt   = fig.add_subplot(inner[0])
        ax_div  = fig.add_subplot(inner[1])
        ax_pred = fig.add_subplot(inner[2])

        ax_div.axis("off")
        ax_div.axvline(0.5, color="#cccccc", linewidth=1.5)

        # Column headers on first row only
        if ri == 0:
            ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold", pad=6,
                            color="#333333")
            ax_pred.set_title("Prediction", fontsize=11, fontweight="bold", pad=6,
                              color="#333333")

        ax_gt.set_ylabel(label, fontsize=8)
        for ax in (ax_gt, ax_pred):
            ax.tick_params(labelsize=7)
            ax.set_facecolor("white")
            if row < 4:   # not last row
                ax.tick_params(labelbottom=False)

        axes_gt.append(ax_gt)
        axes_pred.append(ax_pred)

    # ── Fill each row ────────────────────────────────────────────────────

    # Azimuth
    for ax, vals in [(axes_gt[0], gt_az), (axes_pred[0], pred_az)]:
        ax.set_ylim(-195, 195)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    for s in range(S):
        c, active = SLOT_COLORS[s], gt["mask"][:, s]
        if active.any():
            axes_gt[0].plot(time, np.where(active, gt_az[:, s], np.nan),
                            color=c, linewidth=1.2, label=f"Slot {s}")
        cm = pred["conf"][:, s] >= conf_thr
        if cm.any():
            axes_pred[0].plot(time, np.where(cm, pred_az[:, s], np.nan),
                              color=c, linewidth=1.2)

    # Elevation
    for ax in (axes_gt[1], axes_pred[1]):
        ax.set_ylim(-100, 100)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    for s in range(S):
        c, active = SLOT_COLORS[s], gt["mask"][:, s]
        if active.any():
            axes_gt[1].plot(time, np.where(active, gt_el[:, s], np.nan),
                            color=c, linewidth=1.2)
        cm = pred["conf"][:, s] >= conf_thr
        if cm.any():
            axes_pred[1].plot(time, np.where(cm, pred_el[:, s], np.nan),
                              color=c, linewidth=1.2)

    # Loudness
    for s in range(S):
        c, active = SLOT_COLORS[s], gt["mask"][:, s]
        if active.any():
            axes_gt[2].plot(time, np.where(active, gt["loud"][:, s], np.nan),
                            color=c, linewidth=1.2)
        cm = pred["conf"][:, s] >= conf_thr
        if cm.any():
            axes_pred[2].plot(time, np.where(cm, pred["loudness"][:, s], np.nan),
                              color=c, linewidth=1.2)
    for ax in (axes_gt[2], axes_pred[2]):
        ax.set_ylabel("dBFS", fontsize=8)

    # Class ID
    for s in range(S):
        c, active = SLOT_COLORS[s], gt["mask"][:, s]
        if active.any():
            vals = gt["cls"][:, s].astype(float)
            vals[~active] = np.nan
            axes_gt[3].scatter(time[active], vals[active], color=c, s=8, alpha=0.8)
        cm = pred["conf"][:, s] >= conf_thr
        if cm.any():
            axes_pred[3].scatter(time[cm], pred["cls_id"][cm, s], color=c, s=8, alpha=0.8)
    for ax in (axes_gt[3], axes_pred[3]):
        ax.set_xlabel("Time (s)", fontsize=8)

    # Confidence (spans full width)
    inner_conf = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[5])
    ax_conf = fig.add_subplot(inner_conf[0])
    ax_conf.set_facecolor("white")
    ax_conf.set_title("Prediction Confidence  (shaded = GT active region)", fontsize=9)
    ax_conf.set_ylim(-0.05, 1.1)
    ax_conf.set_ylabel("Confidence", fontsize=8)
    ax_conf.set_xlabel("Time (s)", fontsize=8)
    ax_conf.axhline(conf_thr, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    for s in range(S):
        c, active = SLOT_COLORS[s], gt["mask"][:, s]
        ax_conf.plot(time, pred["conf"][:, s], color=c, linewidth=1.0,
                     alpha=0.9, label=f"Slot {s}")
        if active.any():
            ax_conf.fill_between(time, 0, 1, where=active,
                                 color=c, alpha=0.08, zorder=0)
    ax_conf.legend(fontsize=8, loc="upper right", ncol=5)

    # Shared slot color legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SLOT_COLORS[s], linewidth=2.5, label=f"Slot {s}")
               for s in range(S)]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))

    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       required=True)
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--n-scenes",      type=int,   default=8)
    p.add_argument("--conf-thr",      type=float, default=0.3)
    p.add_argument("--window-frames", type=int,   default=None,
                   help="Frames per clip (None = full 45s scene)")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg     = SLEDConfig()
    model   = build_sled(cfg).to(device).eval()
    preproc = BinauralPreprocessor().to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    preproc.load_state_dict(ckpt["preproc"])
    print(f"Checkpoint: epoch {ckpt.get('epoch','?')}, "
          f"best_val={ckpt.get('best_val', float('nan')):.4f}\n")

    val_ds = SLEDDataset(
        args.dataset, split="val",
        window_frames=args.window_frames,
        augment_scs=False,
    )
    rng     = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(val_ds), generator=rng)[:args.n_scenes].tolist()

    all_metrics = []
    for rank, idx in enumerate(indices):
        sample   = val_ds[idx]
        scene_id = sample["scene_id"]
        print(f"[{rank+1}/{args.n_scenes}] {scene_id}")

        pred = run_inference(model, preproc, sample["audio"], device)
        gt   = {
            "doa":  sample["doa"].numpy(),
            "mask": sample["mask"].numpy(),
            "cls":  sample["cls"].numpy(),
            "loud": sample["loud"].numpy(),
        }

        metrics = compute_metrics(pred, gt, args.conf_thr)
        all_metrics.append(metrics)

        print(f"         DOA mean={metrics['doa_mean_err_deg']:.1f}°  "
              f"median={metrics['doa_median_err_deg']:.1f}°  "
              f"loud MAE={metrics['loud_mae_db']:.2f}dB  "
              f"cls acc={metrics['cls_acc_pct']:.1f}%  "
              f"F1={metrics['f1']:.2f}")

        save_path = out_dir / f"{rank+1:02d}_{scene_id}.png"
        plot_scene(pred, gt, metrics, scene_id, args.conf_thr, save_path)

    # ── Aggregate summary ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("="*60)
    for key in ["doa_mean_err_deg", "doa_median_err_deg",
                "loud_mae_db", "cls_acc_pct", "precision", "recall", "f1"]:
        vals = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if vals:
            print(f"{key:<25} {np.mean(vals):>10.3f} {np.std(vals):>10.3f}")
    print("="*60)
    print(f"\nDone. {args.n_scenes} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
