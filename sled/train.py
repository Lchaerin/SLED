"""
SLED training script.

Features:
  - Curriculum learning (max_sources grows over epochs)
  - AdamW + cosine LR decay with linear warmup
  - Gradient clipping (max_norm=5.0)
  - Mixed precision (torch.amp)
  - DDP-ready (torch.distributed)
  - Checkpointing (best val loss + periodic)
  - TensorBoard logging

Usage:
    python -m sled.train \
        --dataset /path/to/dataset \
        --output  /path/to/checkpoints \
        --epochs  300 \
        --batch   32 \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sled.model      import SLED, SLEDConfig, build_sled, parameter_summary
from sled.preprocess import BinauralPreprocessor
from sled.loss       import SLEDLoss
from dataset.torch_dataset import SLEDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sled.train")


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum: max active sources per epoch
# ──────────────────────────────────────────────────────────────────────────────

def curriculum_max_sources(epoch: int) -> int:
    """
    Epoch 1–50:   max 2 sources
    Epoch 51–100: max 3 sources
    Epoch 101+:   max 5 sources
    """
    if epoch <= 50:
        return 2
    elif epoch <= 100:
        return 3
    else:
        return 5


def apply_curriculum_mask(
    gt: dict[str, torch.Tensor],
    max_sources: int,
) -> dict[str, torch.Tensor]:
    """
    Zero out GT slots beyond max_sources to enforce curriculum.
    Assumes slots are ordered by onset (earliest first).
    """
    if max_sources >= gt["mask"].shape[-1]:
        return gt
    gt = {k: v.clone() for k, v in gt.items()}
    gt["mask"][..., max_sources:] = False
    gt["cls"][..., max_sources:]  = -1
    return gt


# ──────────────────────────────────────────────────────────────────────────────
# LR scheduler: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# One training epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:      SLED,
    preproc:    BinauralPreprocessor,
    criterion:  SLEDLoss,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    step:       int,
    warmup_steps: int,
    total_steps:  int,
    lr_max:     float,
    lr_min:     float,
    grad_clip:  float,
    writer:     SummaryWriter | None,
) -> tuple[dict[str, float], int]:

    model.train()
    preproc.train()

    meters = {k: AverageMeter() for k in ["total", "cls", "doa", "loud", "conf", "sce"]}
    max_src = curriculum_max_sources(epoch)
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        # ── LR schedule ─────────────────────────────────────────────────
        lr = get_lr(step, warmup_steps, total_steps, lr_max, lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        audio  = batch["audio"].to(device, non_blocking=True)   # (B, 2, N)
        gt = {
            "cls":  batch["cls"].to(device, non_blocking=True),    # (B, T, 5)
            "doa":  batch["doa"].to(device, non_blocking=True),    # (B, T, 5, 3)
            "loud": batch["loud"].to(device, non_blocking=True),   # (B, T, 5)
            "mask": batch["mask"].to(device, non_blocking=True),   # (B, T, 5)
        }
        gt = apply_curriculum_mask(gt, max_src)

        with autocast("cuda"):
            feat  = preproc(audio)                # (B, 5, 64, T)
            pred  = model(feat)                   # dict
            losses = criterion(pred, gt)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(preproc.parameters()),
            grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        B = audio.size(0)
        for k in meters:
            if k in losses:
                meters[k].update(float(losses[k]), B)

        if writer is not None and step % 50 == 0:
            writer.add_scalar("train/loss",    meters["total"].val, step)
            writer.add_scalar("train/loss_cls", meters["cls"].val,  step)
            writer.add_scalar("train/loss_doa", meters["doa"].val,  step)
            writer.add_scalar("train/lr",       lr,                 step)

        step += 1

    elapsed = time.time() - t0
    logger.info(
        "Epoch %d [train] loss=%.4f cls=%.4f doa=%.4f loud=%.4f conf=%.4f sce=%.4f  "
        "time=%.1fs  max_src=%d  lr=%.2e",
        epoch,
        meters["total"].avg, meters["cls"].avg, meters["doa"].avg,
        meters["loud"].avg,  meters["conf"].avg, meters["sce"].avg,
        elapsed, max_src, lr,
    )
    return {k: m.avg for k, m in meters.items()}, step


# ──────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:     SLED,
    preproc:   BinauralPreprocessor,
    criterion: SLEDLoss,
    loader:    DataLoader,
    device:    torch.device,
    epoch:     int,
    writer:    SummaryWriter | None,
) -> dict[str, float]:

    model.eval()
    preproc.eval()
    meters = {k: AverageMeter() for k in ["total", "cls", "doa", "loud", "conf"]}

    for batch in loader:
        audio = batch["audio"].to(device, non_blocking=True)
        gt = {
            "cls":  batch["cls"].to(device, non_blocking=True),
            "doa":  batch["doa"].to(device, non_blocking=True),
            "loud": batch["loud"].to(device, non_blocking=True),
            "mask": batch["mask"].to(device, non_blocking=True),
        }
        with autocast("cuda"):
            feat  = preproc(audio)
            pred  = model(feat)
            losses = criterion(pred, gt)

        B = audio.size(0)
        for k in meters:
            if k in losses:
                meters[k].update(float(losses[k]), B)

    avg = {k: m.avg for k, m in meters.items()}
    logger.info(
        "Epoch %d [val]   loss=%.4f cls=%.4f doa=%.4f loud=%.4f conf=%.4f",
        epoch,
        avg["total"], avg["cls"], avg["doa"], avg["loud"], avg["conf"],
    )
    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"val/{k}", v, epoch)

    return avg


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────
    train_ds = SLEDDataset(
        args.dataset, split="train",
        window_frames=args.window_frames,
        augment_scs=True,
    )
    val_ds = SLEDDataset(
        args.dataset, split="val",
        window_frames=args.window_frames,
        augment_scs=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, prefetch_factor=4,
        pin_memory=True, persistent_workers=(args.workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers // 2, prefetch_factor=4,
        pin_memory=True, persistent_workers=(args.workers > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────
    cfg = SLEDConfig()
    model   = build_sled(cfg).to(device)
    preproc = BinauralPreprocessor().to(device)
    logger.info("\n%s", parameter_summary(model))
    logger.info("Preprocessor params: %d", sum(p.numel() for p in preproc.parameters()))

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = SLEDLoss(n_classes=cfg.n_classes)

    # ── Optimizer ─────────────────────────────────────────────────────
    all_params = list(model.parameters()) + list(preproc.parameters())
    optimizer  = optim.AdamW(
        all_params, lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    scaler = GradScaler("cuda")

    # ── LR schedule ───────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = 5 * steps_per_epoch        # 5 epoch warmup
    lr_min          = 1e-5

    # ── Checkpointing ─────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")
    global_step = 0

    if args.resume and (output_dir / "latest.pt").exists():
        ckpt = torch.load(output_dir / "latest.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        preproc.load_state_dict(ckpt["preproc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val     = ckpt.get("best_val", float("inf"))
        global_step  = ckpt.get("step", 0)
        logger.info("Resumed from epoch %d (best_val=%.4f)", start_epoch - 1, best_val)

    writer = SummaryWriter(output_dir / "tb_logs") if args.tb else None

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics, global_step = train_epoch(
            model=model, preproc=preproc, criterion=criterion,
            loader=train_loader, optimizer=optimizer, scaler=scaler,
            device=device, epoch=epoch, step=global_step,
            warmup_steps=warmup_steps, total_steps=total_steps,
            lr_max=args.lr, lr_min=lr_min,
            grad_clip=5.0, writer=writer,
        )

        if epoch % args.val_every == 0:
            val_metrics = val_epoch(
                model=model, preproc=preproc, criterion=criterion,
                loader=val_loader, device=device, epoch=epoch, writer=writer,
            )

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                _save_checkpoint(
                    output_dir / "best.pt",
                    model, preproc, optimizer, scaler, epoch, global_step, best_val,
                )
                logger.info("  ★ New best val loss: %.4f  (saved best.pt)", best_val)

        if epoch % args.save_every == 0:
            _save_checkpoint(
                output_dir / "latest.pt",
                model, preproc, optimizer, scaler, epoch, global_step, best_val,
            )
            _save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model, preproc, optimizer, scaler, epoch, global_step, best_val,
            )

    if writer:
        writer.close()
    logger.info("Training complete. Best val loss: %.4f", best_val)


def _save_checkpoint(
    path:        Path,
    model:       SLED,
    preproc:     BinauralPreprocessor,
    optimizer:   optim.Optimizer,
    scaler:      GradScaler,
    epoch:       int,
    step:        int,
    best_val:    float,
) -> None:
    torch.save({
        "model":     model.state_dict(),
        "preproc":   preproc.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "step":      step,
        "best_val":  best_val,
    }, path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train SLED model")
    p.add_argument("--dataset",       required=True,   help="Path to dataset/ directory")
    p.add_argument("--output",        required=True,   help="Checkpoint output directory")
    p.add_argument("--epochs",        type=int, default=300)
    p.add_argument("--batch",         type=int, default=32)
    p.add_argument("--workers",       type=int, default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--window-frames", type=int, default=256,
                   help="Number of 20ms frames per training window (default 256 = 5.12 s)")
    p.add_argument("--val-every",     type=int, default=5,  help="Validate every N epochs")
    p.add_argument("--save-every",    type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--tb",            action="store_true", help="Enable TensorBoard logging")
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()
