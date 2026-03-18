"""
PyTorch Dataset for SLED training.

Design goals (bottleneck-free training):
  * Load pre-synthesized audio + dense .npy annotations (no on-the-fly synthesis).
  * Memory-map .npy files so the OS page-cache handles repeated access.
  * Return per-scene random windows of configurable length (e.g., 4 s chunks).
  * Apply online augmentations: SCS (Stereo Channel Swap) and SpecAugment.
  * The 5-channel feature extraction (L-mel, R-mel, cos-IPD, sin-IPD, ILD)
    is done on-the-fly on GPU via the model's preprocessor, NOT here.
    Here we return raw float32 stereo waveforms + labels.

DataLoader recommended settings:
    DataLoader(dataset, batch_size=32, num_workers=16, prefetch_factor=4,
               pin_memory=True, persistent_workers=True)

Item returned:
    audio:      Tensor[2, N_samples]              float32, stereo waveform
    cls:        Tensor[T, 5]                       int64   class ids (-1 = empty)
    doa:        Tensor[T, 5, 3]                    float32 unit vectors
    loud:       Tensor[T, 5]                       float32 dBFS
    mask:       Tensor[T, 5]                       bool    active slots

Scene-level windowing:
    A fixed window of `window_frames` frames (and corresponding audio samples)
    is randomly cropped from each 45-second scene each epoch.
    Set window_frames=None to return the full scene (for evaluation).
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

logger = logging.getLogger(__name__)

MAX_SLOTS = 5
EMPTY_CLASS = -1
HOP_SAMPLES = 960    # 20 ms at 48 kHz
SAMPLE_RATE = 48_000


class SLEDDataset(Dataset):
    """
    Dataset of pre-synthesized SLED binaural scenes.

    Args:
        dataset_root: path to the dataset/ directory
        split:        'train', 'val', or 'test'
        window_frames: number of frames per returned sample (None = full scene)
        augment_scs:  apply Stereo Channel Swap augmentation (train only)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        window_frames: Optional[int] = 256,   # 256 × 20ms = 5.12 s
        augment_scs: bool = True,
    ):
        self.root = Path(dataset_root)
        self.split = split
        self.window_frames = window_frames
        self.augment_scs = augment_scs and (split == "train")

        # Load split.json to enumerate scenes
        split_path = self.root / "meta" / "split.json"
        with open(split_path) as f:
            split_json = json.load(f)
        self.scene_ids: list[str] = split_json.get(split, [])
        if not self.scene_ids:
            raise ValueError(f"No scenes found for split '{split}' in {split_path}")

        # Directories
        self.audio_dir = self.root / "audio" / split
        self.dense_dir = self.root / "annotations_dense" / split

        logger.info("SLEDDataset[%s]: %d scenes", split, len(self.scene_ids))

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        scene_id = self.scene_ids[idx]
        stem = f"scene_{scene_id}"

        # ---- Load audio (stereo, float32) -----------------------------------
        wav_path = self.audio_dir / f"{stem}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        # audio: (N, 2) → (2, N)
        audio = torch.from_numpy(audio.T.copy())  # (2, N)

        # ---- Memory-map dense annotations -----------------------------------
        cls  = np.load(str(self.dense_dir / f"{stem}_cls.npy"),  mmap_mode="r")  # (T,5)
        doa  = np.load(str(self.dense_dir / f"{stem}_doa.npy"),  mmap_mode="r")  # (T,5,3)
        loud = np.load(str(self.dense_dir / f"{stem}_loud.npy"), mmap_mode="r")  # (T,5)
        mask = np.load(str(self.dense_dir / f"{stem}_mask.npy"), mmap_mode="r")  # (T,5)

        T_full = cls.shape[0]

        # ---- Random window crop ---------------------------------------------
        if self.window_frames is not None and T_full > self.window_frames:
            start_frame = random.randint(0, T_full - self.window_frames)
            end_frame = start_frame + self.window_frames
        else:
            start_frame = 0
            end_frame = T_full

        # Crop labels
        cls_w  = cls [start_frame:end_frame].copy()
        doa_w  = doa [start_frame:end_frame].copy()
        loud_w = loud[start_frame:end_frame].copy()
        mask_w = mask[start_frame:end_frame].copy()

        # Crop audio
        start_sample = start_frame * HOP_SAMPLES
        end_sample   = end_frame   * HOP_SAMPLES
        audio = audio[:, start_sample:end_sample]

        # ---- Stereo Channel Swap augmentation --------------------------------
        if self.augment_scs and random.random() < 0.5:
            audio, doa_w = _stereo_channel_swap(audio, doa_w)

        # ---- Convert to tensors ---------------------------------------------
        item = {
            "audio": audio,                                    # (2, N) float32
            "cls":   torch.from_numpy(cls_w.astype(np.int64)),   # (T, 5)
            "doa":   torch.from_numpy(doa_w.astype(np.float32)), # (T, 5, 3)
            "loud":  torch.from_numpy(loud_w.astype(np.float32)),# (T, 5)
            "mask":  torch.from_numpy(mask_w.astype(bool)),      # (T, 5)
            "scene_id": scene_id,
        }
        return item


def _stereo_channel_swap(
    audio: torch.Tensor,       # (2, N)
    doa: np.ndarray,           # (T, 5, 3) float32
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Swap L and R channels + negate azimuth (y-component of unit vector).
    This is a free 2× data augmentation for left-right symmetry.
    """
    audio_swapped = torch.stack([audio[1], audio[0]], dim=0)
    doa_swapped = doa.copy()
    doa_swapped[..., 1] *= -1.0   # negate y (right component → left)
    return audio_swapped, doa_swapped


def build_dataloader(
    dataset_root: str | Path,
    split: str,
    batch_size: int = 32,
    window_frames: Optional[int] = 256,
    augment_scs: bool = True,
    num_workers: int = 16,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Convenience factory for bottleneck-free DataLoaders.
    Uses persistent_workers=True to avoid per-epoch worker restart overhead.
    """
    dataset = SLEDDataset(
        dataset_root=dataset_root,
        split=split,
        window_frames=window_frames,
        augment_scs=augment_scs,
    )
    is_train = (split == "train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=is_train,
    )
    return loader
