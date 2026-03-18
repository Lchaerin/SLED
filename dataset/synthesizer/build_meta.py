"""
Build dataset meta files:
  meta/class_map.json       class_id (int) → label (str)
  meta/hrtf_registry.json   subject_id → file path + n_positions
  meta/split.json           train/val/test file lists
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np

from config import SynthConfig, DEFAULT_CFG

logger = logging.getLogger(__name__)


def build_class_map(cfg: SynthConfig = DEFAULT_CFG) -> dict:
    """Read FSD50K vocabulary and build class_id→label map."""
    import csv
    class_map = {}
    with open(cfg.fsd50k_vocab_csv, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            cid = int(row[0])
            label = row[1].strip()
            class_map[cid] = label

    # Class 300 reserved for empty/silence in the SLED model
    class_map[300] = "__empty__"
    return class_map


def build_hrtf_registry(cfg: SynthConfig = DEFAULT_CFG) -> list[dict]:
    """Scan HRTF dir and record metadata for each subject."""
    registry = []
    for path in sorted(cfg.hrtf_dir.glob("p*.sofa")):
        try:
            with h5py.File(path, "r") as f:
                n_pos = int(f["Data.IR"].shape[0])
                sr = float(f["Data.SamplingRate"][0])
        except Exception as e:
            logger.warning("Cannot read %s: %s", path, e)
            continue
        registry.append({
            "subject_id": path.stem,
            "file": str(path.relative_to(cfg.output_dir)),
            "n_positions": n_pos,
            "sample_rate": int(sr),
        })
    return registry


def build_split_json(cfg: SynthConfig = DEFAULT_CFG) -> dict:
    """Build split.json from existing audio files if they exist."""
    split_dict: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        audio_dir = cfg.output_dir / "audio" / split
        if not audio_dir.exists():
            continue
        for wav_path in sorted(audio_dir.glob("scene_*.wav")):
            split_dict[split].append(wav_path.stem)
    return split_dict


def build_all(cfg: SynthConfig = DEFAULT_CFG) -> None:
    meta_dir = cfg.meta_dir
    meta_dir.mkdir(parents=True, exist_ok=True)

    # class_map.json
    class_map = build_class_map(cfg)
    with open(meta_dir / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    logger.info("Wrote class_map.json (%d classes)", len(class_map))

    # hrtf_registry.json
    registry = build_hrtf_registry(cfg)
    with open(meta_dir / "hrtf_registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Wrote hrtf_registry.json (%d subjects)", len(registry))

    # split.json (populated after synthesis)
    split_json = build_split_json(cfg)
    with open(meta_dir / "split.json", "w") as f:
        json.dump(split_json, f, indent=2)
    logger.info("Wrote split.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_all()
    print("Meta files written to:", DEFAULT_CFG.meta_dir)
