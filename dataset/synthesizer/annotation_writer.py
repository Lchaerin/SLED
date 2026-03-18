"""
Writes per-scene annotations in two formats:

1. JSON (human-readable / debugging)
   annotations/{split}/scene_{id:06d}.json

2. Dense numpy binary (training / fast I/O)
   annotations_dense/{split}/scene_{id:06d}_cls.npy   [T, 5] int16
   annotations_dense/{split}/scene_{id:06d}_doa.npy   [T, 5, 3] float16  (unit vec xyz)
   annotations_dense/{split}/scene_{id:06d}_loud.npy  [T, 5] float16  (dBFS)
   annotations_dense/{split}/scene_{id:06d}_mask.npy  [T, 5] bool

Slot assignment: sources are sorted by onset_frame (earliest first).
Inactive slots are filled with class_id=-1, doa=zeros, loud=-120, mask=False.

Unit vector convention (SLED model):
  x = cos(el) * cos(az)   (forward)
  y = cos(el) * sin(az)   (right, CW convention)
  z = sin(el)             (up)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scene_synth import SceneMeta, SourceMeta, TrajectoryPoint, _interpolate_trajectory

MAX_SLOTS = 5
EMPTY_CLASS = -1
EMPTY_LOUD = -120.0


def _az_el_to_unit_vec(az: float, el: float) -> tuple[float, float, float]:
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return x, y, z


def write_annotations(
    scene_id: str,
    meta: SceneMeta,
    output_dir: Path,
) -> None:
    """Write JSON + dense .npy annotations for one scene."""
    split = meta.split
    T = meta.n_frames

    # ---- Dense arrays -------------------------------------------------------
    cls_arr = np.full((T, MAX_SLOTS), EMPTY_CLASS, dtype=np.int16)
    doa_arr = np.zeros((T, MAX_SLOTS, 3), dtype=np.float16)
    loud_arr = np.full((T, MAX_SLOTS), EMPTY_LOUD, dtype=np.float16)
    mask_arr = np.zeros((T, MAX_SLOTS), dtype=bool)

    # Sort sources by onset_frame for stable slot assignment
    sources_sorted = sorted(meta.sources, key=lambda s: s.onset_frame)

    for slot, src in enumerate(sources_sorted[:MAX_SLOTS]):
        az_frames, el_frames = _interpolate_trajectory(src.trajectory, T)
        for t in range(src.onset_frame, min(src.offset_frame, T)):
            cls_arr[t, slot] = src.class_id
            x, y, z = _az_el_to_unit_vec(float(az_frames[t]), float(el_frames[t]))
            doa_arr[t, slot] = [x, y, z]
            loud_val = float(src.loudness_frames[t])
            loud_arr[t, slot] = loud_val if not math.isnan(loud_val) else EMPTY_LOUD
            mask_arr[t, slot] = True

    # ---- Save dense arrays --------------------------------------------------
    dense_dir = output_dir / "annotations_dense" / split
    dense_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_{scene_id}"
    np.save(dense_dir / f"{stem}_cls.npy", cls_arr)
    np.save(dense_dir / f"{stem}_doa.npy", doa_arr)
    np.save(dense_dir / f"{stem}_loud.npy", loud_arr)
    np.save(dense_dir / f"{stem}_mask.npy", mask_arr)

    # ---- JSON ---------------------------------------------------------------
    json_dir = output_dir / "annotations" / split
    json_dir.mkdir(parents=True, exist_ok=True)
    json_data = _build_json(scene_id, meta, sources_sorted)
    with open(json_dir / f"scene_{scene_id}.json", "w") as f:
        json.dump(json_data, f, indent=2)


def _build_json(
    scene_id: str,
    meta: SceneMeta,
    sources_sorted: list[SourceMeta],
) -> dict:
    sources_json = []
    for src in sources_sorted:
        traj_json = [
            {
                "frame": int(p.frame),
                "azimuth_deg": round(math.degrees(p.az_rad), 2),
                "elevation_deg": round(math.degrees(p.el_rad), 2),
            }
            for p in src.trajectory
        ]
        sources_json.append({
            "source_id": int(src.source_idx),
            "class_id": int(src.class_id),
            "class_name": src.class_name,
            "onset_frame": int(src.onset_frame),
            "offset_frame": int(src.offset_frame),
            "trajectory": traj_json,
        })

    return {
        "scene_id": scene_id,
        "audio_file": f"audio/{meta.split}/scene_{scene_id}.wav",
        "sample_rate": 48000,
        "duration_sec": 45.0,
        "synthesis_meta": {
            "room": meta.srir_room,
            "srir_rt60_idx": meta.srir_rt60_idx,
            "srir_dist_idx": meta.srir_dist_idx,
            "hrtf_subject": meta.hrtf_subject,
            "snr_db": round(meta.snr_db, 2),
        },
        "frame_config": {
            "hop_sec": 0.02,
            "total_frames": meta.n_frames,
        },
        "sources": sources_json,
    }
