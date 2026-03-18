"""
FSD50K audio dataset loader.

Catalog: dev.csv + eval.csv
  Columns: fname (int), labels (comma-separated), mids, split
  split ∈ {train, val, test}  (only in dev.csv)

vocabulary.csv:
  Columns: index, label, mid
  200 classes total, indices 0-199

Audio files: {fname}.wav, mono or stereo, variable sample rate.

This loader:
1. Parses the vocabulary to build class_id ↔ label mapping.
2. Parses dev+eval CSVs to build per-class file lists.
3. Provides stratified random sampling of audio clips, with:
   - Resampling to target_sr
   - Stereo→mono downmix
   - Random crop/loop to fill a requested duration
   - RMS normalization to a target dBFS level
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class FSD50KCatalog:
    """Catalog of FSD50K audio clips with per-class stratified sampling."""

    def __init__(
        self,
        dev_audio_dir: Path,
        eval_audio_dir: Path,
        dev_csv: Path,
        eval_csv: Path,
        vocab_csv: Path,
        target_sr: int = 48_000,
    ):
        self.dev_dir = Path(dev_audio_dir)
        self.eval_dir = Path(eval_audio_dir)
        self.target_sr = target_sr

        # 1. Build vocabulary
        self.class_id_to_label: dict[int, str] = {}
        self.label_to_class_id: dict[str, int] = {}
        self.mid_to_class_id: dict[str, int] = {}
        self._parse_vocab(vocab_csv)

        # 2. Build per-class file lists: class_id → list of (path, str)
        self._class_files: dict[int, list[Path]] = {
            cid: [] for cid in self.class_id_to_label
        }
        self._parse_csv(dev_csv, self.dev_dir)
        self._parse_csv(eval_csv, self.eval_dir)

        # Filter to existing files
        for cid in list(self._class_files.keys()):
            self._class_files[cid] = [
                p for p in self._class_files[cid] if p.exists()
            ]
            if not self._class_files[cid]:
                logger.debug("Class %d (%s) has no audio files",
                             cid, self.class_id_to_label[cid])

        # Active classes (have at least one file)
        self.active_classes: list[int] = [
            cid for cid, files in self._class_files.items() if files
        ]
        logger.info("FSD50K: %d active classes, %d total clips",
                    len(self.active_classes),
                    sum(len(v) for v in self._class_files.values()))

    def _parse_vocab(self, vocab_csv: Path) -> None:
        import csv
        with open(vocab_csv, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                cid = int(row[0])
                label = row[1].strip()
                mid = row[2].strip() if len(row) > 2 else ""
                self.class_id_to_label[cid] = label
                self.label_to_class_id[label] = cid
                if mid:
                    self.mid_to_class_id[mid] = cid

    def _parse_csv(self, csv_path: Path, audio_dir: Path) -> None:
        import csv
        if not csv_path.exists():
            logger.warning("CSV not found: %s", csv_path)
            return
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["fname"].strip()
                labels_str = row.get("labels", row.get("label", "")).strip()
                mids_str = row.get("mids", "").strip()
                path = audio_dir / f"{fname}.wav"

                # Resolve class_id from labels/mids
                # Use the first label that maps to a known class
                class_id = self._resolve_class(labels_str, mids_str)
                if class_id is not None:
                    self._class_files[class_id].append(path)

    def _resolve_class(
        self, labels_str: str, mids_str: str
    ) -> Optional[int]:
        """Return first matching class_id from the label/mid strings."""
        for mid in mids_str.split(","):
            mid = mid.strip()
            if mid in self.mid_to_class_id:
                return self.mid_to_class_id[mid]
        for label in labels_str.split(","):
            label = label.strip()
            if label in self.label_to_class_id:
                return self.label_to_class_id[label]
        return None

    def sample_clip(
        self,
        rng: np.random.Generator,
        duration_sec: float,
        target_dbfs: float = -20.0,
        class_id: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Return (mono_audio, class_id) at self.target_sr.
        Audio is looped/cropped to exactly duration_sec and
        normalized to target_dbfs (RMS).
        """
        if class_id is None:
            class_id = int(rng.choice(self.active_classes))

        files = self._class_files.get(class_id, [])
        if not files:
            # Fallback: pick another class
            class_id = int(rng.choice(self.active_classes))
            files = self._class_files[class_id]

        # Pick a random file
        path = files[int(rng.integers(0, len(files)))]
        audio = self._load_audio(path, rng, duration_sec)
        audio = _normalize_rms(audio, target_dbfs)
        return audio, class_id

    def _load_audio(
        self, path: Path, rng: np.random.Generator, duration_sec: float
    ) -> np.ndarray:
        """Load, resample, downmix, and loop/crop to duration_sec."""
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            logger.debug("Error reading %s: %s — returning silence", path, e)
            n = int(self.target_sr * duration_sec)
            return np.zeros(n, dtype=np.float32)

        # Stereo → mono
        audio = data.mean(axis=1)  # (N,)

        # Resample to target_sr
        if sr != self.target_sr:
            import scipy.signal
            n_out = int(len(audio) * self.target_sr / sr)
            audio = scipy.signal.resample(audio, n_out).astype(np.float32)

        # Loop or crop to desired length
        n_target = int(self.target_sr * duration_sec)
        audio = _loop_or_crop(audio, n_target, rng)
        return audio


def _loop_or_crop(
    audio: np.ndarray, n_target: int, rng: np.random.Generator
) -> np.ndarray:
    """Loop short clips, randomly crop long ones."""
    n = len(audio)
    if n == 0:
        return np.zeros(n_target, dtype=np.float32)
    if n >= n_target:
        start = int(rng.integers(0, n - n_target + 1))
        return audio[start: start + n_target]
    # Loop
    repeats = n_target // n + 1
    tiled = np.tile(audio, repeats)
    start = int(rng.integers(0, len(tiled) - n_target + 1))
    return tiled[start: start + n_target]


def _normalize_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Normalize audio RMS to target_dbfs."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_dbfs / 20.0)
    return (audio * (target_rms / rms)).astype(np.float32)
