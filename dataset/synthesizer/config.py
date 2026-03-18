"""
Synthesis configuration for SLED binaural dataset.

Coordinate conventions:
  - Azimuth: 0 = front, positive = RIGHT (clockwise from above), range [-π, π]
             consistent with SLED model: atan2(y, x) where y-axis = right
  - Elevation: 0 = horizontal, positive = UP, range [-π/2, π/2]

TAU-SRIR DB:
  - FOA channels in ACN order: ch0=W, ch1=Y, ch2=Z, ch3=X
  - 360 source positions for circular-trajectory rooms (0-359°, CCW convention)
  - Sampled at 24 kHz → resampled to 48 kHz in loader

SONICOM HRTF:
  - SOFA format, 828 positions, 256-tap HRIR at 48 kHz
  - SourcePosition: (azimuth_deg, elevation_deg, radius_m)
  - Azimuth: 0=front, 90=LEFT (CCW convention) → converted to CW on load

FSD50K:
  - 200 sound classes, ~51K clips
  - Used as mono dry sources
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent  # dataset/


@dataclass
class SynthConfig:
    # ---- Paths ----------------------------------------------------------------
    srir_dir: Path = BASE_DIR / "sources" / "SRIR" / "TAU-SRIR_DB"
    snoise_dir: Path = BASE_DIR / "sources" / "SRIR" / "TAU-SNoise_DB"
    hrtf_dir: Path = BASE_DIR / "sources" / "HRTF"
    fsd50k_dev_audio: Path = BASE_DIR / "sources" / "sound_effects" / "FSD50K.dev_audio"
    fsd50k_eval_audio: Path = BASE_DIR / "sources" / "sound_effects" / "FSD50K.eval_audio"
    fsd50k_dev_csv: Path = BASE_DIR / "sources" / "sound_effects" / "FSD50K.ground_truth" / "dev.csv"
    fsd50k_eval_csv: Path = BASE_DIR / "sources" / "sound_effects" / "FSD50K.ground_truth" / "eval.csv"
    fsd50k_vocab_csv: Path = BASE_DIR / "sources" / "sound_effects" / "FSD50K.ground_truth" / "vocabulary.csv"
    output_dir: Path = BASE_DIR
    meta_dir: Path = BASE_DIR / "meta"

    # ---- Audio parameters -----------------------------------------------------
    sample_rate: int = 48_000
    srir_sample_rate: int = 24_000  # TAU-SRIR is captured at 24 kHz
    hrtf_sample_rate: int = 48_000  # SONICOM is at 48 kHz
    scene_duration: float = 45.0   # seconds per scene

    # ---- Frame parameters (must match SLED model) ----------------------------
    hop_sec: float = 0.02          # 20 ms hop → 960 samples at 48 kHz
    window_sec: float = 0.04       # 40 ms window (not used in synthesis, just for reference)

    # ---- Dataset splits -------------------------------------------------------
    n_train: int = 10_000
    n_val: int = 1_000
    n_test: int = 500

    # ---- Source count distribution (index = n_sources, prob sums to 1.0) -----
    # 0 sources: 5%  (ambient-only scenes for robustness)
    # 1-3 sources: 80%  (most common)
    # 4-5 sources: 15%
    source_count_probs: List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.30, 0.25, 0.10, 0.05]
    )

    # ---- Moving source probability (per source) ------------------------------
    moving_source_prob: float = 0.25

    # ---- SNR range for ambient noise (dB) ------------------------------------
    snr_range: Tuple[float, float] = (5.0, 25.0)

    # ---- Per-source level range (dBFS of mono source) ------------------------
    source_level_min_dbfs: float = -30.0
    source_level_max_dbfs: float = -6.0

    # ---- Elevation range (degrees) -------------------------------------------
    elevation_min_deg: float = -45.0
    elevation_max_deg: float = 45.0

    # ---- Virtual loudspeaker count for FOA→binaural decoding -----------------
    # 50 points from golden-spiral; adequate for 1st-order ambisonics
    n_vls: int = 50

    # ---- Onset / offset handling ---------------------------------------------
    # Probability that a source does not span the full scene
    onset_offset_prob: float = 0.4
    min_source_duration: float = 2.0   # seconds
    max_source_duration: float = 30.0  # seconds

    # ---- I/O ------------------------------------------------------------------
    n_workers: int = 16
    seed_base: int = 42

    # ---- Output audio format -------------------------------------------------
    output_bit_depth: int = 16   # PCM-16 WAV

    @property
    def hop_samples(self) -> int:
        return int(self.sample_rate * self.hop_sec)

    @property
    def scene_samples(self) -> int:
        return int(self.sample_rate * self.scene_duration)

    @property
    def n_frames(self) -> int:
        return int(self.scene_duration / self.hop_sec)


# Singleton for convenience
DEFAULT_CFG = SynthConfig()
