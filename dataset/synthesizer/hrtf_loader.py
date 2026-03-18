"""
SONICOM HRTF Dataset loader.

Each .sofa file contains:
  Data.IR:         (M, 2, N)  M=828 positions, 2 ears, N=256 tap HRIR at 48 kHz
  SourcePosition:  (M, 3)     (azimuth_deg, elevation_deg, radius_m)
                               azimuth: 0=front, 90=LEFT (SOFA CCW)
                               elevation: -90=below, 0=horizontal, +90=above

Loaded data uses SLED convention:
  azimuth:   0=front, positive=RIGHT (CW), range [-π, π]
  elevation: 0=horizontal, positive=UP, range [-π/2, π/2]
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class HRTFSubject:
    """HRIR data for a single SONICOM subject."""

    def __init__(self, path: Path):
        self.path = path
        self.subject_id = path.stem  # e.g. "p0001"
        self._load(path)

    def _load(self, path: Path) -> None:
        with h5py.File(path, "r") as f:
            # Data.IR: (M, 2, N) — left ear is index 0, right ear is index 1
            self.hrir = f["Data.IR"][:].astype(np.float32)    # (828, 2, 256)
            sr = float(f["Data.SamplingRate"][0])
            assert int(sr) == 48_000, f"Unexpected HRTF sample rate {sr}"
            pos_raw = f["SourcePosition"][:]                   # (828, 3) degrees

        # SOFA convention → SLED convention
        # SOFA azimuth: CCW, 0=front, 90=left
        # SLED azimuth: CW, 0=front, +right → negate azimuth
        az_deg_sofa = pos_raw[:, 0]   # 0–360
        el_deg = pos_raw[:, 1]        # -90 to +90

        # Convert azimuth to [-180, 180] then negate for CW
        az_deg_cw = -(az_deg_sofa % 360)          # CCW→CW sign flip
        # Map to [-180, 180]
        az_deg_cw = (az_deg_cw + 180) % 360 - 180

        self.az_rad = np.deg2rad(az_deg_cw).astype(np.float32)   # [-π, π]
        self.el_rad = np.deg2rad(el_deg).astype(np.float32)       # [-π/2, π/2]

        # Build 3-D unit vectors for fast KD-tree lookup
        self._xyz = _angles_to_xyz(self.az_rad, self.el_rad)     # (M, 3)
        self._kdtree = cKDTree(self._xyz)

    def find_nearest(self, az_rad: float, el_rad: float) -> int:
        """Return index of closest measured HRTF position."""
        q = _angles_to_xyz(np.array([az_rad]), np.array([el_rad]))
        _, idx = self._kdtree.query(q)
        return int(idx[0])

    def get_hrir(self, pos_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (hrir_L, hrir_R) each [256] float32."""
        return self.hrir[pos_idx, 0], self.hrir[pos_idx, 1]


class HRTFLibrary:
    """
    Lazy-loading library of all SONICOM subjects.
    Call .load_all() to pre-cache everything into RAM (~450 MB).
    """

    def __init__(self, hrtf_dir: Path):
        self.hrtf_dir = Path(hrtf_dir)
        self._paths = sorted(
            p for p in self.hrtf_dir.glob("p*.sofa")
        )
        if not self._paths:
            raise FileNotFoundError(f"No .sofa files found in {hrtf_dir}")
        self._cache: dict[int, HRTFSubject] = {}
        logger.info("HRTFLibrary: found %d subjects", len(self._paths))

    @property
    def n_subjects(self) -> int:
        return len(self._paths)

    def get(self, subject_idx: int) -> HRTFSubject:
        if subject_idx not in self._cache:
            self._cache[subject_idx] = HRTFSubject(self._paths[subject_idx])
        return self._cache[subject_idx]

    def load_all(self) -> None:
        """Pre-load all subjects into RAM."""
        logger.info("Loading all %d HRTF subjects...", len(self._paths))
        for i in range(len(self._paths)):
            self.get(i)
        logger.info("All HRTF subjects loaded.")

    def random_subject_idx(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self._paths)))


def _angles_to_xyz(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Convert azimuth/elevation (radians) to unit Cartesian vectors."""
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=-1)
