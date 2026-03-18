"""
TAU-SRIR DB loader.

File structure per room (HDF5/MATLAB v7.3):
  rirs/foa:  (N_rt60, N_dist) object array of references
  Each reference → array (N_az, 4, N_samples)
    N_az:      number of source azimuths (360 for circular rooms, 72-93 for linear)
    4:         FOA channels in ACN/SN3D order: W(0), Y(1), Z(2), X(3)
    N_samples: 7200 at 24 kHz (= 300 ms)

All SRIRs are captured at 24 kHz and are resampled to 48 kHz on load.

FOA ACN channel ordering:
  ch0 = W = 1                       (omnidirectional)
  ch1 = Y = sin(az) * cos(el)       (left/right)
  ch2 = Z = sin(el)                 (up/down)
  ch3 = X = cos(az) * cos(el)       (front/back)
  where az/el follow acoustic CCW convention (0=front, 90=left)

Azimuth index → azimuth angle mapping for circular rooms:
  index i corresponds to azimuth i degrees (CCW, 0=front)
  Converted to SLED CW convention: az_sled_deg = -i (mod 360)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

# Azimuth range for circular rooms (360 uniformly spaced points at 1°)
_CIRCULAR_N_AZ = 360

# Rooms with circular (360°) trajectory coverage
CIRCULAR_ROOMS = {
    "bomb_shelter": "rirs_01_bomb_shelter.mat",
    "gym": "rirs_02_gym.mat",
    "pb132": "rirs_03_pb132.mat",
    "pc226": "rirs_04_pc226.mat",
    "tc352": "rirs_10_tc352.mat",
}

# Rooms with non-circular trajectories — limited azimuth coverage
TRAJECTORY_ROOMS = {
    "sa203": "rirs_05_sa203.mat",
    "sc203": "rirs_06_sc203.mat",
    "se203": "rirs_08_se203.mat",
    "tb103": "rirs_09_tb103.mat",
}


@dataclass
class SRIRCondition:
    """Metadata for one loaded SRIR condition."""
    room: str
    rt60_idx: int        # index into the RT60 axis
    dist_idx: int        # index into the distance axis
    rirs: np.ndarray     # (N_az, 4, N_samples_48k) float32
    n_az: int
    is_circular: bool    # True → azimuth index = angle in degrees


class SRIRLibrary:
    """
    Loads and caches TAU-SRIR_DB room impulse responses.
    Each room is loaded on first access and kept in memory.
    """

    def __init__(self, srir_dir: Path, target_sr: int = 48_000):
        self.srir_dir = Path(srir_dir)
        self.target_sr = target_sr
        self.src_sr = 24_000
        self._conditions: list[SRIRCondition] = []
        self._loaded_rooms: set[str] = set()
        self._load_all_rooms()

    def _load_all_rooms(self) -> None:
        all_rooms = {**CIRCULAR_ROOMS, **TRAJECTORY_ROOMS}
        for room_name, filename in all_rooms.items():
            path = self.srir_dir / filename
            if not path.exists():
                logger.warning("SRIR file not found: %s", path)
                continue
            self._load_room(room_name, path, circular=(room_name in CIRCULAR_ROOMS))
        logger.info("Loaded %d SRIR conditions from %d rooms",
                    len(self._conditions), len(self._loaded_rooms))

    def _load_room(self, room_name: str, path: Path, circular: bool) -> None:
        logger.debug("Loading SRIR room: %s", room_name)
        with h5py.File(path, "r") as f:
            foa_refs = f["rirs/foa"]          # (N_rt60, N_dist) object array
            n_rt60, n_dist = foa_refs.shape
            for rt60_idx in range(n_rt60):
                for dist_idx in range(n_dist):
                    ref = foa_refs[rt60_idx, dist_idx]
                    rir_raw = f[ref][:]        # (N_az, 4, N_samples) float64
                    rir = self._resample(rir_raw.astype(np.float32))
                    cond = SRIRCondition(
                        room=room_name,
                        rt60_idx=rt60_idx,
                        dist_idx=dist_idx,
                        rirs=rir,
                        n_az=rir.shape[0],
                        is_circular=circular,
                    )
                    self._conditions.append(cond)
        self._loaded_rooms.add(room_name)

    def _resample(self, rir: np.ndarray) -> np.ndarray:
        """Resample (N_az, 4, N_src) from src_sr to target_sr."""
        if self.src_sr == self.target_sr:
            return rir
        ratio = self.target_sr / self.src_sr
        n_out = int(rir.shape[2] * ratio)
        # Resample each (N_az, 4) slice along time axis
        return scipy.signal.resample(rir, n_out, axis=2).astype(np.float32)

    @property
    def n_conditions(self) -> int:
        return len(self._conditions)

    def random_condition(self, rng: np.random.Generator) -> SRIRCondition:
        idx = int(rng.integers(0, len(self._conditions)))
        return self._conditions[idx]

    def get_rir_for_azimuth(
        self,
        cond: SRIRCondition,
        az_sled_rad: float,
    ) -> np.ndarray:
        """
        Return (4, N_samples) FOA RIR for the given source azimuth.
        az_sled_rad: azimuth in SLED convention (CW, 0=front, +right), radians.
        """
        if cond.is_circular:
            # Convert SLED CW az → CCW degrees → index
            az_ccw_deg = -np.rad2deg(az_sled_rad) % 360  # 0–360
            az_idx = int(round(az_ccw_deg)) % _CIRCULAR_N_AZ
        else:
            # Non-circular: use nearest among measured azimuths
            # Positions are stored in the order of measurement trajectories
            # We sample uniformly from all available positions
            az_idx = int(rng_placeholder := np.random.randint(0, cond.n_az))  # noqa
        return cond.rirs[az_idx]  # (4, N_samples)

    def get_rir_for_azimuth_with_rng(
        self,
        cond: SRIRCondition,
        az_sled_rad: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Thread-safe version with explicit RNG for non-circular rooms."""
        if cond.is_circular:
            az_ccw_deg = -np.rad2deg(az_sled_rad) % 360
            az_idx = int(round(az_ccw_deg)) % _CIRCULAR_N_AZ
        else:
            az_idx = int(rng.integers(0, cond.n_az))
        return cond.rirs[az_idx].copy()  # (4, N_samples)
