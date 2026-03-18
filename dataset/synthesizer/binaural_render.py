"""
Binaural rendering engine: FOA SRIR + HRTF → BRIR.

Pipeline for each source at (az, el):
  1. Look up FOA SRIR at closest azimuth in the selected room condition.
     SRIR:  (4, N_srir)  at 48 kHz, ACN order W/Y/Z/X
  2. Compute binaural FOA decoder filters from the HRTF:
     For each virtual loudspeaker k at (az_k, el_k):
       vls_weights[k, j] = FOA encoding of speaker k for channel j
     Combined binaural FOA filter: BF[j, ear] = Σ_k vls_weights[k,j] * HRIR_k[ear]
     BF shape: (4, 2, N_hrir)
  3. Convolve SRIR channel j with BF[j, ear] and sum → BRIR: (2, N_brir)
  4. Convolve dry mono audio with BRIR → binaural room-reverberant signal.

Virtual loudspeaker grid:
  50-point golden-spiral on the unit sphere, adequate for 1st-order ambisonics.
  Each VLS uses nearest-neighbor HRTF measurement.

FOA encoding (ACN/SN3D, amplitude-normalised):
  ch0 (W):  1
  ch1 (Y):  sin(az) * cos(el)
  ch2 (Z):  sin(el)
  ch3 (X):  cos(az) * cos(el)
  where az, el are in the SLED CW convention (0=front, +=right/up).

The VLS gain normalization factor 4π/N_vls ensures energy preservation.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import scipy.signal

from hrtf_loader import HRTFSubject

logger = logging.getLogger(__name__)


def _golden_spiral_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n approximately uniform points on the unit sphere.
    Returns (az_rad, el_rad) arrays in SLED CW convention.
    """
    indices = np.arange(n, dtype=float) + 0.5
    phi_colat = np.arccos(1.0 - 2.0 * indices / n)   # colatitude 0→π
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices     # azimuth (0→...)

    el_rad = np.pi / 2.0 - phi_colat                  # elevation -π/2 → π/2
    az_rad = theta % (2.0 * np.pi)
    # Map az to [-π, π]
    az_rad = np.where(az_rad > np.pi, az_rad - 2.0 * np.pi, az_rad)
    return az_rad.astype(np.float32), el_rad.astype(np.float32)


def _foa_encode(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """
    Compute FOA (ACN/SN3D) encoding vectors.
    Returns Y: (N, 4) for [W, Y, Z, X] channels.
    """
    N = len(az)
    Y = np.zeros((N, 4), dtype=np.float32)
    Y[:, 0] = 1.0                                    # W
    Y[:, 1] = np.sin(az) * np.cos(el)               # Y (ACN 1)
    Y[:, 2] = np.sin(el)                             # Z (ACN 2)
    Y[:, 3] = np.cos(az) * np.cos(el)               # X (ACN 3)
    return Y


def compute_binaural_foa_filters(
    hrtf_subject: HRTFSubject,
    n_vls: int = 50,
) -> np.ndarray:
    """
    Precompute binaural FOA decoder filters for one HRTF subject.

    Returns BF: (4, 2, N_hrir) float32
      BF[foa_ch, ear, :] is the combined binaural filter for FOA channel foa_ch.

    Interpretation:
      For binaural output:
        binaural[ear] = Σ_j convolve(SRIR_foa[j], BF[j, ear])
    """
    n_hrir = hrtf_subject.hrir.shape[2]          # 256
    vls_az, vls_el = _golden_spiral_sphere(n_vls)

    # FOA encoding matrix for VLS positions: (N_vls, 4)
    Y = _foa_encode(vls_az, vls_el)

    # Sampling decoder weights: g_k = (4π / N_vls) * Y[k, :]
    # Sum of decoded signal at a VLS → energy-correct binaural output
    weight = (4.0 * np.pi / n_vls)

    BF = np.zeros((4, 2, n_hrir), dtype=np.float32)
    for k in range(n_vls):
        pos_idx = hrtf_subject.find_nearest(float(vls_az[k]), float(vls_el[k]))
        hrir_l, hrir_r = hrtf_subject.get_hrir(pos_idx)  # each (256,)
        for j in range(4):
            w_kj = weight * float(Y[k, j])
            BF[j, 0] += w_kj * hrir_l
            BF[j, 1] += w_kj * hrir_r

    return BF  # (4, 2, N_hrir)


def build_brir(
    srir_foa: np.ndarray,   # (4, N_srir) at 48 kHz
    bf: np.ndarray,         # (4, 2, N_hrir)
) -> np.ndarray:
    """
    Convolve FOA SRIR channels with binaural FOA filters → BRIR (2, N_brir).
    N_srir and N_hrir are both short (≤14400, 256), so standard fftconvolve is fine.
    """
    n_srir = srir_foa.shape[1]
    n_hrir = bf.shape[2]
    n_out = n_srir + n_hrir - 1
    brir = np.zeros((2, n_out), dtype=np.float32)
    for j in range(4):
        for ear in range(2):
            brir[ear] += scipy.signal.fftconvolve(
                srir_foa[j], bf[j, ear]
            ).astype(np.float32)
    return brir


def spatialize(
    mono: np.ndarray,      # (N_audio,) dry mono at 48 kHz
    brir: np.ndarray,      # (2, N_brir) binaural room impulse response
) -> np.ndarray:
    """
    Convolve dry mono audio with BRIR → binaural (2, N_audio+N_brir-1).
    Uses overlap-add (oaconvolve) which is optimal for long signal × short filter.
    """
    n_out = len(mono) + brir.shape[1] - 1
    out = np.zeros((2, n_out), dtype=np.float32)
    for ear in range(2):
        out[ear] = scipy.signal.oaconvolve(mono, brir[ear]).astype(np.float32)
    return out


class BinauralRenderer:
    """
    Per-scene renderer. Holds a precomputed binaural FOA filter (BF) for
    the scene's HRTF subject, enabling fast multi-source rendering.

    Usage:
        renderer = BinauralRenderer(hrtf_subject, n_vls=50)
        brir = renderer.get_brir(srir_foa, az_rad, el_rad)
        binaural = spatialize(mono_audio, brir)
    """

    def __init__(self, hrtf_subject: HRTFSubject, n_vls: int = 50):
        self.hrtf = hrtf_subject
        self.bf = compute_binaural_foa_filters(hrtf_subject, n_vls)
        logger.debug("BinauralRenderer: BF shape %s", self.bf.shape)

    def get_brir(
        self,
        srir_foa: np.ndarray,   # (4, N_srir)
        az_sled_rad: float,
        el_sled_rad: float,
    ) -> np.ndarray:
        """
        Build BRIR from SRIR + BF.

        The SRIR already encodes the room response for the given azimuth.
        The BF provides head-related filtering including elevation cues
        (decoded from virtual loudspeakers spanning the full sphere).

        Note: room acoustics in the SRIR are at elevation≈0; elevation
        cues come primarily from the HRTF (direct-sound component via VLS).
        """
        return build_brir(srir_foa, self.bf)
