"""
Scene synthesis: assembles one binaural scene from SRIR, HRTF, and FSD50K sources.

Scene layout
------------
Duration: 45 s at 48 kHz  (2,160,000 samples)
Sources:  0–5, biased towards 1–3 (per source_count_probs in config)

For each source:
  * class_id drawn from FSD50K active classes
  * onset/offset: some sources don't span the full scene
  * azimuth: uniform in [-π, π]  (SLED CW convention)
  * elevation: uniform in [elevation_min, elevation_max] (degrees)
  * trajectory: 25 % of sources move linearly or follow smooth arc

Loudness GT (per source, per frame)
------------------------------------
  Measured on the **dry mono** audio before spatialization.
  RMS of the source's mono signal in each 20-ms hop window → dBFS.
  This is what the SLED model regresses.

Ambient noise
-------------
  Random TAU-SNoise_DB room noise decoded to binaural.
  SNR randomly chosen per scene from [5, 25] dB.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf

from config import SynthConfig
from hrtf_loader import HRTFLibrary
from srir_loader import SRIRLibrary, SRIRCondition
from fsd50k_loader import FSD50KCatalog, _normalize_rms
from binaural_render import BinauralRenderer, spatialize, build_brir, compute_binaural_foa_filters

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    frame: int
    az_rad: float   # SLED CW convention
    el_rad: float


@dataclass
class SourceMeta:
    source_idx: int          # 0-4
    class_id: int
    class_name: str
    onset_frame: int
    offset_frame: int        # exclusive
    trajectory: list[TrajectoryPoint]
    mono_audio: np.ndarray   # full 45-s mono (zeros outside onset/offset)
    loudness_frames: np.ndarray  # (T,) dBFS per frame (NaN outside active)


@dataclass
class SceneMeta:
    scene_id: str
    split: str
    srir_room: str
    srir_rt60_idx: int
    srir_dist_idx: int
    hrtf_subject: str
    snr_db: float
    sources: list[SourceMeta]
    n_frames: int


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return -120.0
    return float(20.0 * math.log10(rms))


def _frame_loudness(
    mono: np.ndarray,
    onset_frame: int,
    offset_frame: int,
    hop: int,
    n_frames: int,
) -> np.ndarray:
    """
    Compute per-frame RMS dBFS of dry mono audio.
    Returns (n_frames,) float32, NaN outside [onset, offset).
    """
    loudness = np.full(n_frames, np.nan, dtype=np.float32)
    for f in range(onset_frame, min(offset_frame, n_frames)):
        s = f * hop
        e = s + hop
        segment = mono[s:e]
        if len(segment) < hop:
            segment = np.pad(segment, (0, hop - len(segment)))
        loudness[f] = _rms_dbfs(segment)
    return loudness


def _interpolate_trajectory(
    trajectory: list[TrajectoryPoint],
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate azimuth/elevation over all n_frames.
    Returns (az_frames, el_frames) each (n_frames,) float32.
    """
    frames = np.array([p.frame for p in trajectory], dtype=float)
    azs = np.array([p.az_rad for p in trajectory], dtype=float)
    els = np.array([p.el_rad for p in trajectory], dtype=float)
    t = np.arange(n_frames, dtype=float)
    az_interp = np.interp(t, frames, azs).astype(np.float32)
    el_interp = np.interp(t, frames, els).astype(np.float32)
    return az_interp, el_interp


def _load_snoise(
    snoise_dir: Path,
    room_name: str,
    rng: np.random.Generator,
    target_sr: int,
    duration_samples: int,
) -> Optional[np.ndarray]:
    """
    Load TAU-SNoise_DB ambient noise for a room (partial read for speed).

    The noise files are ~28 minutes long at 24 kHz. We read only the needed
    duration plus a small padding, so we avoid loading the whole file.
    Returns (4, duration_samples) float32 FOA noise, or None if not found.
    """
    room_subdir_patterns = {
        "bomb_shelter": "01_bomb",
        "gym": "02_gym",
        "pb132": "03_pb132",
        "pc226": "04_pc226",
        "sa203": "05_sa203",
        "sc203": "06_sc203",
        "se203": "08_se203",
        "tb103": "09_tb103",
        "tc352": "10_tc352",
    }
    pattern = room_subdir_patterns.get(room_name)
    if pattern is None:
        return None
    dirs = [d for d in snoise_dir.iterdir() if d.is_dir() and pattern in d.name]
    if not dirs:
        return None
    wav_path = dirs[0] / "ambience_foa_sn3d_24k_edited.wav"
    if not wav_path.exists():
        return None

    try:
        info = sf.info(str(wav_path))
        src_sr = info.samplerate                    # 24000
        total_frames = info.frames                  # total samples in file

        # We need duration_samples at target_sr → convert to src_sr frames
        ratio = target_sr / src_sr
        # Request slightly more than needed to account for resampling edge effects
        need_src = int(np.ceil(duration_samples / ratio)) + src_sr  # +1s padding
        need_src = min(need_src, total_frames)

        # Random start position in source file
        max_start = max(0, total_frames - need_src)
        start_frame = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

        data, sr = sf.read(
            str(wav_path), dtype="float32", always_2d=True,
            start=start_frame, frames=need_src,
        )
    except Exception:
        return None

    audio = data.T.astype(np.float32)  # (4, N_src)

    # Resample to target_sr
    if src_sr != target_sr:
        n_out = int(audio.shape[1] * ratio)
        audio = scipy.signal.resample(audio, n_out, axis=1).astype(np.float32)

    # Crop to exact duration
    if audio.shape[1] < duration_samples:
        reps = duration_samples // audio.shape[1] + 1
        audio = np.tile(audio, (1, reps))
    return audio[:, :duration_samples]


def synthesize_scene(
    scene_id: str,
    split: str,
    cfg: SynthConfig,
    hrtf_lib: HRTFLibrary,
    srir_lib: SRIRLibrary,
    fsd50k: FSD50KCatalog,
    rng: np.random.Generator,
) -> tuple[np.ndarray, SceneMeta]:
    """
    Synthesize one binaural scene.

    Returns:
        binaural: (2, N_samples) float32
        meta: SceneMeta with all ground-truth information
    """
    hop = cfg.hop_samples
    n_frames = cfg.n_frames
    n_samples = cfg.scene_samples
    sr = cfg.sample_rate

    # ---- 1. Pick SRIR room condition ----------------------------------------
    srir_cond: SRIRCondition = srir_lib.random_condition(rng)

    # ---- 2. Pick HRTF subject ------------------------------------------------
    hrtf_idx = hrtf_lib.random_subject_idx(rng)
    hrtf_subj = hrtf_lib.get(hrtf_idx)
    renderer = BinauralRenderer(hrtf_subj, n_vls=cfg.n_vls)

    # ---- 3. Pick number of sources ------------------------------------------
    n_sources = int(rng.choice(
        len(cfg.source_count_probs), p=cfg.source_count_probs
    ))

    # ---- 4. Synthesize each source ------------------------------------------
    mix = np.zeros((2, n_samples + srir_cond.rirs.shape[2] + 256), dtype=np.float32)
    source_metas: list[SourceMeta] = []

    for src_idx in range(n_sources):
        # 4a. Sample class + audio
        target_dbfs = float(rng.uniform(
            cfg.source_level_min_dbfs, cfg.source_level_max_dbfs
        ))
        mono_full, class_id = fsd50k.sample_clip(
            rng=rng,
            duration_sec=cfg.scene_duration,
            target_dbfs=target_dbfs,
        )

        # 4b. Onset / offset
        has_onset_offset = rng.random() < cfg.onset_offset_prob
        if has_onset_offset:
            max_dur = min(cfg.max_source_duration, cfg.scene_duration - cfg.min_source_duration)
            dur_sec = float(rng.uniform(cfg.min_source_duration, max(cfg.min_source_duration + 0.1, max_dur)))
            max_onset = max(0.0, cfg.scene_duration - dur_sec)
            onset_sec = float(rng.uniform(0, max_onset)) if max_onset > 0 else 0.0
            onset_frame = int(onset_sec / cfg.hop_sec)
            offset_frame = min(n_frames, onset_frame + int(dur_sec / cfg.hop_sec))
        else:
            onset_frame = 0
            offset_frame = n_frames

        onset_sample = onset_frame * hop
        offset_sample = offset_frame * hop

        # Zero out the source outside its active window
        mono_active = np.zeros(n_samples, dtype=np.float32)
        end = min(offset_sample, n_samples)
        mono_active[onset_sample:end] = mono_full[onset_sample:end]

        # 4c. Trajectory
        is_moving = rng.random() < cfg.moving_source_prob
        az_start = float(rng.uniform(-np.pi, np.pi))
        el_start = float(np.deg2rad(rng.uniform(
            cfg.elevation_min_deg, cfg.elevation_max_deg
        )))

        if is_moving:
            # Random arc: 2–4 waypoints within active window
            n_waypoints = int(rng.integers(2, 5))
            wpt_frames = sorted(rng.integers(
                onset_frame, max(onset_frame + 1, offset_frame),
                size=n_waypoints
            ).tolist())
            # Unique frames
            wpt_frames = sorted(set([onset_frame] + wpt_frames + [offset_frame - 1]))
            trajectory = []
            az_cur, el_cur = az_start, el_start
            for wf in wpt_frames:
                trajectory.append(TrajectoryPoint(frame=wf, az_rad=az_cur, el_rad=el_cur))
                az_cur = float(rng.uniform(-np.pi, np.pi))
                el_cur = float(np.deg2rad(rng.uniform(
                    cfg.elevation_min_deg, cfg.elevation_max_deg
                )))
        else:
            trajectory = [
                TrajectoryPoint(frame=onset_frame, az_rad=az_start, el_rad=el_start),
                TrajectoryPoint(frame=max(offset_frame - 1, onset_frame), az_rad=az_start, el_rad=el_start),
            ]

        # 4d. Spatialize: for each active frame, accumulate binaural output
        #     We convolve in chunks per unique (azimuth, elevation) segment.
        az_frames, el_frames = _interpolate_trajectory(trajectory, n_frames)

        # Group consecutive frames with same SRIR index (1° quantization)
        # and render in bulk segments for efficiency
        binaural_src = _spatialize_with_trajectory(
            mono_active=mono_active,
            az_frames=az_frames,
            el_frames=el_frames,
            onset_frame=onset_frame,
            offset_frame=offset_frame,
            srir_cond=srir_cond,
            renderer=renderer,
            srir_lib=srir_lib,
            rng=rng,
            hop=hop,
            n_samples=n_samples,
        )

        # Accumulate into mix (binaural_src may be longer due to reverb tail)
        mix_len = min(mix.shape[1], binaural_src.shape[1])
        mix[:, :mix_len] += binaural_src[:, :mix_len]

        # 4e. Loudness GT (per-frame, dry mono before spatialization)
        loudness_frames = _frame_loudness(
            mono_full, onset_frame, offset_frame, hop, n_frames
        )

        source_metas.append(SourceMeta(
            source_idx=src_idx,
            class_id=class_id,
            class_name=fsd50k.class_id_to_label[class_id],
            onset_frame=onset_frame,
            offset_frame=offset_frame,
            trajectory=trajectory,
            mono_audio=mono_full,
            loudness_frames=loudness_frames,
        ))

    # ---- 5. Add ambient noise -----------------------------------------------
    snr_db = float(rng.uniform(*cfg.snr_range))
    mix_final = _add_ambient_noise(
        mix=mix[:, :n_samples],
        snr_db=snr_db,
        srir_cond=srir_cond,
        renderer=renderer,
        cfg=cfg,
        rng=rng,
    )

    # ---- 6. Final clip to scene length and normalize -------------------------
    mix_final = mix_final[:, :n_samples]
    peak = float(np.max(np.abs(mix_final)))
    if peak > 1.0:
        mix_final = mix_final / peak * 0.99

    scene_meta = SceneMeta(
        scene_id=scene_id,
        split=split,
        srir_room=srir_cond.room,
        srir_rt60_idx=srir_cond.rt60_idx,
        srir_dist_idx=srir_cond.dist_idx,
        hrtf_subject=hrtf_lib._paths[hrtf_idx].stem,
        snr_db=snr_db,
        sources=source_metas,
        n_frames=n_frames,
    )
    return mix_final, scene_meta


def _spatialize_with_trajectory(
    mono_active: np.ndarray,
    az_frames: np.ndarray,
    el_frames: np.ndarray,
    onset_frame: int,
    offset_frame: int,
    srir_cond: SRIRCondition,
    renderer: BinauralRenderer,
    srir_lib: SRIRLibrary,
    rng: np.random.Generator,
    hop: int,
    n_samples: int,
    n_time_blocks: int = 16,
) -> np.ndarray:
    """
    Spatialize mono audio using a block-based BRIR rendering strategy.

    The source's active window is divided into n_time_blocks equal blocks.
    For each block, the BRIR is computed at the block's midpoint azimuth.
    Adjacent blocks with the same SRIR bin share the same BRIR (static sources
    = 1 block = 1 convolution). Moving sources get up to n_time_blocks BRIRs.

    Block audio is rendered with oaconvolve (optimal for long signal × short filter)
    and overlap-added into the output buffer.

    Returns (2, n_samples + N_brir - 1) float32.
    """
    n_brir = srir_cond.rirs.shape[2] + renderer.bf.shape[2] - 1
    out_len = n_samples + n_brir
    out = np.zeros((2, out_len), dtype=np.float32)

    if onset_frame >= offset_frame:
        return out

    active_frames = offset_frame - onset_frame
    n_blocks = min(n_time_blocks, active_frames)
    block_frames = active_frames / n_blocks

    def _az_srir_idx(az_rad: float) -> int:
        if srir_cond.is_circular:
            az_ccw = (-az_rad * 180.0 / np.pi) % 360.0
            return int(round(az_ccw)) % 360
        else:
            return int(rng.integers(0, srir_cond.n_az))

    last_az_idx: int = -1
    cached_brir: np.ndarray = None  # reused when az_idx unchanged

    for b in range(n_blocks):
        f_start = onset_frame + int(b * block_frames)
        f_end   = onset_frame + int((b + 1) * block_frames)
        f_end   = min(f_end, offset_frame)
        f_mid   = (f_start + f_end) // 2

        s_start = f_start * hop
        s_end   = min(f_end * hop, n_samples)
        if s_end <= s_start:
            continue

        az  = float(az_frames[min(f_mid, len(az_frames) - 1)])
        az_idx = _az_srir_idx(az)

        if az_idx != last_az_idx or cached_brir is None:
            srir_foa  = srir_cond.rirs[az_idx]           # (4, N_srir)
            cached_brir = build_brir(srir_foa, renderer.bf)  # (2, N_brir)
            last_az_idx = az_idx

        segment = mono_active[s_start:s_end]
        seg_bin = spatialize(segment, cached_brir)        # oaconvolve inside

        end_out  = min(s_start + seg_bin.shape[1], out_len)
        seg_clip = end_out - s_start
        out[:, s_start:end_out] += seg_bin[:, :seg_clip]

    return out


def _add_ambient_noise(
    mix: np.ndarray,
    snr_db: float,
    srir_cond: SRIRCondition,
    renderer: BinauralRenderer,
    cfg: SynthConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Decode TAU-SNoise FOA ambient noise to binaural and add at target SNR.
    Falls back to white noise if SNoise file not found.
    """
    n_samples = mix.shape[1]
    sr = cfg.sample_rate

    # Load ambient noise FOA
    noise_foa = _load_snoise(
        cfg.snoise_dir, srir_cond.room, rng, sr, n_samples
    )

    if noise_foa is None:
        # Fallback: shaped white noise (diffuse field approximation)
        # Encode as omnidirectional (W only) white noise
        white = rng.standard_normal(n_samples).astype(np.float32) * 0.01
        noise_foa = np.zeros((4, n_samples), dtype=np.float32)
        noise_foa[0] = white

    # Decode FOA noise to binaural using oaconvolve (long signal × short filter)
    noise_binaural = np.zeros((2, n_samples), dtype=np.float32)
    for j in range(4):
        for ear in range(2):
            conv = scipy.signal.oaconvolve(
                noise_foa[j, :n_samples], renderer.bf[j, ear]
            ).astype(np.float32)
            conv_len = min(len(conv), n_samples)
            noise_binaural[ear, :conv_len] += conv[:conv_len]

    # Scale noise to achieve target SNR
    mix_rms = float(np.sqrt(np.mean(mix ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise_binaural ** 2)))
    if mix_rms < 1e-9 or noise_rms < 1e-9:
        return mix
    target_noise_rms = mix_rms * 10 ** (-snr_db / 20.0)
    noise_binaural = noise_binaural * (target_noise_rms / noise_rms)

    return mix + noise_binaural
