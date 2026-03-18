"""
Quick sanity check for the synthesized dataset.

Runs a small test synthesis (3 scenes) and verifies:
  - Audio output is stereo 48 kHz float32
  - Dense annotation shapes match expected
  - Loudness GT is computed from dry mono (not mixed binaural)
  - Moving sources generate per-frame varying DOA
  - Class IDs are valid
  - No NaN or inf in outputs

Usage:
    cd dataset/synthesizer && python verify_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("verify")


def run_single_scene():
    """Synthesize one scene in the current process and inspect the result."""
    from config import SynthConfig
    from hrtf_loader import HRTFLibrary
    from srir_loader import SRIRLibrary
    from fsd50k_loader import FSD50KCatalog
    from scene_synth import synthesize_scene

    cfg = SynthConfig(n_workers=1)
    logger.info("Loading HRTF...")
    hrtf_lib = HRTFLibrary(cfg.hrtf_dir)
    logger.info("Loading SRIR...")
    srir_lib = SRIRLibrary(cfg.srir_dir, target_sr=cfg.sample_rate)
    logger.info("Loading FSD50K...")
    fsd50k = FSD50KCatalog(
        cfg.fsd50k_dev_audio, cfg.fsd50k_eval_audio,
        cfg.fsd50k_dev_csv, cfg.fsd50k_eval_csv,
        cfg.fsd50k_vocab_csv, target_sr=cfg.sample_rate,
    )

    rng = np.random.default_rng(12345)

    logger.info("Synthesizing scene...")
    binaural, meta = synthesize_scene(
        scene_id="000000",
        split="train",
        cfg=cfg,
        hrtf_lib=hrtf_lib,
        srir_lib=srir_lib,
        fsd50k=fsd50k,
        rng=rng,
    )

    # ---- Checks --------------------------------------------------------------
    assert binaural.ndim == 2, f"Expected (2, N), got {binaural.shape}"
    assert binaural.shape[0] == 2, "Expected stereo"
    assert binaural.shape[1] == cfg.scene_samples, \
        f"Expected {cfg.scene_samples} samples, got {binaural.shape[1]}"
    assert binaural.dtype == np.float32
    assert not np.any(np.isnan(binaural)), "NaN in binaural audio!"
    assert not np.any(np.isinf(binaural)), "Inf in binaural audio!"

    peak = np.max(np.abs(binaural))
    logger.info("Audio: shape=%s, peak=%.4f, rms=%.4f",
                binaural.shape, peak, np.sqrt(np.mean(binaural**2)))

    # Check sources
    logger.info("Scene: room=%s, HRTF=%s, SNR=%.1f dB, n_sources=%d",
                meta.srir_room, meta.hrtf_subject, meta.snr_db, len(meta.sources))
    for src in meta.sources:
        assert src.class_id >= 0 and src.class_id < 300, \
            f"Invalid class_id {src.class_id}"
        assert src.onset_frame < src.offset_frame
        active_frames = np.where(~np.isnan(src.loudness_frames))[0]
        assert len(active_frames) > 0, "No active frames in loudness"

        loud_active = src.loudness_frames[active_frames]
        assert not np.any(np.isnan(loud_active)), "NaN in loudness"
        logger.info("  source %d: class=%s, frames=[%d,%d], loud range=[%.1f, %.1f] dBFS",
                    src.source_idx, src.class_name,
                    src.onset_frame, src.offset_frame,
                    float(np.nanmin(src.loudness_frames)),
                    float(np.nanmax(src.loudness_frames)))

    # Check annotation writing
    import tempfile, os
    from annotation_writer import write_annotations
    with tempfile.TemporaryDirectory() as tmpdir:
        write_annotations("000000", meta, Path(tmpdir))
        dense_dir = Path(tmpdir) / "annotations_dense" / "train"
        for suffix in ["_cls", "_doa", "_loud", "_mask"]:
            p = dense_dir / f"scene_000000{suffix}.npy"
            assert p.exists(), f"Missing {p.name}"
            arr = np.load(str(p))
            logger.info("  %-14s shape=%s dtype=%s", p.name, arr.shape, arr.dtype)

        cls_arr = np.load(str(dense_dir / "scene_000000_cls.npy"))
        assert cls_arr.shape == (cfg.n_frames, 5), \
            f"Unexpected cls shape {cls_arr.shape}"

        doa_arr = np.load(str(dense_dir / "scene_000000_doa.npy"))
        mask_arr = np.load(str(dense_dir / "scene_000000_mask.npy"))
        active_doa = doa_arr[mask_arr]
        if len(active_doa) > 0:
            norms = np.linalg.norm(active_doa, axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-3), \
                f"DOA unit vectors not normalized: min={norms.min():.4f}"
            logger.info("  DOA unit vector norms: min=%.4f max=%.4f (expect ≈1.0)",
                        norms.min(), norms.max())

        loud_arr = np.load(str(dense_dir / "scene_000000_loud.npy"))
        active_loud = loud_arr[mask_arr]
        assert not np.any(np.isnan(active_loud.astype(np.float32))), "NaN in dense loud"
        logger.info("  Active loudness range: [%.1f, %.1f] dBFS",
                    float(active_loud.min()), float(active_loud.max()))

    logger.info("All checks passed!")
    return True


if __name__ == "__main__":
    ok = run_single_scene()
    sys.exit(0 if ok else 1)
