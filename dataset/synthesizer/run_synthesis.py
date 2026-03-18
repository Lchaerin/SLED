"""
Main synthesis runner.

Usage:
    python run_synthesis.py                       # synthesize all splits
    python run_synthesis.py --split train         # train only
    python run_synthesis.py --workers 8           # override worker count
    python run_synthesis.py --resume              # skip already-done scenes

Pipeline (per worker process):
    1. Each worker owns its own copy of HRTF/SRIR/FSD50K objects (no shared state).
    2. Scenes are assigned by a deterministic scene_id ↔ seed mapping
       → fully reproducible; any subset can be re-generated.
    3. Results are written to disk as soon as a scene is done.
    4. A checkpoint file (meta/progress_{split}.txt) records completed IDs.

I/O layout produced:
    dataset/
    ├── audio/{split}/scene_{id:06d}.wav          stereo 16-bit 48 kHz
    ├── annotations/{split}/scene_{id:06d}.json
    ├── annotations_dense/{split}/
    │   ├── scene_{id:06d}_cls.npy   [T,5]   int16
    │   ├── scene_{id:06d}_doa.npy   [T,5,3] float16
    │   ├── scene_{id:06d}_loud.npy  [T,5]   float16
    │   └── scene_{id:06d}_mask.npy  [T,5]   bool
    └── meta/
        ├── class_map.json
        ├── hrtf_registry.json
        └── split.json

After synthesis, run build_meta.py once to refresh split.json.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add synthesizer dir to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SynthConfig, DEFAULT_CFG
from hrtf_loader import HRTFLibrary
from srir_loader import SRIRLibrary
from fsd50k_loader import FSD50KCatalog
from scene_synth import synthesize_scene
from annotation_writer import write_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("run_synthesis")


# ---- Worker initializer: load heavy objects once per process ----------------

_worker_state: dict = {}


def _worker_init(cfg: SynthConfig) -> None:
    """Called once per worker process. Loads HRTF, SRIR, FSD50K into RAM."""
    pid = os.getpid()
    logger.info("Worker %d: loading data...", pid)
    hrtf_lib = HRTFLibrary(cfg.hrtf_dir)
    # Pre-load all HRTF subjects into this worker's memory (~450 MB each worker)
    hrtf_lib.load_all()
    srir_lib = SRIRLibrary(cfg.srir_dir, target_sr=cfg.sample_rate)
    fsd50k = FSD50KCatalog(
        dev_audio_dir=cfg.fsd50k_dev_audio,
        eval_audio_dir=cfg.fsd50k_eval_audio,
        dev_csv=cfg.fsd50k_dev_csv,
        eval_csv=cfg.fsd50k_eval_csv,
        vocab_csv=cfg.fsd50k_vocab_csv,
        target_sr=cfg.sample_rate,
    )
    _worker_state["hrtf_lib"] = hrtf_lib
    _worker_state["srir_lib"] = srir_lib
    _worker_state["fsd50k"] = fsd50k
    _worker_state["cfg"] = cfg
    logger.info("Worker %d: ready", pid)


def _worker_synthesize(args: tuple) -> str:
    """
    Synthesize one scene. Returns scene_id on success, raises on error.
    args = (scene_id, split, cfg)
    """
    scene_id, split, cfg = args
    cfg_local: SynthConfig = _worker_state["cfg"]
    hrtf_lib: HRTFLibrary = _worker_state["hrtf_lib"]
    srir_lib: SRIRLibrary = _worker_state["srir_lib"]
    fsd50k: FSD50KCatalog = _worker_state["fsd50k"]

    seed = cfg_local.seed_base + _split_offset(split) + int(scene_id)
    rng = np.random.default_rng(seed)

    binaural, meta = synthesize_scene(
        scene_id=scene_id,
        split=split,
        cfg=cfg_local,
        hrtf_lib=hrtf_lib,
        srir_lib=srir_lib,
        fsd50k=fsd50k,
        rng=rng,
    )

    # Write audio
    audio_dir = cfg_local.output_dir / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"scene_{scene_id}.wav"
    # binaural: (2, N) float32 → transpose to (N, 2)
    sf.write(
        str(wav_path),
        binaural.T,
        samplerate=cfg_local.sample_rate,
        subtype="PCM_16",
    )

    # Write annotations
    write_annotations(
        scene_id=scene_id,
        meta=meta,
        output_dir=cfg_local.output_dir,
    )

    return scene_id


def _split_offset(split: str) -> int:
    return {"train": 0, "val": 1_000_000, "test": 2_000_000}.get(split, 0)


def _load_done(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()
    with open(progress_file) as f:
        return set(line.strip() for line in f if line.strip())


def _mark_done(progress_file: Path, scene_id: str) -> None:
    with open(progress_file, "a") as f:
        f.write(scene_id + "\n")


def synthesize_split(
    split: str,
    n_scenes: int,
    cfg: SynthConfig,
    n_workers: int,
    resume: bool,
) -> None:
    """Synthesize all scenes for one split using a multiprocessing pool."""
    # Scene IDs: zero-padded 6-digit strings
    id_start = {"train": 0, "val": 10_000, "test": 11_000}[split]
    scene_ids = [f"{id_start + i:06d}" for i in range(n_scenes)]

    progress_file = cfg.meta_dir / f"progress_{split}.txt"
    done = _load_done(progress_file) if resume else set()
    todo = [sid for sid in scene_ids if sid not in done]
    logger.info("Split %s: %d/%d scenes to generate", split, len(todo), n_scenes)

    if not todo:
        return

    args_list = [(sid, split, cfg) for sid in todo]
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(cfg,),
    ) as pool:
        try:
            from tqdm import tqdm
            results = tqdm(
                pool.imap_unordered(_worker_synthesize, args_list),
                total=len(args_list),
                desc=f"[{split}]",
            )
        except ImportError:
            results = pool.imap_unordered(_worker_synthesize, args_list)

        for scene_id in results:
            _mark_done(progress_file, scene_id)

    elapsed = time.time() - t0
    logger.info(
        "Split %s done: %d scenes in %.1f min (%.2f scenes/sec)",
        split, len(todo), elapsed / 60, len(todo) / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SLED binaural dataset synthesis")
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="all")
    parser.add_argument("--workers", type=int, default=DEFAULT_CFG.n_workers)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed scenes")
    parser.add_argument("--n-train", type=int, default=DEFAULT_CFG.n_train)
    parser.add_argument("--n-val", type=int, default=DEFAULT_CFG.n_val)
    parser.add_argument("--n-test", type=int, default=DEFAULT_CFG.n_test)
    args = parser.parse_args()

    cfg = SynthConfig(
        n_workers=args.workers,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )
    cfg.meta_dir.mkdir(parents=True, exist_ok=True)

    # Build meta files first (class_map, hrtf_registry)
    from build_meta import build_all
    build_all(cfg)

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]
    n_map = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}

    for split in splits_to_run:
        synthesize_split(
            split=split,
            n_scenes=n_map[split],
            cfg=cfg,
            n_workers=args.workers,
            resume=args.resume,
        )

    # Refresh split.json
    from build_meta import build_split_json
    import json
    split_json = build_split_json(cfg)
    with open(cfg.meta_dir / "split.json", "w") as f:
        json.dump(split_json, f, indent=2)
    logger.info("Dataset synthesis complete.")


if __name__ == "__main__":
    main()
