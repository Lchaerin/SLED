# SLED Dataset — Binaural Synthetic Dataset

> 생성 기준: 2026-03-18
> 학습 대상: SLED (Sound Localization & Event Detection) 모델
> 입력 형식: Stereo binaural 44.1→48kHz, 2채널 WAV
> 목표 규모: Train 10,000 + Val 1,000 + Test 500 scenes × 45s = **144시간**

---

## 1. 디렉토리 구조

```
dataset/
├── audio/
│   ├── train/   scene_000000.wav … scene_009999.wav   (stereo 16-bit 48 kHz)
│   ├── val/     scene_010000.wav … scene_010999.wav
│   └── test/    scene_011000.wav … scene_011499.wav
│
├── annotations/              (JSON — 디버깅/시각화용)
│   ├── train/   scene_000000.json …
│   ├── val/
│   └── test/
│
├── annotations_dense/        (numpy binary — 학습용 고속 I/O)
│   ├── train/
│   │   ├── scene_000000_cls.npy    [T, 5]     int16   class_id (-1=empty)
│   │   ├── scene_000000_doa.npy    [T, 5, 3]  float16 unit vector (x,y,z)
│   │   ├── scene_000000_loud.npy   [T, 5]     float16 dBFS per source
│   │   └── scene_000000_mask.npy   [T, 5]     bool    slot active
│   ├── val/
│   └── test/
│
├── meta/
│   ├── class_map.json        class_id(0-199) → label, + 300="__empty__"
│   ├── hrtf_registry.json    HRTF subject list with n_positions, sample_rate
│   ├── split.json            train/val/test scene_id lists
│   └── progress_{split}.txt  합성 진행 체크포인트 (자동 생성)
│
├── sources/                  (원본 데이터 — 합성에 사용, 변경 없음)
│   ├── HRTF/                 SONICOM HRTF (140 subjects, .sofa)
│   ├── SRIR/
│   │   ├── TAU-SRIR_DB/      9 rooms × conditions (.mat HDF5)
│   │   └── TAU-SNoise_DB/    Room ambience FOA recordings
│   └── sound_effects/        FSD50K (200 classes, ~51K clips)
│
├── synthesizer/              합성 코드 (아래 §3 참조)
└── torch_dataset.py          PyTorch DataLoader
```

---

## 2. Scene 구성

### 2.1 오디오

| 항목 | 값 |
|---|---|
| Sample rate | 48,000 Hz |
| Channels | 2 (binaural L/R) |
| Bit depth | 16-bit PCM WAV |
| Scene 길이 | 45 s (2,160,000 samples) |
| Frame hop | 20 ms (960 samples) → T = 2,250 frames |

### 2.2 음원 수 분포

| 음원 수 | 비율 |
|---|---|
| 0 (ambient only) | 5% |
| 1 | 25% |
| 2 | 30% |
| 3 | 25% |
| 4 | 10% |
| 5 | 5% |

1~3개가 80%로 가장 많음 (Curriculum 학습에 유리).

### 2.3 음원 위치

- **Azimuth**: 균등 랜덤 [−π, π] (SLED CW convention: 0=front, +=right)
- **Elevation**: 균등 랜덤 [−45°, +45°]
- **이동 음원**: 전체 음원의 약 25%가 움직임
  - 최대 4개 waypoint로 선형 보간
  - Annotation에 per-frame DOA unit vector 기록

### 2.4 Onset / Offset

- 약 40% 확률로 씬 내 일부 구간만 활성화
- 최소 2초 / 최대 30초

---

## 3. 합성 파이프라인

### 3.1 흐름도

```
FSD50K mono clip
  │   (resample to 48 kHz, loop/crop to 45 s)
  │
  ├─── 1. 위치 배정 (az, el) + 이동 trajectory 생성
  │
  ├─── 2. Loudness GT 계산 ← dry mono per-frame RMS → dBFS
  │         (합성 이전 개별 음원 기준; SLED 모델 regression target)
  │
  ├─── 3. SRIR 룩업 (TAU-SRIR_DB)
  │         → 가장 가까운 azimuth의 FOA RIR (4ch, 48 kHz, ~300 ms)
  │
  ├─── 4. FOA → Binaural 변환 (BF Filter, precomputed per HRTF subject)
  │         → BRIR (2, ~14,655 samples)
  │
  └─── 5. Overlap-Add 합성 (oaconvolve)
          → binaural spatialized source (2, 45 s + reverb tail)

All sources → Mix (sum)
  │
  ├─── 6. TAU-SNoise ambient noise 추가
  │         (FOA partial read → binaural decode → SNR scaling [5–25 dB])
  │
  └─── 7. Peak normalize → stereo WAV + JSON + dense .npy 저장
```

### 3.2 FOA → Binaural 변환 원리

**Virtual Loudspeaker (VLS) 방법:**

1. Golden spiral로 50-point VLS 그리드 생성 (구 위에 균일 분포)
2. 각 VLS 위치에서 FOA encoding vector: `[W, Y, Z, X]` (ACN/SN3D)
3. **Binaural FOA Filter (BF)** 사전 계산 (HRTF subject당 1회):
   ```
   BF[foa_ch, ear, :] = Σ_k (4π/N_vls) × Y[k, foa_ch] × HRTF_k[ear, :]
   ```
   `BF shape: (4, 2, 256)`  — HRTF subject당 캐시됨
4. **BRIR 합성** (SRIR position당):
   ```
   BRIR[ear] = Σ_j convolve(SRIR_foa[j], BF[j, ear])
   ```
   `BRIR shape: (2, 14,655 samples)`
5. **Spatialization**: `mono_audio ★ BRIR` (overlap-add)

**효과**: SRIR이 실제 공간의 room acoustics(잔향, 조기반사)를 제공하고, HRTF가 머리 형태에 따른 방향 단서(ITD, ILD)를 제공. 두 요소가 물리적으로 올바르게 결합됨.

### 3.3 Loudness Ground Truth

```
per_source_loudness[frame] = 20 × log10( RMS( mono_dry[frame×hop : (frame+1)×hop] ) )
```

- **mix된 binaural이 아닌** 각 음원의 dry mono 기준
- 씬 합성 전에 계산하므로 공간화 효과에 무관
- Inactive frame → NaN (dense .npy에서 mask==False인 slot)

---

## 4. 데이터 소스

### 4.1 SONICOM HRTF Dataset

| 항목 | 값 |
|---|---|
| Subjects | 140개 (.sofa 파일) |
| Positions per subject | 828 |
| HRIR length | 256 samples (at 48 kHz = 5.3 ms) |
| Sample rate | 48,000 Hz |
| Format | SOFA (HDF5) |
| 좌표 | azimuth 0–360° CCW (SOFA) → CW 변환 후 사용 |

### 4.2 TAU-SRIR Database

| 항목 | 값 |
|---|---|
| Rooms | 9개 (bomb_shelter, gym, pb132, pc226, sa203, sc203, se203, tb103, tc352) |
| 원본 SR | 24,000 Hz → 48,000 Hz 리샘플 |
| FOA channels | 4 (ACN order: W, Y, Z, X) |
| RIR length | 7,200 samples @ 24 kHz → 14,400 @ 48 kHz (300 ms) |
| Conditions | 156개 (room × RT60 × source_distance) |
| 원형 rooms (360 az) | bomb_shelter, gym, pb132, pc226, tc352 |
| 비원형 rooms | sa203 (72), sc203 (61), se203 (93), tb103 (84) |

**TAU-SNoise_DB**: 각 room에 대응하는 FOA ambient noise (~28분 길이, 24 kHz)
합성 시 partial read로 필요한 구간만 로드해 I/O 최소화.

### 4.3 FSD50K

| 항목 | 값 |
|---|---|
| 클래스 수 | 200 (vocabulary.csv, index 0–199) |
| 총 클립 수 | ~51,000 (dev 40,966 + eval 10,231) |
| 활성 클래스 | 199개 (모든 파일 존재 확인 후) |
| 오디오 형식 | 다양한 SR → 48 kHz 리샘플, 모노 다운믹스 |

---

## 5. Annotation 스키마

### 5.1 JSON (annotations/{split}/scene_{id}.json)

```json
{
  "scene_id": "000000",
  "audio_file": "audio/train/scene_000000.wav",
  "sample_rate": 48000,
  "duration_sec": 45.0,
  "synthesis_meta": {
    "room": "bomb_shelter",
    "srir_rt60_idx": 3,
    "srir_dist_idx": 0,
    "hrtf_subject": "p0042",
    "snr_db": 14.7
  },
  "frame_config": {
    "hop_sec": 0.02,
    "total_frames": 2250
  },
  "sources": [
    {
      "source_id": 0,
      "class_id": 42,
      "class_name": "Dog",
      "onset_frame": 100,
      "offset_frame": 1800,
      "trajectory": [
        {"frame": 100, "azimuth_deg": 45.0, "elevation_deg": 10.0},
        {"frame": 950, "azimuth_deg": -30.0, "elevation_deg": 5.0},
        {"frame": 1800, "azimuth_deg": -30.0, "elevation_deg": 5.0}
      ]
    }
  ]
}
```

### 5.2 Dense Binary (annotations_dense/{split}/scene_{id}_*.npy)

| 파일 | Shape | Dtype | 설명 |
|---|---|---|---|
| `_cls.npy` | [T, 5] | int16 | class_id, -1=inactive |
| `_doa.npy` | [T, 5, 3] | float16 | (x, y, z) unit vector |
| `_loud.npy` | [T, 5] | float16 | dBFS of dry mono per frame |
| `_mask.npy` | [T, 5] | bool | slot active flag |

**Slot 배정**: onset_frame 기준 오름차순 정렬 (Hungarian matching은 학습 시 수행).

**DOA unit vector 변환**:
```
x = cos(el) × cos(az)    (forward)
y = cos(el) × sin(az)    (right, az+=right)
z = sin(el)              (up)
```
학습 시 역변환: `azimuth = atan2(y, x)`, `elevation = asin(z)`

---

## 6. 합성 코드 구조

```
dataset/synthesizer/
├── config.py              SynthConfig dataclass (paths, params)
├── hrtf_loader.py         HRTFLibrary: SONICOM SOFA 로더 + KD-tree 위치검색
├── srir_loader.py         SRIRLibrary: TAU-SRIR_DB 로더 + azimuth 매핑
├── fsd50k_loader.py       FSD50KCatalog: 클래스별 샘플링, 리샘플, 정규화
├── binaural_render.py     BinauralRenderer: BF filter 계산, BRIR 합성
├── scene_synth.py         synthesize_scene(): 씬 합성 오케스트레이터
├── annotation_writer.py   write_annotations(): JSON + .npy 저장
├── build_meta.py          class_map.json, hrtf_registry.json 생성
├── run_synthesis.py       multiprocessing 메인 (spawn Pool)
├── verify_dataset.py      단일 씬 합성 + 검증 스크립트
└── requirements.txt       numpy, scipy, soundfile, h5py, tqdm, torch
```

### 핵심 모듈 역할

**`binaural_render.py`** — 물리 기반 binaural 렌더링
- `compute_binaural_foa_filters(hrtf_subject, n_vls=50)` → BF `(4, 2, 256)`
  - HRTF subject당 1회만 계산, scene 합성 중 재사용
- `build_brir(srir_foa, bf)` → BRIR `(2, 14655)`: 8번의 FFT 합성곱
- `spatialize(mono, brir)` → binaural: **`oaconvolve`** 사용 (긴 신호 × 짧은 필터에 최적)

**`scene_synth.py`** — 씬 합성
- `synthesize_scene()`: SRIR/HRTF/FSD50K를 조합해 1개 씬 생성
- `_spatialize_with_trajectory()`: 블록 기반 시변 렌더링
  - static source → 1 BRIR × 1 convolution
  - moving source → 16 blocks × 1 BRIR each (캐싱으로 중복 계산 방지)
- `_load_snoise()`: 노이즈 파일 partial read (28분 파일 전체 로드 회피)

**`run_synthesis.py`** — 병렬 합성
- `mp.Pool(spawn)`: 각 워커가 독립적인 HRTF/SRIR/FSD50K 인스턴스 소유
- worker init에서 HRTF 전체 pre-load (~450 MB/worker)
- 씬ID ↔ seed 결정적 매핑 → 완전 재현 가능, 부분 재생성 가능
- `progress_{split}.txt` 체크포인트 → `--resume`으로 중단 후 재개

---

## 7. 학습용 PyTorch DataLoader

```python
# dataset/torch_dataset.py
from dataset.torch_dataset import build_dataloader

train_loader = build_dataloader(
    dataset_root="/path/to/dataset",
    split="train",
    batch_size=32,
    window_frames=256,      # 256 × 20ms = 5.12 s 랜덤 윈도우
    augment_scs=True,       # Stereo Channel Swap (L↔R + azimuth 부호 반전)
    num_workers=16,
    prefetch_factor=4,
    pin_memory=True,
)
```

**반환 item:**

| Key | Shape | Dtype | 설명 |
|---|---|---|---|
| `audio` | [2, N] | float32 | stereo waveform (5.12 s) |
| `cls` | [T, 5] | int64 | class ids |
| `doa` | [T, 5, 3] | float32 | DOA unit vectors |
| `loud` | [T, 5] | float32 | dBFS |
| `mask` | [T, 5] | bool | active slot mask |
| `scene_id` | str | — | 디버깅용 |

`audio` → 모델 preprocessor에서 5채널 변환:
```python
# SLED 모델 입력: [B, 5, 64, T]
# [L-mel, R-mel, cos(IPD), sin(IPD), ILD]
```

**Data Augmentation (DataLoader 내):**

| 기법 | 위치 | 적용률 |
|---|---|---|
| **SCS** (Stereo Channel Swap) | `torch_dataset.py` | 50% (train) |
| **SpecAugment** (time/freq mask) | 모델 preprocessor 이후 | 학습 중 적용 권장 |
| **Mixup** | Trainer 레벨 | 선택 사항 |

---

## 8. 합성 실행

```bash
cd dataset/synthesizer

# 전체 데이터셋 합성 (16 worker, ~12분 예상)
python run_synthesis.py --split all --workers 16

# 재개 (중단 후)
python run_synthesis.py --split train --resume --workers 16

# 검증 (1개 씬 합성 + 형식 확인)
python verify_dataset.py

# 메타 파일만 재빌드 (split.json 갱신 등)
python build_meta.py
```

**성능 (실측):**

| 항목 | 값 |
|---|---|
| 45초 씬 합성 시간 (single core) | ~1 sec |
| 16 워커 처리량 | ~16 scenes/sec |
| 전체 11,500 씬 예상 시간 | ~12분 |

---

## 9. 데이터 규모

| 구성 | 수량 | 오디오 용량 |
|---|---|---|
| Train | 10,000 scenes × 45 s | ~86 GB |
| Val | 1,000 scenes | ~8.6 GB |
| Test | 500 scenes | ~4.3 GB |
| Dense annotations | 11,500 scenes | ~1.4 GB |
| JSON annotations | 11,500 scenes | ~57 MB |
| **총계** | **144시간** | **~100 GB** |

---

## 10. 주의사항 및 설계 결정

### Loudness GT
합성 전 **dry mono 기준** RMS를 측정. 공간화(잔향, HRTF 레벨 변화)에 의존하지 않아야 SLED 모델이 intrinsic loudness를 학습할 수 있음.

### SRIR 좌표 규약
- TAU-SRIR DB: 원형 배열 room의 360개 position = azimuth 0°~359° (CCW)
- 코드 내부에서 SLED CW convention (0=front, +=right)으로 변환
- 비원형 room(sa203 등): 측정된 trajectory position에서 랜덤 선택

### HRTF 좌표 규약
- SOFA azimuth: 0=front, 90=left (CCW) → `az_sled = -az_sofa` 변환
- 828개 측정 위치 → KD-tree (3D unit vector)로 최근접 위치 검색

### on-the-fly vs pre-synthesized
현재 구현은 **pre-synthesized** (디스크에 미리 저장).
학습 중 on-the-fly 합성은 CPU 병목(특히 HRTF pre-load ~450 MB/worker)을 야기하므로 권장하지 않음.
데이터 다양성 증대가 필요하면 pre-synthesized 데이터를 추가로 생성할 것.

### FSD50K 클래스 → SLED 클래스
FSD50K: 200 클래스 (ID 0–199)
SLED 모델 설계: 300 클래스 + 1 empty
→ 현재 데이터셋은 FSD50K 200 클래스 사용. 추가 데이터소스(AudioSet segments 등) 확보 시 301개 full class 활용 가능.

### 이동 음원 렌더링
블록 기반 (16 blocks per source): 매 block마다 midpoint azimuth에서 BRIR 계산.
동일 SRIR bin이면 이전 BRIR 재사용(캐싱) → static source = 1 BRIR.
엄밀한 time-varying convolution 대신 piecewise-stationary approximation.
블록 경계에서의 불연속은 잔향이 masking함.
