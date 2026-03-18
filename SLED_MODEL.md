# SLED — Sound Localization & Event Detection Model

> 구현 레퍼런스 문서 (2026-03-18 기준)
> Binaural audio 입력 → 최대 5개 동시 음원의 (class, azimuth, elevation, loudness) 출력

---

## 모델 개요

실시간 binaural audio에서 음원을 감지하고 공간적 위치를 추정하는 모델.
Audio-VLA(SpatialAudio-SmolVLA)에 결합하여 사용하며, VLA에는 detection 결과뿐 아니라 encoder의 source embedding도 함께 전달한다.

**핵심 스펙:**
- 입력: 4채널 (L mel, R mel, IPD, ILD)
- 출력: 최대 5개 음원 × (class_id, azimuth, elevation, loudness, confidence) + source_embed
- 분류 클래스: 300 + 1 (empty)
- 파라미터: ~2.1M
- 목표 latency: <10ms/frame (GPU), <30ms/frame (CPU + INT8)

---

## 입력 전처리

### Binaural Audio → 4채널 변환

```
Raw binaural [2, T_samples]
  │
  ├─ STFT (n_fft=1024, hop=480, win=960)   ← 48kHz 기준 20ms hop, 40ms window
  │     → complex spectrogram [2, F, T_frames]
  │
  ├─ Left Mel-spectrogram:  log(|STFT_L| @ mel_bank)   → [1, 64, T]
  ├─ Right Mel-spectrogram: log(|STFT_R| @ mel_bank)   → [1, 64, T]
  ├─ IPD: angle(STFT_L * conj(STFT_R)) @ mel_bank      → [1, 64, T]
  └─ ILD: log(|STFT_L|) - log(|STFT_R|) @ mel_bank     → [1, 64, T]

Stack → [B, 4, 64, T]
```

### 파라미터 요약

| 항목 | 값 | 비고 |
|---|---|---|
| sample_rate | 48000 Hz | DCASE 표준 |
| n_fft | 1024 | |
| hop_length | 480 (10ms) 또는 960 (20ms) | 실시간 여유에 따라 선택 |
| win_length | 960 (20ms) 또는 1920 (40ms) | hop의 2배 |
| n_mels | 64 | |
| fmin / fmax | 50 / 22050 Hz | |

IPD는 frequency-dependent interaural phase를 인코딩하여 저주파 공간 정보를 제공하고,
ILD는 frequency-dependent interaural level difference를 인코딩하여 고주파 공간 정보를 제공한다.

---

## 모델 아키텍처

```
Input [B, 4, 64, T]
  │
  ▼
ConvBlock ×3                    ← Feature extraction
  4 → 32 → 64 → 128 channels
  freq축만 MaxPool, time축 유지
  │
  ▼
SE Block (channel attention)    ← IPD/ILD band 가중치 학습
  │
  ▼
Flatten freq → Linear(128×8, 128)
  [B, 128, T]
  │
  ▼
Conformer ×4                    ← Temporal modeling
  d_model=128, heads=4
  FFN_dim=512, conv_kernel=31
  │
  ▼                               ▼
BiFPN Neck (2 layers)        Attention Pool (per slot)
  multi-scale fusion              → source_embed [B, 5, 128]   ← VLA 전달용
  │
  ▼
Temporal Attention Pooling
  [B, 128, T] → [B, 128]
  │
  ▼
Broadcast to 5 slots → [B, 5, 128]
  │
  ├─→ Class Head:      FC(128 → 301)        softmax → class probabilities
  ├─→ DOA Head:        FC(128 → 3) → L2norm → unit vector (x, y, z)
  ├─→ Loudness Head:   FC(128 → 1)          → dB regression
  └─→ Confidence Head: FC(128 → 1) → σ      → [0, 1]
```

### ConvBlock 상세

```python
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool(freq only)"""
    def __init__(self, in_ch, out_ch):
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # freq축만 downsample

    def forward(self, x):  # x: [B, C, F, T]
        return self.pool(F.relu(self.bn(self.conv(x))))
```

3단 적용 후 freq축: 64 → 32 → 16 → 8. 출력: `[B, 128, 8, T]`.
Flatten: `[B, 128×8, T]` → `Linear(1024, 128)` → `[B, 128, T]`.

### SE Block (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    """Channel-wise attention after conv blocks"""
    def __init__(self, channels, reduction=16):
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):  # x: [B, C, F, T]
        s = x.mean(dim=(2, 3))          # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]
```

IPD/ILD 정보가 특정 frequency band에 집중되므로, SE block이 유용한 band를 강조한다.

### Conformer Layer

```python
class ConformerBlock(nn.Module):
    """FFN → MHSA → ConvModule → FFN (Macaron style)"""
    def __init__(self, d_model=128, n_heads=4, ffn_dim=512, conv_kernel=31):
        self.ffn1 = FeedForward(d_model, ffn_dim, dropout=0.1)
        self.mhsa = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.conv = ConvModule(d_model, conv_kernel)
        self.ffn2 = FeedForward(d_model, ffn_dim, dropout=0.1)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])

    def forward(self, x):  # x: [B, T, 128]
        x = x + 0.5 * self.ffn1(self.norms[0](x))
        x = x + self.mhsa(self.norms[1](x), self.norms[1](x), self.norms[1](x))[0]
        x = x + self.conv(self.norms[2](x))
        x = x + 0.5 * self.ffn2(self.norms[3](x))
        return x
```

4 layer × (128d MHSA + 128d depthwise conv + 512d FFN) ≈ 1M params.

### BiFPN Neck

Conv block 중간 출력(32ch, 64ch, 128ch scale)을 가져와서 2-layer BiFPN으로 top-down/bottom-up fusion.
다양한 크기의 음원 이벤트(짧은 충격음 vs 긴 환경음)를 동시에 잘 잡기 위함. ~100K params.

### Fixed-slot Detection Heads

Conformer 출력 `[B, 128, T]`에 temporal attention pooling을 적용한 후 5개 slot으로 broadcast.

**DOA head — unit vector 출력:**
azimuth/elevation을 직접 regression하면 angular wrapping 문제 발생.
unit vector `(x, y, z)` 3D Cartesian 출력 후 변환:
```python
def unit_vec_to_spherical(v):
    """v: [..., 3] unit vector → azimuth, elevation in radians"""
    x, y, z = v.unbind(-1)
    azimuth = torch.atan2(y, x)      # [-π, π]
    elevation = torch.asin(z.clamp(-1, 1))  # [-π/2, π/2]
    return azimuth, elevation
```

**Confidence head:**
각 slot에 실제 음원이 있는지에 대한 확신도. Inference 시 threshold(예: 0.5) + NMS 수행.
Empty class(class_id=300) 확률과 상호보완적으로 사용.

### Source Embedding 추출 (VLA 전달용)

```python
class SlotAttentionPool(nn.Module):
    """Conformer feature에서 slot별 attention-weighted pooling"""
    def __init__(self, d_model=128):
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

    def forward(self, feat, slot_features):
        # feat: [B, T, 128] (Conformer output)
        # slot_features: [B, 5, 128] (from detection heads)
        q = self.query(slot_features)     # [B, 5, 128]
        k = self.key(feat)                # [B, T, 128]
        attn = torch.bmm(q, k.transpose(1, 2)) / (128 ** 0.5)  # [B, 5, T]
        attn = attn.softmax(dim=-1)
        pooled = torch.bmm(attn, feat)    # [B, 5, 128]
        return pooled
```

이 128d source embedding은 class 정보 + acoustic texture를 모두 인코딩하고 있다.
VLA에 전달 시 class_id(discrete)보다 source_embed(continuous)가 더 풍부한 정보를 제공한다.

---

## SLED 출력 인터페이스

### Detection 결과 (list[list[dict]])

```python
# batch × sources
[
    [  # batch item 0
        {
            "class_id": 42,
            "class_name": "dog_bark",        # 디버깅/로깅용
            "azimuth": 0.3,                  # 라디안, 0=정면, +=우측
            "elevation": 0.1,                # 라디안, 0=수평, +=위
            "loudness": -12.3,               # dB
            "confidence": 0.95,
            "source_embed": Tensor[128],     # Conformer attention-pooled feature
        },
        ...  # 최대 5개
    ],
    ...  # batch
]
```

### VLA 전달용 추가 출력

```python
source_embeds: Tensor[B, 5, 128]   # slot별 attention-pooled Conformer feature
clap_embeds: Tensor[B, 5, 512]     # CLAP space projection (alignment adapter 사용 시)
```

---

## 학습 설정

### Loss 구성

Hungarian matching으로 prediction slot과 GT를 매칭한 뒤 합산:

| Head | Loss | Weight |
|---|---|---|
| Classification | Focal Loss (α=0.25, γ=2.0) | 1.0 |
| DOA | Cosine distance: `1 - cos(pred, gt)` | 1.0 |
| Loudness | Smooth L1 | 0.5 |
| Confidence | BCE | 1.0 |
| SCE auxiliary | MSE on `unit_doa × loudness` (학습 시에만) | 0.3 |

SCE auxiliary loss: DOA head의 unit vector에 loudness head 출력을 곱해서
"predicted source coordinate"를 만들고, GT와의 MSE로 DOA-loudness 일관성을 강화.
Inference 시에는 이 branch를 제거하므로 latency 증가 없음.

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.98) |
| LR | 1e-3 (peak), warmup 5 epochs |
| LR schedule | Cosine decay → 1e-5 |
| Batch size | 32~64 (GPU VRAM에 따라) |
| Epochs | 200~300 |
| Weight decay | 0.01 |
| Gradient clipping | max_norm=5.0 |
| Dropout | 0.1 (Conformer 내부) |

### Curriculum 학습

초기에는 동시 음원 수를 제한하고 점진적으로 늘림:
- Epoch 1~50: 최대 2개 음원
- Epoch 51~100: 최대 3개 음원
- Epoch 101~: 최대 5개 음원

---

## 데이터셋 구조

### 디렉토리 레이아웃

```
dataset/
├── meta/
│   ├── class_map.json            # class_id → label (300 classes)
│   ├── hrtf_registry.json        # HRTF DB metadata
│   └── split.json                # train/val/test file lists
├── audio/
│   ├── train/
│   │   ├── scene_000000.wav      # stereo (2ch), 48kHz, float32
│   │   └── ...
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   │   ├── scene_000000.json     # 메타데이터 + source 정보
│   │   └── ...
│   ├── val/
│   └── test/
├── annotations_dense/            # 학습용 바이너리 (pre-computed)
│   ├── train/
│   │   ├── scene_000000_cls.npy  # [T, 5] int16
│   │   ├── scene_000000_doa.npy  # [T, 5, 3] float16 (unit vector)
│   │   ├── scene_000000_loud.npy # [T, 5] float16 (dB)
│   │   └── scene_000000_mask.npy # [T, 5] bool (slot active)
│   └── ...
└── sources/                      # mono source 원본 (재합성용)
    ├── speech/
    ├── music/
    ├── environmental/
    └── ...
```

### Annotation JSON 스키마

```json
{
  "scene_id": "scene_000000",
  "audio_file": "audio/train/scene_000000.wav",
  "sample_rate": 48000,
  "duration_sec": 45.0,
  "synthesis_meta": {
    "room": {
      "dimensions_m": [6.2, 4.8, 3.0],
      "rt60_sec": 0.45,
      "materials": ["carpet", "plaster", "glass"]
    },
    "hrtf_id": "CIPIC_subject_003",
    "snr_db": 15.0,
    "noise_type": "ambient_cafe"
  },
  "frame_config": {
    "hop_sec": 0.02,
    "window_sec": 0.04,
    "total_frames": 2250
  },
  "sources": [
    {
      "source_id": 0,
      "class_id": 42,
      "class_name": "dog_bark",
      "mono_file": "sources/environmental/dog_bark_017.wav",
      "onset_frame": 100,
      "offset_frame": 350,
      "trajectory": [
        {"frame": 100, "azimuth_deg": 45.0, "elevation_deg": 0.0},
        {"frame": 350, "azimuth_deg": 50.0, "elevation_deg": 2.0}
      ]
    }
  ]
}
```

JSON은 메타데이터/디버깅용. 실제 학습에는 `.npy` dense 텐서를 사용하여 I/O 병목을 제거.

### 데이터 규모

| 구성 | 수량 |
|---|---|
| Train scenes | 10,000 (× 45초 ≈ 125시간) |
| Val scenes | 1,000 |
| Test scenes | 500 |
| 클래스당 최소 등장 | 200회 |

---

## 데이터 합성 파이프라인

### 합성 흐름

```
1. Mono source 선택 (FSD50K, AudioSet 등에서 클래스별 추출)
2. 랜덤 spatial position (azimuth, elevation) 배정
   - 일부 source에 trajectory 부여 (시간에 따라 위치 변화)
   - 일부 source에 onset/offset 설정 (중간 등장/퇴장)
3. HRIR convolution → binaural source
   - HRTF DB: MIT KEMAR, CIPIC, HUTUBS 등 복수 사용 (head shape 일반화)
4. Room simulation (pyroomacoustics 또는 Spatial Scaper)
   - 방 크기, RT60, 벽 재질 랜덤화
5. 여러 binaural source를 mix
   - SNR 랜덤화 (5~25 dB)
   - 배경 노이즈 추가
6. GT annotation 동시 생성 (JSON + .npy)
```

### Data Augmentation

| 기법 | 설명 | 효과 |
|---|---|---|
| **SCS (Stereo Channel Swap)** | L↔R swap + azimuth 부호 반전 | 데이터 2배, 무비용 |
| **SRIR 합성** | mono source × simulated room impulse response | 음향 환경 다양성 |
| **SpecAugment** | mel-spectrogram에 time/freq masking | 과적합 방지 |
| **Mixup** | 두 scene의 source를 합쳐서 새 scene 생성 | decision boundary smoothing |

#### SCS 적용 (4채널 대응)

```python
def stereo_channel_swap(mel_l, mel_r, ipd, ild, annotations):
    """L/R swap → spatial cue 반전, azimuth 부호 반전"""
    mel_l_new = mel_r
    mel_r_new = mel_l
    ipd_new = -ipd       # phase difference 반전
    ild_new = -ild        # level difference 반전

    for source in annotations:
        source["azimuth"] = -source["azimuth"]
        # DOA unit vector의 경우: y → -y
        if "doa_vec" in source:
            source["doa_vec"][1] = -source["doa_vec"][1]

    return mel_l_new, mel_r_new, ipd_new, ild_new, annotations
```

---

## 학습 시간 추정

10,000 scenes × 45초 평균, 프레임 hop 20ms 기준 scene당 ~2,250 frames.

### 1× GPU 기준 (100 epochs)

| GPU | Batch size | Scenes/sec | Epoch 소요 | 100 epochs |
|---|---|---|---|---|
| V100 (32GB) | 32 | ~18 | ~9.3 min | ~15.5 hrs |
| A100 (80GB) | 64 | ~52 | ~3.2 min | ~5.3 hrs |
| RTX 5090 (32GB) | 64 | ~40 | ~4.2 min | ~7.0 hrs |

### 수렴까지 (200~300 epochs)

| GPU | 1× GPU | 4× GPU (DDP) |
|---|---|---|
| V100 | ~46 hrs | ~12.5 hrs |
| A100 | ~16 hrs | ~4.5 hrs |
| RTX 5090 | ~21 hrs | ~5.7 hrs |

DDP scaling 효율 ~0.92 가정. On-the-fly 합성 시 CPU 병목으로 20~40% 추가 소요 가능.

---

## CLAP Alignment Adapter (VLA 연결용)

SLED encoder의 128d source embedding을 CLAP audio space(512d)에 정렬하는 projection head.
이를 통해 VLA의 language token과 audio source 간 semantic matching이 가능해진다.

### 구조

```python
class CLAPProjectionHead(nn.Module):
    """SLED 128d → CLAP 512d alignment"""
    def __init__(self, sled_dim=128, clap_dim=512):
        self.proj = nn.Sequential(
            nn.Linear(sled_dim, 256),
            nn.GELU(),
            nn.Linear(256, clap_dim),
        )

    def forward(self, source_embeds):
        # source_embeds: [B, 5, 128]
        projected = self.proj(source_embeds)           # [B, 5, 512]
        return F.normalize(projected, dim=-1)          # L2 normalize
```

### 학습

SLED encoder는 frozen. CLAP projection head만 학습.
Loss: SLED가 예측한 class_id에 해당하는 CLAP text embedding과의 cosine similarity 최대화.

```python
# class_text_embeds: [300, 512] — CLAP text encoder로 사전 생성, frozen
# clap_embeds: [B, 5, 512] — projection head 출력
# class_ids: [B, 5] — GT or SLED predicted class

target_embeds = class_text_embeds[class_ids]          # [B, 5, 512]
loss = 1 - F.cosine_similarity(clap_embeds, target_embeds, dim=-1).mean()
```

수천 step (수 분)이면 수렴. SLED가 이미 class 구분 가능한 feature를 학습했으므로 정렬이 빠름.

---

## 실시간 Inference

### 추론 파이프라인

```python
class SLEDInference:
    def __init__(self, checkpoint_path, device="cuda", use_half=True):
        self.model = load_sled(checkpoint_path).eval()
        self.clap_head = load_clap_head(checkpoint_path).eval()
        if use_half:
            self.model.half()
            self.clap_head.half()

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.clap_head.parameters():
            p.requires_grad = False

    def __call__(self, binaural_audio):
        """
        Args:
            binaural_audio: [B, 2, T_samples] raw waveform
        Returns:
            detections: list[list[dict]] — slot별 detection 결과
            source_embeds: [B, 5, 128]
            clap_embeds: [B, 5, 512]
        """
        features = self.preprocess(binaural_audio)    # → [B, 4, 64, T]
        encoder_out = self.model.encoder(features)    # → [B, 128, T]
        detections = self.model.heads(encoder_out)
        source_embeds = self.model.slot_pool(encoder_out, detections)
        clap_embeds = self.clap_head(source_embeds)

        # confidence threshold + NMS
        filtered = self.filter_detections(detections, threshold=0.5)
        return filtered, source_embeds, clap_embeds
```

### 양자화 (CPU 배포 시)

```bash
# ONNX 변환
python export_onnx.py --checkpoint best.pt --output sled.onnx

# ONNX Runtime INT8 양자화
python -m onnxruntime.quantization.quantize \
    --input sled.onnx \
    --output sled_int8.onnx \
    --quant_format QDQ
```

INT8 양자화 시 RTX 3060에서 ~3ms/frame, CPU(i7-12700)에서 ~20ms/frame 달성 가능.

---

## DCASE 2025 참고사항 (NERC-SLIP 1st place)

DCASE 2025 Task 3에서 1등을 달성한 NERC-SLIP 팀의 접근에서 채택한 요소:

| 요소 | 채택 여부 | 비고 |
|---|---|---|
| ResNet-Conformer backbone | ✅ | 우리 구조와 거의 동일 |
| Conformer ×8 | ✗ → ×4 | 실시간 제약으로 절반 사용 |
| SCS augmentation | ✅ | L/R swap + azimuth 반전 |
| SRIR 합성 데이터 | ✅ | Spatial Scaper 라이브러리 |
| Mixup | ✅ | source-level mixup으로 적용 |
| Cartesian unit vector DOA | ✅ | angular wrapping 회피 |
| 3-model ensemble | ✗ | 실시간 제약으로 단일 모델 |
| SCE auxiliary loss | ✅ | 학습 시에만, inference 시 제거 |

---

## 주의사항

1. **HRTF 다양성**: 단일 HRTF로 학습하면 특정 head shape에 overfitting.
   MIT KEMAR, CIPIC, HUTUBS 등 3개 이상 DB를 혼합.

2. **IPD wrapping**: IPD 값이 [-π, π] 범위에서 wrapping 발생 가능.
   cos(IPD), sin(IPD)로 분해하면 연속적인 표현이 되지만 채널이 5개로 늘어남.
   성능/비용 trade-off 고려. --> 기존 4채널(L/R mel-spectogram, IPD, ILD)에서 5채널(L/R mel-spectogram, cos(IPD), sin(IPD), ILD)오로 하자

3. **Loudness GT 정의**: mix 후 binaural signal이 아닌 개별 mono source 기준
   해당 frame 구간의 RMS를 dBFS로 변환. 개별 source 기준이어야 GT가 의미 있음.

4. **On-the-fly 합성**: 디스크 절약 가능하나 CPU 병목 발생.
   초기 실험은 pre-synthesized, 모델 안정 후 on-the-fly로 전환 권장.
   `num_workers=16`, `prefetch_factor=4` 이상 설정.

5. **Slot 순서**: GT에서 5개 slot 배정 순서는 Hungarian matching이 해결.
   .npy 생성 시 onset 순서로 정렬하면 디버깅 편의성 향상.
