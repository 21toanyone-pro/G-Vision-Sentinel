# G-Vision Sentinel 기술 문서

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [AI/ML 핵심 기법](#2-aiml-핵심-기법)
3. [합성 데이터 생성](#3-합성-데이터-생성)
4. [모델 학습](#4-모델-학습)
5. [추론 파이프라인](#5-추론-파이프라인)
6. [하이퍼파라미터](#6-하이퍼파라미터)
7. [결과물](#7-결과물)

---

## 1. 프로젝트 개요

### 1.1 문제 정의
ESP 핵은 벽 너머 적 위치를 빨간 박스로 표시하는 치트임. 근데 실제 핵 영상 수집이 법적/윤리적으로 어려워서, 합성 데이터로 학습시키는 방식을 선택함.

### 1.2 파이프라인 요약
```
[정상 영상] → [사람 탐지] → [ESP 오버레이 합성] → [YOLO 라벨 생성] → [학습] → [탐지]
```

핵심은 **오버레이 자체**를 탐지하는 거지, 사람을 탐지하는게 아님.

---

## 2. AI/ML 핵심 기법

### 2.1 Transfer Learning (전이 학습)

COCO 데이터셋으로 사전학습된 YOLOv11n 가중치를 가져와서 ESP 탐지용으로 fine-tuning함.

| 항목 | 내용 |
|------|------|
| Base Model | yolo11n.pt (COCO 80클래스 학습됨) |
| Target | esp_overlay 1클래스 |
| 방식 | 전체 네트워크 fine-tuning |

왜 이렇게 하냐면, 처음부터 학습하면 수천 에포크 돌려야 하는데, 전이학습 쓰면 10~15 에포크면 충분함.

### 2.2 Synthetic Data Generation (합성 데이터)

실제 핵 데이터 구하기 어려우니까 그냥 직접 만듦:

1. 정상 게임 영상에서 `person` 클래스 탐지 (YOLOv11n pretrained)
2. 탐지된 사람 위치에 빨간/초록 박스 그리기
3. 그 박스 좌표를 YOLO 라벨로 저장

### 2.3 Data Augmentation (데이터 증강)

다양한 ESP 스타일 시뮬레이션:

| 증강 종류 | 구현 방식 | 왜? |
|-----------|-----------|-----|
| Color | 빨강/초록/주황/노랑/마젠타 랜덤 | 다양한 핵 색상 대응 |
| Thickness | 1~3px 랜덤 | 다양한 디자인 대응 |
| Style | Full Box / Corner Box / Health Bar | ESP 유형별 대응 |
| Jitter | ±5px 좌표 변동 | 완벽하지 않은 실제 핵 시뮬레이션 |
| Negative Sampling | 50% 프레임은 ESP 없음 | False Positive 줄이기 |

### 2.4 Negative Sampling 왜 필요?

ESP 있는 프레임만 학습시키면 모델이 "화면에 뭐 있으면 무조건 ESP"로 학습해버림. 그래서 50%는 깨끗한 프레임으로 넣어서 "이건 정상"도 같이 학습시킴.

결과: False Positive 대폭 감소

---

## 3. 합성 데이터 생성

### 3.1 생성 과정

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frame     │───>│   YOLOv11n  │───>│   Draw ESP  │───>│   Save      │
│   Extract   │    │   (Person)  │    │   Overlay   │    │   jpg + txt │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3.2 ESP 스타일

**FULL_BOX (전체 박스)** - 가장 흔한 형태
```
┌────────────────────┐
│                    │
│      Target        │
│                    │
└────────────────────┘
```

**CORNER_BOX (코너 박스)** - 프리미엄 핵에서 자주 보임
```
┌──                ──┐
│                    │
                      
│                    │
└──                ──┘
```

**HEALTH_BAR (체력바)** - 체력 정보까지 보여주는 핵
```
█ ┌────────────────────┐
█ │                    │
█ │      Target        │
  │                    │
  └────────────────────┘
```

### 3.3 YOLO 라벨 형식

```
0 0.577734 0.756250 0.614844 0.484722
│    │        │        │        │
│    │        │        │        └── height (정규화)
│    │        │        └── width (정규화)
│    │        └── y_center (정규화)
│    └── x_center (정규화)
└── class_id (0 = esp_overlay)
```

정규화 공식:
```
x_center = (x1 + x2) / 2 / image_width
y_center = (y1 + y2) / 2 / image_height
width = (x2 - x1) / image_width
height = (y2 - y1) / image_height
```

### 3.4 생성 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| frame_skip | 5 | N번째 프레임만 처리 |
| max_frames | None | 최대 프레임 수 |
| train_split | 0.8 | Train:Val = 80:20 |
| apply_probability | 0.5 | ESP 적용 확률 50% |

---

## 4. 모델 학습

### 4.1 학습 결과 시각화

학습 배치 샘플:

![학습 배치](../runs/esp_detector_20251223_112005/train_batch0.jpg)

ESP 오버레이가 그려진 프레임들이 모자이크 증강 적용되어 학습됨.

### 4.2 기본 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| epochs | 10~15 | 합성 데이터는 빨리 수렴함 |
| batch_size | 16 | GPU 메모리에 따라 조절 |
| imgsz | 640 | YOLO 표준 크기 |
| patience | 5 | Early Stopping |

### 4.3 옵티마이저

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| optimizer | AdamW | 가중치 감쇠 개선된 Adam |
| lr0 | 0.001 | 초기 학습률 |
| lrf | 0.01 | 최종 학습률 (lr0 × lrf) |

### 4.4 학습 시 증강

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| hsv_h | 0.015 | 색조 변화 |
| hsv_s | 0.4 | 채도 변화 |
| hsv_v | 0.3 | 명도 변화 |
| degrees | 5.0 | 회전 각도 |
| translate | 0.1 | 이동 비율 |
| scale | 0.2 | 스케일 변화 |
| flipud | 0.0 | 상하 반전 (게임 화면은 안함) |
| fliplr | 0.3 | 좌우 반전 |
| mosaic | 0.5 | 모자이크 증강 |

### 4.5 data.yaml 구조

```yaml
path: C:/Users/LIM/Desktop/Project_GD/data/synthetic
train: images/train
val: images/val
names:
  0: esp_overlay
```

### 4.6 학습 지표

| 지표 | 의미 | 목표 |
|------|------|------|
| mAP@50 | IoU 0.5에서 평균 정밀도 | > 70% |
| mAP@50-95 | IoU 0.5~0.95 평균 | > 50% |
| Precision | 탐지 정확도 | > 85% |
| Recall | 탐지 재현율 | > 65% |

---

## 5. 추론 파이프라인

### 5.1 탐지 결과 예시

![탐지 결과](../runs/detect/new_sample_800.jpg)

- 상단 빨간 배너: `CHEAT DETECTED - ESP OVERLAY IDENTIFIED`
- 탐지된 ESP: `ILLEGAL 0.54` (신뢰도 54%)

### 5.2 추론 설정

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| confidence_threshold | 0.25 | 최소 신뢰도 |
| iou_threshold | 0.45 | NMS IoU 임계값 |
| max_detections | 100 | 프레임당 최대 탐지 수 |

### 5.3 NMS (Non-Maximum Suppression)

겹치는 박스 중 신뢰도 높은 것만 남김:

```
Before NMS:              After NMS:
┌─────────┐              ┌─────────┐
│ 0.9     │─┐            │ 0.9     │  ← 얘만 남음
│         │ │            │         │
└─────────┘ │            └─────────┘
  ┌─────────┘
  │ 0.7     ← IoU > 0.45면 제거됨
```

### 5.4 실시간 성능

| 목표 FPS | 최대 추론 시간 |
|----------|----------------|
| 60 FPS | < 16.67ms |
| 30 FPS | < 33.33ms |

현재 성능: **~10ms (100+ FPS 가능)**

---

## 6. 하이퍼파라미터

### 6.1 ESP 스타일 설정

```python
@dataclass
class ESPStyleConfig:
    colors: List[Tuple[int, int, int]] = [
        (0, 0, 255),      # Red (BGR)
        (0, 255, 0),      # Green
        (0, 165, 255),    # Orange
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
    ]
    thickness_range: Tuple[int, int] = (1, 3)
    jitter_range: Tuple[int, int] = (-5, 5)
    corner_ratio: float = 0.25
    apply_probability: float = 0.5
```

### 6.2 학습 설정

```python
@dataclass
class TrainingConfig:
    model_name: str = "yolo11n.pt"
    epochs: int = 10
    imgsz: int = 640
    batch_size: int = 16
    device: str = ""  # 자동 감지
    workers: int = 4
    patience: int = 5
```

### 6.3 추론 설정

```python
@dataclass
class InferenceConfig:
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100
```

---

## 7. 결과물

### 7.1 검증 결과

![검증 결과](../runs/esp_detector_20251223_112005/val_batch0_pred.jpg)

ESP 오버레이가 `esp_overlay 1.0`, `esp_overlay 0.9` 등 높은 신뢰도로 탐지됨.

### 7.2 생성 파일

| 경로 | 설명 |
|------|------|
| `runs/esp_detector_*/weights/best.pt` | 최고 성능 모델 |
| `runs/esp_detector_*/weights/last.pt` | 마지막 체크포인트 |
| `runs/detect/*.mp4` | 추론 결과 영상 |

### 7.3 성능 요약

마지막 학습 기준:

| 지표 | 값 |
|------|-----|
| mAP@50 | 81.5% |
| mAP@50-95 | 52.7% |
| Precision | 99.5% |
| Recall | 72.4% |
| 추론 시간 | ~10ms |
| FPS | 100+ |

---

## 부록: 명령어 요약

```bash
# 환경 설정
conda create -n DG python=3.10 -y
conda activate DG
pip install ultralytics opencv-python numpy tqdm pyyaml

# 데이터 생성
python -m src.generator --video "data/raw/gameplay.mp4" --frame-skip 3 --max-frames 800

# 학습
python -m src.train --epochs 15 --batch 16

# 추론
python -m src.inference --video "data/raw/gameplay.mp4" --output "runs/detect/result.mp4"
```

---

문서 작성일: 2024-12-23  
프로젝트: G-Vision Sentinel v1.0.0
