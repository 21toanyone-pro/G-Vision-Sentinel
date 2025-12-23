# G-Vision Sentinel

**ESP/월핵 탐지용 시각 안티치트 시스템**

합성 데이터 생성 + YOLOv11 조합으로 게임 영상에서 ESP 핵 오버레이를 탐지하는 시스템임.

---

## 어떤 프로젝트?

ESP 핵(월핵)은 벽 너머 적 위치를 빨간 박스 같은 오버레이로 보여주는 치트인데, 이거 탐지하려면 학습 데이터가 필요함. 근데 실제 핵 영상 구하기가 법적으로도 그렇고 쉽지 않아서, 그냥 직접 합성 데이터를 만들어서 학습시킴.

**핵심 아이디어:**
1. 정상 게임 영상에서 사람 탐지 (YOLOv11 pretrained)
2. 탐지된 사람 위에 가짜 ESP 오버레이 그리기 (빨강/초록 박스 등)
3. 그 오버레이 좌표를 YOLO 라벨로 저장
4. 이걸로 모델 학습시키면 ESP 오버레이만 찾아냄

---

## 탐지 결과 예시

### 실제 추론 결과
ESP 오버레이가 있는 화면에서 탐지가 제대로 동작하는 모습:

![탐지 결과](../runs/detect/new_sample_800.jpg)

상단에 빨간 배너로 `CHEAT DETECTED - ESP OVERLAY IDENTIFIED` 표시되고, 오른쪽에 `ILLEGAL 0.54`로 탐지된 ESP 박스가 보임.

### 학습 데이터 샘플
학습에 사용된 합성 데이터 예시 (모자이크 증강 적용됨):

![학습 배치](../runs/esp_detector_20251223_112005/train_batch0.jpg)

### 검증 결과
검증 데이터에서 ESP 오버레이 탐지 결과:

![검증 결과](../runs/esp_detector_20251223_112005/val_batch0_pred.jpg)

`esp_overlay 1.0`, `esp_overlay 0.9` 같이 높은 신뢰도로 탐지되는거 확인 가능.

---

## 프로젝트 구조

```
Project_GD/
├── data/
│   ├── raw/                    # 원본 게임 영상 넣는 곳
│   └── synthetic/              # 생성된 학습 데이터
│       ├── images/train/       # 학습 이미지
│       ├── images/val/         # 검증 이미지
│       ├── labels/train/       # 학습 라벨
│       └── labels/val/         # 검증 라벨
├── runs/
│   ├── esp_detector_*/         # 학습 결과물
│   └── detect/                 # 추론 결과물
├── src/
│   ├── generator.py            # 합성 데이터 생성
│   ├── train.py                # 모델 학습
│   ├── inference.py            # 추론
│   └── utils/                  # 설정/로깅
└── docs/                       # 문서
```

---

## 사용법

### 1. 환경 설정

```bash
conda create -n DG python=3.10 -y
conda activate DG
pip install -r requirements.txt
```

### 2. 데이터 생성

영상을 `data/raw/`에 넣고:

```bash
python -m src.generator --video "data/raw/gameplay.mp4" --frame-skip 3 --max-frames 800
```

### 3. 학습

```bash
python -m src.train --epochs 15 --batch 16
```

### 4. 추론

```bash
python -m src.inference --video "data/raw/gameplay.mp4" --output "runs/detect/result.mp4"
```

---

## 파라미터 정리

### 데이터 생성

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video` | 영상 경로 | - |
| `--frame-skip` | N프레임마다 추출 | 5 |
| `--max-frames` | 최대 프레임 수 | 전체 |

### 학습

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--epochs` | 에포크 수 | 10 |
| `--batch` | 배치 크기 | 16 |
| `--imgsz` | 이미지 크기 | 640 |

### 추론

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video` | 입력 영상 | (필수) |
| `--model` | 모델 경로 | 최신거 자동 탐지 |
| `--conf` | 신뢰도 임계값 | 0.25 |

---

## 성능

마지막 학습 결과:

| 지표 | 값 |
|------|-----|
| mAP@50 | 81.5% |
| Precision | 99.5% |
| Recall | 72.4% |
| 추론 시간 | ~10ms |
| 실시간 FPS | 100+ |

---

자세한 기술 문서는 `TECHNICAL_DOCUMENTATION.md` 참고.
