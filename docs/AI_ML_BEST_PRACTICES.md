# AI/ML 모범 사례 가이드

이 프로젝트에서 쓴 AI/ML 패턴들 정리해놓음.

---

## 1. 코드 구조

### 1.1 Dataclass로 설정 관리

하이퍼파라미터를 딕셔너리로 관리하면 오타나 타입 실수가 생기기 쉬움. Dataclass 쓰면 IDE 자동완성도 되고 타입 체크도 됨.

```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ESPStyleConfig:
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
    ])
    thickness_range: Tuple[int, int] = (1, 3)
```

### 1.2 Enum으로 상태 관리

매직 넘버 쓰면 코드 읽기 힘들어짐. Enum 쓰면 의미도 명확하고 순회도 가능.

```python
from enum import Enum, auto

class ESPStyle(Enum):
    FULL_BOX = auto()
    CORNER_BOX = auto()
    HEALTH_BAR = auto()

# 사용
style = random.choice(list(ESPStyle))
```

### 1.3 Type Hints

타입 힌트 넣으면 mypy로 정적 검사 가능하고, IDE 자동완성이 엄청 좋아짐.

```python
def detect(
    self,
    frame: np.ndarray,
    conf_threshold: Optional[float] = None
) -> Tuple[List[DetectionResult], float]:
    ...
```

---

## 2. 데이터 파이프라인

### 2.1 Augmentation 파라미터화

하드코딩하면 다양성이 떨어짐:

```python
# 나쁜 예 - 항상 같은 스타일
color = (0, 0, 255)  # 항상 빨강
thickness = 2        # 항상 2px

# 좋은 예 - 랜덤 선택
color = random.choice(config.colors)
thickness = random.randint(*config.thickness_range)
```

### 2.2 Negative Sampling

모든 프레임에 ESP 넣으면 모델이 "뭔가 있으면 ESP"로 학습해버림. 50%는 깨끗한 프레임 넣어야 False Positive 줄어듦.

```python
apply_esp = random.random() < 0.5  # 50% 확률

if apply_esp:
    # ESP 그리고 라벨 생성
else:
    # 빈 라벨 파일 생성 (이게 중요)
```

### 2.3 Train/Val 분할

순차적으로 자르면 안됨 (같은 장면이 양쪽에 들어갈 수 있음). 랜덤하게 분할해야 함.

```python
def get_split(train_ratio: float = 0.8) -> str:
    return "train" if random.random() < train_ratio else "val"
```

---

## 3. 학습 패턴

### 3.1 Transfer Learning

처음부터 학습하면 오래 걸림. 사전학습 가중치 쓰면 빠르게 수렴함.

```python
# 처음부터 학습 (오래 걸림)
model = YOLO()
model.train(epochs=1000)

# 전이 학습 (빠름)
model = YOLO("yolo11n.pt")  # COCO 사전학습
model.train(epochs=10)
```

원리:
```
COCO 사전학습 모델
├── Backbone (특징 추출) → 재사용
│   ├── 에지 탐지
│   ├── 텍스처 인식
│   └── 형태 인식
└── Head (클래스 분류) → Fine-tuning
    └── 80개 클래스 → 1개 클래스
```

### 3.2 Early Stopping

과적합 막으려면 validation loss가 안 떨어지면 학습 중단해야 함.

```python
model.train(
    epochs=100,
    patience=5,  # 5 에포크 동안 개선 없으면 중단
)
```

### 3.3 Learning Rate Schedule

학습 후반에는 학습률 낮춰야 fine-tuning이 잘 됨.

```python
model.train(
    lr0=0.001,   # 초기 학습률
    lrf=0.01,    # 최종 학습률 비율 (0.001 × 0.01 = 0.00001)
)
```

---

## 4. 추론 최적화

### 4.1 Model Warmup

첫 추론은 CUDA 커널 컴파일 때문에 느림. 더미 이미지로 워밍업 해놔야 일관된 성능 나옴.

```python
def __init__(self, model_path: str):
    self.model = YOLO(model_path)
    
    # 워밍업
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    self.model(dummy, verbose=False)
```

### 4.2 Performance Monitoring

추론 시간 측정해서 실시간 가능한지 확인해야 함.

```python
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.times = deque(maxlen=window_size)
    
    def add_sample(self, inference_time_ms: float):
        self.times.append(inference_time_ms)
    
    @property
    def avg_time_ms(self) -> float:
        return sum(self.times) / len(self.times)
```

---

## 5. YOLO 라벨 형식

### 5.1 좌표 정규화

픽셀 좌표 쓰면 해상도 바뀔 때 문제됨. 0~1로 정규화해야 함.

```python
def box_to_normalized(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
```

### 5.2 라벨 파일

```
# gameplay_000123.txt
0 0.577734 0.756250 0.614844 0.484722
0 0.123456 0.234567 0.100000 0.200000
```

각 줄이 하나의 객체. class_id가 맨 앞.

---

## 6. 에러 처리

### 6.1 입력 검증 먼저

파일 없으면 바로 에러 던져야 함. 나중에 터지면 디버깅 어려움.

```python
def process_video(video_path: Path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
```

### 6.2 Graceful Degradation

모델 못 찾아도 바로 죽지 말고 None 반환해서 호출자가 처리하게.

```python
def find_latest_model() -> Optional[Path]:
    best_models = list(RUNS_DIR.glob("*/weights/best.pt"))
    if not best_models:
        return None
    return max(best_models, key=lambda p: p.stat().st_mtime)
```

---

## 7. 체크리스트

### 학습 전
- [ ] 데이터셋 Train/Val 분할 확인
- [ ] 라벨 파일 형식 검증
- [ ] GPU 메모리 확인
- [ ] data.yaml 경로 확인

### 추론 전
- [ ] 모델 파일 존재 확인
- [ ] 신뢰도 임계값 설정

### 배포 전
- [ ] 다양한 해상도 테스트
- [ ] False Positive 비율 확인
- [ ] 추론 속도 측정

---

문서 작성일: 2024-12-23
