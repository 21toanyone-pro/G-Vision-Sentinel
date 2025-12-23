# ESP/Wallhack Detection

**Visual Anti-Cheat System for ESP/Wallhack Detection**

A deep learning-based system that detects ESP (Extra Sensory Perception) hack overlays in game footage using synthetic data generation and object detection models.

---

## Overview

ESP hacks display enemy positions through walls using visual overlays (red boxes, health bars, etc.). Since collecting real hack data is legally/ethically difficult, this project generates synthetic training data by simulating ESP overlays on clean gameplay footage.

**How it works:**
1. Detect `person` objects in clean gameplay using pretrained YOLOv11
2. Draw fake ESP overlays (red/green boxes) on detected persons
3. Save overlay coordinates as YOLO labels
4. Train model to detect these overlay patterns
5. Deploy for real-time cheat detection

---

## Detection Results

### Real-time ESP Detection

![Detection Result](runs/detect/test_detection.jpg)

Multiple ESP overlays detected simultaneously (`ESP 0.48`, `ESP 0.55`) with warning banner displayed.

### Training Data Sample

![Training Batch](runs/esp_detector_20251223_112005/train_batch0.jpg)

Synthetic ESP overlays with mosaic augmentation applied during training.

### Validation Results

![Validation](runs/esp_detector_20251223_112005/val_batch0_pred.jpg)

High confidence detections (`esp_overlay 1.0`, `esp_overlay 0.9`) on validation set.

---

## Model Comparison

Four models were trained and evaluated on the same dataset with focus on **False Positive Rate** and **Speed**.

### Performance Summary

![Model Comparison](docs/model_comparison.png)

### Benchmark Results

| Model | Precision | Recall | F1 Score | False Positives | FP Rate | Inference | FPS |
|-------|-----------|--------|----------|-----------------|---------|-----------|-----|
| **YOLOv8n** | 95.7% | 95.7% | 95.7% | 1 | 0.9% | 8.1ms | **124** |
| **YOLOv10n** | 100% | 69.6% | 82.0% | 0 | **0.0%** | 9.8ms | 102 |
| **YOLOv11n** | 95.5% | 91.3% | 93.4% | 1 | 0.9% | 10.2ms | 98 |
| **RT-DETR-l** | 100% | 100% | **100%** | 0 | **0.0%** | 29.4ms | 34 |

### Confusion Matrix

| Model | TP | FP | TN | FN |
|-------|----|----|----|----|
| YOLOv8n | 22 | 1 | 106 | 1 |
| YOLOv10n | 16 | 0 | 107 | 7 |
| YOLOv11n | 21 | 1 | 106 | 2 |
| RT-DETR-l | 23 | 0 | 107 | 0 |

### Test Video Results (1920x1080 @ 60fps)

| Model | Detection Frames | Detection Rate | Inference | FPS |
|-------|------------------|----------------|-----------|-----|
| **YOLOv8n** | 261/1800 | 14.5% | 8.07ms | **124** |
| YOLOv10n | 162/1800 | 9.0% | 9.78ms | 102 |
| YOLOv11n | 227/1800 | 12.6% | 10.23ms | 98 |
| RT-DETR-l | 273/1800 | **15.2%** | 29.38ms | 34 |

---

## Model Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Real-time Monitoring** | YOLOv8n | Fastest (124 FPS), FP rate < 1% |
| **Zero False Positives** | YOLOv10n or RT-DETR | 0% FP rate guaranteed |
| **Offline Replay Analysis** | RT-DETR-l | Perfect accuracy (100%/100%) |
| **Balanced Performance** | YOLOv11n | Latest architecture, good trade-off |

---

## Project Structure

```
Project_GD/
├── data/
│   ├── raw/                    # Source gameplay videos
│   └── synthetic/              # Generated training data
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       └── labels/val/
├── docs/
│   ├── MODEL_COMPARISON.md     # Detailed model analysis
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── AI_ML_BEST_PRACTICES.md
│   └── README_KR.md            # Korean documentation
├── runs/
│   ├── yolov8n_esp/            # YOLOv8n training results
│   ├── yolov10n_esp/           # YOLOv10n training results
│   ├── esp_detector_*/         # YOLOv11n training results
│   ├── rtdetr_esp/             # RT-DETR training results
│   └── detect/                 # Inference outputs
├── src/
│   ├── generator.py            # Synthetic data generation
│   ├── train.py                # Model training
│   ├── inference.py            # Real-time detection
│   └── utils/
└── requirements.txt
```

---

## Quick Start

### 1. Setup Environment

```bash
conda create -n DG python=3.10 -y
conda activate DG
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python -m src.generator --video "data/raw/gameplay.mp4" --frame-skip 3 --max-frames 800
```

### 3. Train Model

```bash
# YOLOv8n (recommended for speed)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').train(data='runs/data.yaml', epochs=15)"

# YOLOv11n
python -m src.train --epochs 15 --batch 16

# RT-DETR (recommended for accuracy)
python -c "from ultralytics import RTDETR; RTDETR('rtdetr-l.pt').train(data='runs/data.yaml', epochs=15, batch=8)"
```

### 4. Run Inference

```bash
python -m src.inference --video "data/raw/test.mp4" --output "runs/detect/result.mp4"
```

---

## Parameters

### Data Generation

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Input video path | - |
| `--frame-skip` | Extract every Nth frame | 5 |
| `--max-frames` | Maximum frames to process | All |
| `--input-dir` | Directory with videos | data/raw |

### Training

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Training epochs | 10 |
| `--batch` | Batch size | 16 |
| `--imgsz` | Image size | 640 |
| `--resume` | Resume from checkpoint | False |

### Inference

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Input video path | (required) |
| `--model` | Model weights path | Auto-detect latest |
| `--output` | Output video path | Auto-generate |
| `--conf` | Confidence threshold | 0.25 |

---

## Key Techniques

- **Transfer Learning**: Fine-tune pretrained YOLO/DETR models on synthetic ESP data
- **Synthetic Data Generation**: Simulate ESP overlays using OpenCV on clean gameplay
- **Negative Sampling**: 50% clean frames to minimize false positives
- **Data Augmentation**: Random colors, thickness, styles, coordinate jitter
- **Multi-model Comparison**: YOLOv8, YOLOv10, YOLOv11, RT-DETR benchmarking

---

## Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA TITAN RTX (24GB) |
| Dataset | 580 train / 130 val images |
| Epochs | 15 |
| Batch Size | 16 (YOLO) / 8 (DETR) |
| Image Size | 640x640 |

---

## Documentation

- [Model Comparison Analysis](docs/MODEL_COMPARISON.md)
- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)
- [AI/ML Best Practices](docs/AI_ML_BEST_PRACTICES.md)
- [Korean Documentation](docs/README_KR.md)

---

## License

This project is developed for research and educational purposes.

---

