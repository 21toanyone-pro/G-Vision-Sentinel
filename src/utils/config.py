"""
Configuration Module for G-Vision Sentinel

This module provides centralized configuration management for the
anti-cheat detection system, including paths, hyperparameters, and
ESP overlay generation settings.

Anti-Cheat Logic:
    ESP (Extra Sensory Perception) hacks display hidden enemy positions
    through walls using distinctive visual overlays. This configuration
    defines the visual characteristics we aim to detect.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import yaml


# === Project Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class ESPStyleConfig:
    """
    Configuration for ESP overlay visual styles.
    
    Real ESP hacks vary in appearance - some use simple boxes,
    others use corner markers or health bars. We simulate this
    variety to make our detector robust.
    
    Attributes:
        colors: RGB tuples for overlay colors (typically red/green in real hacks)
        thickness_range: Min/max line thickness in pixels
        jitter_range: Pixel offset range for realistic imperfection
        corner_ratio: Ratio of box edge to draw for corner-style ESP
        apply_probability: Chance to apply ESP to any given frame (0.0-1.0)
    """
    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 255),      # Red (BGR format for OpenCV)
        (0, 255, 0),      # Green
        (0, 165, 255),    # Orange
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
    ])
    thickness_range: Tuple[int, int] = (1, 3)
    jitter_range: Tuple[int, int] = (-5, 5)
    corner_ratio: float = 0.25
    apply_probability: float = 0.5  # 50% frames get ESP overlays


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for the ESP detection model.
    
    We use a nano-sized YOLO model for real-time inference
    capability, critical for live game monitoring.
    
    Attributes:
        model_name: Base YOLO model to fine-tune
        epochs: Number of training epochs
        imgsz: Input image size (square)
        batch_size: Training batch size
        device: Training device (cuda/cpu/auto)
        workers: Number of data loading workers
    """
    model_name: str = "yolo11n.pt"
    epochs: int = 10
    imgsz: int = 640
    batch_size: int = 16
    device: str = ""  # Auto-detect
    workers: int = 4
    patience: int = 5  # Early stopping patience


@dataclass 
class InferenceConfig:
    """
    Inference configuration for real-time detection.
    
    Attributes:
        confidence_threshold: Minimum confidence for detection
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum detections per frame
    """
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100


class Config:
    """
    Master configuration class aggregating all settings.
    
    Usage:
        config = Config()
        print(config.esp.colors)
        print(config.training.epochs)
    """
    
    def __init__(self) -> None:
        self.esp = ESPStyleConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        
        # Class mapping for YOLO
        self.classes = {
            0: "esp_overlay"
        }
        self.num_classes = len(self.classes)
    
    def generate_data_yaml(self, output_path: Path) -> Path:
        """
        Generate YOLO-format data.yaml for training.
        
        Args:
            output_path: Directory to save the yaml file
            
        Returns:
            Path to the generated yaml file
        """
        data_config = {
            "path": str(SYNTHETIC_DIR.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": self.classes
        }
        
        yaml_path = output_path / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return yaml_path
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  ESP: apply_prob={self.esp.apply_probability}, "
            f"colors={len(self.esp.colors)}\n"
            f"  Training: epochs={self.training.epochs}, "
            f"batch={self.training.batch_size}\n"
            f"  Classes: {self.classes}\n"
            f")"
        )


# Default config instance
default_config = Config()

