"""
Training Script for G-Vision Sentinel ESP Detection Model

=============================================================================
ANTI-CHEAT TRAINING STRATEGY
=============================================================================

This module trains a lightweight YOLOv11 model to detect ESP overlay patterns.
The training strategy is optimized for:

1. SPEED: Using YOLOv11n (nano) for real-time inference capability
   - Target: <10ms inference time per frame
   - This enables live game monitoring without performance impact

2. PRECISION: Training specifically on synthetic ESP patterns
   - The model learns to recognize drawn overlays, not actual objects
   - High-contrast visual features (boxes, lines) are easier to detect

3. GENERALIZATION: Varied training data prevents overfitting
   - Multiple ESP styles (full box, corners, health bars)
   - Random colors, thicknesses, and coordinate jitter
   - 50% negative samples to reduce false positives

=============================================================================
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ultralytics import YOLO

from src.utils.config import Config, SYNTHETIC_DIR, RUNS_DIR, default_config
from src.utils.logger import setup_logger


# Initialize module logger
logger = setup_logger("trainer")


class ESPModelTrainer:
    """
    Trainer class for the ESP overlay detection model.
    
    Handles the complete training pipeline including:
    - Data configuration (data.yaml generation)
    - Model initialization and fine-tuning
    - Training execution with progress tracking
    - Model export and saving
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the ESP model trainer.
        
        Args:
            config: Configuration object (uses default if None)
            output_dir: Directory for training outputs (default: runs/)
        """
        self.config = config or default_config
        self.output_dir = Path(output_dir) if output_dir else RUNS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training run name with timestamp
        self.run_name = f"esp_detector_{datetime.now():%Y%m%d_%H%M%S}"
        
        logger.info(f"Trainer initialized. Output: {self.output_dir}")
    
    def _validate_dataset(self) -> bool:
        """
        Validate that synthetic dataset exists and has data.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        train_images = SYNTHETIC_DIR / "images" / "train"
        val_images = SYNTHETIC_DIR / "images" / "val"
        
        if not train_images.exists() or not val_images.exists():
            logger.error("Dataset directories not found!")
            logger.error(f"Expected: {train_images} and {val_images}")
            return False
        
        train_count = len(list(train_images.glob("*.jpg")))
        val_count = len(list(val_images.glob("*.jpg")))
        
        if train_count == 0:
            logger.error("No training images found!")
            logger.error("Run the generator first: python -m src.generator")
            return False
        
        logger.info(f"Dataset validated: {train_count} train, {val_count} val images")
        return True
    
    def _create_data_config(self) -> Path:
        """
        Create YOLO data configuration file.
        
        Generates a data.yaml file that tells YOLO where to find
        the training data and what classes to detect.
        
        Returns:
            Path to generated data.yaml file
        """
        # Create in output directory (persistent)
        yaml_path = self.config.generate_data_yaml(self.output_dir)
        logger.info(f"Data config created: {yaml_path}")
        
        # Log the configuration
        with open(yaml_path) as f:
            logger.debug(f"data.yaml contents:\n{f.read()}")
        
        return yaml_path
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        imgsz: Optional[int] = None,
        resume: bool = False,
        pretrained_weights: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the ESP detection model.
        
        This method:
        1. Validates the dataset
        2. Creates data configuration
        3. Loads the base YOLO model
        4. Runs training with specified hyperparameters
        5. Returns training results and best model path
        
        Args:
            epochs: Number of training epochs (default from config)
            batch_size: Training batch size (default from config)
            imgsz: Input image size (default from config)
            resume: Whether to resume from last checkpoint
            pretrained_weights: Path to custom pretrained weights
            
        Returns:
            Dictionary with training results and paths
            
        Raises:
            RuntimeError: If dataset validation fails
        """
        # Validate dataset
        if not self._validate_dataset():
            raise RuntimeError(
                "Dataset validation failed. Please run the generator first:\n"
                "  python -m src.generator --input-dir data/raw"
            )
        
        # Get hyperparameters (override defaults if provided)
        train_config = self.config.training
        epochs = epochs or train_config.epochs
        batch_size = batch_size or train_config.batch_size
        imgsz = imgsz or train_config.imgsz
        
        logger.info("=" * 60)
        logger.info("G-Vision Sentinel - ESP Detection Model Training")
        logger.info("=" * 60)
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Image Size: {imgsz}")
        logger.info("=" * 60)
        
        # Create data configuration
        data_yaml = self._create_data_config()
        
        # Load model
        model_name = pretrained_weights or train_config.model_name
        logger.info(f"Loading base model: {model_name}")
        model = YOLO(model_name)
        
        # Train the model
        logger.info("Starting training...")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=train_config.device if train_config.device else None,
            workers=train_config.workers,
            patience=train_config.patience,
            project=str(self.output_dir),
            name=self.run_name,
            exist_ok=True,
            verbose=True,
            # Optimization settings
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            # Augmentation (light augmentation for synthetic data)
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            flipud=0.0,  # No vertical flip for game screens
            fliplr=0.3,
            mosaic=0.5,
            # Resume from checkpoint if specified
            resume=resume,
        )
        
        # Get best model path
        best_model_path = self.output_dir / self.run_name / "weights" / "best.pt"
        last_model_path = self.output_dir / self.run_name / "weights" / "last.pt"
        
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Best model saved: {best_model_path}")
        logger.info(f"Last model saved: {last_model_path}")
        logger.info("=" * 60)
        
        return {
            "results": results,
            "best_model": str(best_model_path),
            "last_model": str(last_model_path),
            "run_dir": str(self.output_dir / self.run_name),
            "epochs": epochs,
            "batch_size": batch_size,
        }
    
    def validate(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run validation on the trained model.
        
        Args:
            model_path: Path to model weights (uses best.pt if None)
            
        Returns:
            Validation metrics dictionary
        """
        if model_path is None:
            model_path = self.output_dir / self.run_name / "weights" / "best.pt"
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Running validation with: {model_path}")
        
        model = YOLO(str(model_path))
        data_yaml = self._create_data_config()
        
        results = model.val(data=str(data_yaml))
        
        logger.info(f"Validation mAP50: {results.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {results.box.map:.4f}")
        
        return {
            "map50": results.box.map50,
            "map50_95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }
    
    def export(
        self,
        model_path: Optional[str] = None,
        format: str = "onnx"
    ) -> str:
        """
        Export trained model to different formats.
        
        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, etc.)
            
        Returns:
            Path to exported model
        """
        if model_path is None:
            model_path = self.output_dir / self.run_name / "weights" / "best.pt"
        
        model = YOLO(str(model_path))
        export_path = model.export(format=format)
        
        logger.info(f"Model exported to: {export_path}")
        return export_path


def main():
    """
    Main entry point for model training.
    
    Usage:
        python -m src.train
        
    Or with custom parameters:
        python -m src.train --epochs 20 --batch 32
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="G-Vision Sentinel - Train ESP Detection Model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model for validation/export"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "torchscript", "tflite", "engine"],
        help="Export model to specified format"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ESPModelTrainer()
    
    if args.validate_only:
        # Run validation only
        trainer.validate(model_path=args.model_path)
    elif args.export:
        # Export model
        trainer.export(model_path=args.model_path, format=args.export)
    else:
        # Run training
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            resume=args.resume,
            pretrained_weights=args.weights
        )
        
        # Run validation on best model
        trainer.validate(results["best_model"])
        
        logger.info("\nTraining pipeline complete!")
        logger.info(f"Best model: {results['best_model']}")
        logger.info("\nNext step: Run inference")
        logger.info("  python -m src.inference --video path/to/video.mp4")


if __name__ == "__main__":
    main()

