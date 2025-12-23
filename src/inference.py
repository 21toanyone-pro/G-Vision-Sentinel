"""
Real-Time Inference Script for G-Vision Sentinel

=============================================================================
ANTI-CHEAT DETECTION IN ACTION
=============================================================================

This module demonstrates the practical application of the trained ESP
detection model. It processes video feeds to identify and flag potential
ESP/Wallhack overlays in real-time.

KEY FEATURES:
1. Real-time processing with latency logging
2. Visual flagging of detected illegal overlays
3. Frame-by-frame confidence scoring
4. Demo video generation for verification

DETECTION WORKFLOW:
1. Load trained ESP detection model (best.pt)
2. Process video frames through the model
3. For each detection, draw warning overlay
4. Log inference time to verify real-time capability
5. Output annotated video for review

=============================================================================
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import deque

from tqdm import tqdm
from ultralytics import YOLO

from src.utils.config import Config, RUNS_DIR, default_config
from src.utils.logger import setup_logger


# Initialize module logger
logger = setup_logger("inference")


# Detection visualization settings
DETECTION_SETTINGS = {
    "box_color": (0, 0, 255),      # Red (BGR)
    "text_color": (255, 255, 255),  # White
    "warning_color": (0, 0, 200),   # Dark red
    "box_thickness": 3,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.7,
    "font_thickness": 2,
}


@dataclass
class DetectionResult:
    """
    Represents a single ESP overlay detection.
    
    Attributes:
        bbox: Bounding box (x1, y1, x2, y2)
        confidence: Detection confidence (0-1)
        class_name: Detected class name
        inference_time_ms: Time taken for inference
    """
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    inference_time_ms: float


class PerformanceMonitor:
    """
    Monitors inference performance for real-time capability assessment.
    
    Tracks inference times to ensure the model meets real-time requirements
    (typically <33ms for 30 FPS, <16ms for 60 FPS).
    """
    
    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize the performance monitor.
        
        Args:
            window_size: Number of samples for moving average
        """
        self.times: deque = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.total_detections: int = 0
    
    def add_sample(self, inference_time_ms: float, num_detections: int) -> None:
        """Record an inference sample."""
        self.times.append(inference_time_ms)
        self.total_frames += 1
        self.total_detections += num_detections
    
    @property
    def avg_time_ms(self) -> float:
        """Average inference time in milliseconds."""
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def max_time_ms(self) -> float:
        """Maximum inference time in milliseconds."""
        return max(self.times) if self.times else 0.0
    
    @property
    def min_time_ms(self) -> float:
        """Minimum inference time in milliseconds."""
        return min(self.times) if self.times else 0.0
    
    @property
    def estimated_fps(self) -> float:
        """Estimated FPS based on average inference time."""
        return 1000 / self.avg_time_ms if self.avg_time_ms > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary dictionary."""
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_time_ms": round(self.avg_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "estimated_fps": round(self.estimated_fps, 1),
        }


class ESPDetector:
    """
    Real-time ESP overlay detector.
    
    Uses a trained YOLOv11 model to detect illegal ESP overlays
    in game footage. Designed for minimal latency and high accuracy.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
        device: str = ""
    ) -> None:
        """
        Initialize the ESP detector.
        
        Args:
            model_path: Path to trained model weights (.pt file)
            config: Configuration object
            device: Inference device (cuda/cpu/auto)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.config = config or default_config
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading ESP detection model: {model_path}")
        self.model = YOLO(str(model_path))
        
        # Warm up the model
        logger.info("Warming up model...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)
        
        logger.info("ESP Detector initialized and ready!")
    
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Tuple[List[DetectionResult], float]:
        """
        Detect ESP overlays in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold (default from config)
            iou_threshold: IoU threshold for NMS (default from config)
            
        Returns:
            Tuple of (list of detections, inference time in ms)
        """
        conf = conf_threshold or self.config.inference.confidence_threshold
        iou = iou_threshold or self.config.inference.iou_threshold
        
        # Run inference with timing
        start_time = time.perf_counter()
        results = self.model(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            max_det=self.config.inference.max_detections
        )
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Parse results
        detections: List[DetectionResult] = []
        
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.config.classes.get(cls, f"class_{cls}")
                    
                    detections.append(DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=class_name,
                        inference_time_ms=inference_time
                    ))
        
        return detections, inference_time
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        inference_time_ms: float
    ) -> np.ndarray:
        """
        Draw detection results on frame with warning labels.
        
        Args:
            frame: Input frame (modified in-place)
            detections: List of detections to draw
            inference_time_ms: Inference time for display
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        settings = DETECTION_SETTINGS
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1), (x2, y2),
                settings["box_color"],
                settings["box_thickness"]
            )
            
            # Draw warning label
            label = f"ILLEGAL_Overlay_Detected {det.confidence:.2f}"
            
            # Calculate label background size
            (label_w, label_h), baseline = cv2.getTextSize(
                label,
                settings["font"],
                settings["font_scale"],
                settings["font_thickness"]
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                settings["warning_color"],
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                settings["font"],
                settings["font_scale"],
                settings["text_color"],
                settings["font_thickness"],
                cv2.LINE_AA
            )
        
        # Draw performance overlay
        perf_text = f"Inference: {inference_time_ms:.1f}ms | FPS: {1000/inference_time_ms:.1f}"
        cv2.putText(
            annotated,
            perf_text,
            (10, 30),
            settings["font"],
            0.6,
            (0, 255, 0),  # Green
            2,
            cv2.LINE_AA
        )
        
        # Draw detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(
            annotated,
            det_text,
            (10, 60),
            settings["font"],
            0.6,
            (0, 255, 255) if detections else (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw warning banner if detections found
        if detections:
            h, w = annotated.shape[:2]
            banner_height = 40
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            warning_text = "!! CHEAT DETECTED - ESP OVERLAY IDENTIFIED !!"
            cv2.putText(
                annotated,
                warning_text,
                (w // 2 - 250, 28),
                settings["font"],
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
        
        return annotated


def process_video(
    detector: ESPDetector,
    input_path: str,
    output_path: Optional[str] = None,
    show_preview: bool = False,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a video file for ESP detection.
    
    Args:
        detector: Initialized ESPDetector instance
        input_path: Path to input video
        output_path: Path for output video (optional)
        show_preview: Whether to show live preview window
        max_frames: Maximum frames to process
        
    Returns:
        Dictionary with processing results
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing: {input_path.name}")
    logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Setup output video writer
    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        logger.info(f"Output: {output_path}")
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Process frames
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
    
    pbar = tqdm(
        total=frames_to_process,
        desc="Detecting ESP overlays",
        unit="frames",
        ncols=100
    )
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Run detection
            detections, inference_time = detector.detect(frame)
            
            # Update performance monitor
            monitor.add_sample(inference_time, len(detections))
            
            # Draw annotations
            annotated = detector.draw_detections(frame, detections, inference_time)
            
            # Write to output
            if writer:
                writer.write(annotated)
            
            # Show preview
            if show_preview:
                cv2.imshow("G-Vision Sentinel - ESP Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Preview closed by user")
                    break
            
            frame_count += 1
            pbar.update(1)
            
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
    
    # Log performance summary
    summary = monitor.get_summary()
    
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE - Performance Summary")
    logger.info("=" * 60)
    logger.info(f"Total Frames Processed: {summary['total_frames']}")
    logger.info(f"Total ESP Detections: {summary['total_detections']}")
    logger.info("-" * 60)
    logger.info(f"Average Inference Time: {summary['avg_time_ms']:.2f} ms")
    logger.info(f"Min Inference Time: {summary['min_time_ms']:.2f} ms")
    logger.info(f"Max Inference Time: {summary['max_time_ms']:.2f} ms")
    logger.info(f"Estimated FPS: {summary['estimated_fps']:.1f}")
    logger.info("-" * 60)
    
    # Real-time capability assessment
    if summary['avg_time_ms'] < 16.67:
        logger.info("✓ REAL-TIME CAPABLE (60+ FPS)")
    elif summary['avg_time_ms'] < 33.33:
        logger.info("✓ REAL-TIME CAPABLE (30+ FPS)")
    else:
        logger.warning("✗ NOT REAL-TIME (Consider GPU acceleration)")
    
    logger.info("=" * 60)
    
    if output_path:
        logger.info(f"Output video saved: {output_path}")
    
    return {
        "input": str(input_path),
        "output": str(output_path) if output_path else None,
        **summary
    }


def find_latest_model() -> Optional[Path]:
    """
    Find the most recent trained model.
    
    Returns:
        Path to best.pt or None if not found
    """
    runs_dir = RUNS_DIR
    if not runs_dir.exists():
        return None
    
    # Find all best.pt files
    best_models = list(runs_dir.glob("*/weights/best.pt"))
    
    if not best_models:
        # Also check detect subdirectory (Ultralytics default)
        best_models = list(runs_dir.glob("detect/*/weights/best.pt"))
    
    if not best_models:
        return None
    
    # Return most recent
    return max(best_models, key=lambda p: p.stat().st_mtime)


def main():
    """
    Main entry point for inference.
    
    Usage:
        python -m src.inference --video path/to/video.mp4
        
    With custom model:
        python -m src.inference --video video.mp4 --model runs/best.pt
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="G-Vision Sentinel - ESP Detection Inference"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: auto-detect latest)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output video (default: runs/detect/output_<timestamp>.mp4)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (0-1)"
    )
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_model()
        if not model_path:
            logger.error("No trained model found!")
            logger.error("Please train a model first: python -m src.train")
            logger.error("Or specify a model path: --model path/to/best.pt")
            return
        logger.info(f"Using latest model: {model_path}")
    
    # Set default output path
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"runs/detect/output_{timestamp}.mp4"
    
    # Initialize detector
    config = default_config
    if args.conf:
        config.inference.confidence_threshold = args.conf
    
    detector = ESPDetector(str(model_path), config)
    
    # Run inference
    results = process_video(
        detector,
        args.video,
        output_path=args.output,
        show_preview=args.show,
        max_frames=args.max_frames
    )
    
    logger.info("\nInference pipeline complete!")
    logger.info(f"Results: {results['total_detections']} ESP overlays detected")


if __name__ == "__main__":
    main()

