"""
Synthetic ESP Overlay Data Generator for G-Vision Sentinel

=============================================================================
ANTI-CHEAT LOGIC OVERVIEW
=============================================================================

This module generates synthetic training data by simulating ESP (Wallhack) 
overlays on legitimate gameplay footage. Here's why this approach works:

1. THE PROBLEM:
   - ESP hacks display enemy positions through walls using distinctive overlays
   - Real hack data is difficult to collect (ethical/legal concerns)
   - We need thousands of examples to train a robust detector

2. THE SOLUTION:
   - Use YOLOv11 to detect legitimate 'person' objects in clean gameplay
   - Draw fake ESP overlays (boxes, corners, health bars) on detected persons
   - Generate YOLO-format labels for these SYNTHETIC overlays
   - Train a model to detect these visual patterns, NOT the persons

3. AUGMENTATION STRATEGY:
   - 50% clean frames (negative samples) to reduce false positives
   - Visual variation: random colors, thickness, styles
   - Coordinate jitter for realistic imperfection
   - Multiple ESP styles: full box, corner markers

4. OUTPUT:
   - Modified frames saved as images
   - Corresponding .txt label files in YOLO format (normalized xywh)

=============================================================================
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from tqdm import tqdm
from ultralytics import YOLO

from src.utils.config import Config, SYNTHETIC_DIR, RAW_DIR, default_config
from src.utils.logger import setup_logger


# Initialize module logger
logger = setup_logger("generator")


class ESPStyle(Enum):
    """
    Enumeration of ESP overlay drawing styles.
    
    Real ESP hacks come in various visual styles:
    - FULL_BOX: Complete bounding rectangle
    - CORNER_BOX: Only corner markers (common in premium hacks)
    - HEALTH_BAR: Box with health indicator
    """
    FULL_BOX = auto()
    CORNER_BOX = auto()
    HEALTH_BAR = auto()


@dataclass
class ESPOverlay:
    """
    Represents a single ESP overlay annotation.
    
    Attributes:
        x_center: Normalized x center (0-1)
        y_center: Normalized y center (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        class_id: Class index (always 0 for esp_overlay)
    """
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int = 0
    
    def to_yolo_format(self) -> str:
        """Convert to YOLO label format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class ESPDrawer:
    """
    Draws synthetic ESP overlays with visual variation.
    
    This class simulates the visual appearance of real ESP hacks
    by drawing overlays with randomized properties for data augmentation.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the ESP drawer with configuration.
        
        Args:
            config: Configuration object with ESP style settings
        """
        self.config = config
        self.esp_config = config.esp
        
    def _apply_jitter(self, value: int, img_dim: int) -> int:
        """
        Apply random jitter to a coordinate for realistic imperfection.
        
        Real ESP overlays often have slight tracking imperfections.
        
        Args:
            value: Original coordinate value
            img_dim: Image dimension for boundary clamping
            
        Returns:
            Jittered coordinate value
        """
        jitter = random.randint(*self.esp_config.jitter_range)
        return max(0, min(img_dim - 1, value + jitter))
    
    def _get_random_style(self) -> Tuple[Tuple[int, int, int], int, ESPStyle]:
        """
        Generate random visual properties for an ESP overlay.
        
        Returns:
            Tuple of (color_bgr, thickness, style)
        """
        color = random.choice(self.esp_config.colors)
        thickness = random.randint(*self.esp_config.thickness_range)
        style = random.choice(list(ESPStyle))
        return color, thickness, style
    
    def draw_full_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int
    ) -> np.ndarray:
        """
        Draw a complete bounding box ESP overlay.
        
        Args:
            image: Input image (modified in-place)
            box: Bounding box (x1, y1, x2, y2)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Modified image
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image
    
    def draw_corner_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int
    ) -> np.ndarray:
        """
        Draw corner-only ESP overlay (premium hack style).
        
        Only draws the corners of the bounding box, leaving the
        middle sections empty. This is common in sophisticated hacks.
        
        Args:
            image: Input image (modified in-place)
            box: Bounding box (x1, y1, x2, y2)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Modified image
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        corner_len_x = int(w * self.esp_config.corner_ratio)
        corner_len_y = int(h * self.esp_config.corner_ratio)
        
        # Top-left corner
        cv2.line(image, (x1, y1), (x1 + corner_len_x, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + corner_len_y), color, thickness)
        
        # Top-right corner
        cv2.line(image, (x2, y1), (x2 - corner_len_x, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + corner_len_y), color, thickness)
        
        # Bottom-left corner
        cv2.line(image, (x1, y2), (x1 + corner_len_x, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2 - corner_len_y), color, thickness)
        
        # Bottom-right corner
        cv2.line(image, (x2, y2), (x2 - corner_len_x, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2 - corner_len_y), color, thickness)
        
        return image
    
    def draw_health_bar(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int
    ) -> np.ndarray:
        """
        Draw ESP overlay with simulated health bar.
        
        Draws a bounding box plus a vertical health bar on the left
        side, mimicking health-displaying ESP hacks.
        
        Args:
            image: Input image (modified in-place)
            box: Bounding box (x1, y1, x2, y2)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Modified image
        """
        x1, y1, x2, y2 = box
        
        # Draw main box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw health bar (left side, random health level)
        health_pct = random.uniform(0.2, 1.0)
        bar_width = max(3, (x2 - x1) // 20)
        bar_x = x1 - bar_width - 3
        bar_height = int((y2 - y1) * health_pct)
        bar_y_top = y2 - bar_height
        
        # Health bar background (dark)
        cv2.rectangle(image, (bar_x, y1), (bar_x + bar_width, y2), (50, 50, 50), -1)
        
        # Health bar fill (green to red gradient based on health)
        if health_pct > 0.5:
            bar_color = (0, 255, 0)  # Green
        elif health_pct > 0.25:
            bar_color = (0, 255, 255)  # Yellow
        else:
            bar_color = (0, 0, 255)  # Red
            
        cv2.rectangle(image, (bar_x, bar_y_top), (bar_x + bar_width, y2), bar_color, -1)
        
        return image
    
    def draw_overlay(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Draw an ESP overlay with random style variations.
        
        Args:
            image: Input image (modified in-place)
            box: Original detection box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (modified image, actual drawn box coordinates)
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        
        # Apply jitter to coordinates
        x1 = self._apply_jitter(x1, w)
        y1 = self._apply_jitter(y1, h)
        x2 = self._apply_jitter(x2, w)
        y2 = self._apply_jitter(y2, h)
        
        # Ensure valid box
        if x1 >= x2 or y1 >= y2:
            return image, box
        
        jittered_box = (x1, y1, x2, y2)
        color, thickness, style = self._get_random_style()
        
        # Draw based on style
        if style == ESPStyle.FULL_BOX:
            self.draw_full_box(image, jittered_box, color, thickness)
        elif style == ESPStyle.CORNER_BOX:
            self.draw_corner_box(image, jittered_box, color, thickness)
        elif style == ESPStyle.HEALTH_BAR:
            self.draw_health_bar(image, jittered_box, color, thickness)
        
        return image, jittered_box


class SyntheticDataGenerator:
    """
    Main synthetic data generation engine.
    
    Processes video files to generate synthetic ESP overlay training data.
    Uses YOLOv11 for person detection and applies fake ESP overlays.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        detector_model: str = "yolo11n.pt",
        train_split: float = 0.8
    ) -> None:
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Configuration object (uses default if None)
            detector_model: YOLO model for person detection
            train_split: Fraction of data for training (rest goes to val)
        """
        self.config = config or default_config
        self.train_split = train_split
        self.esp_drawer = ESPDrawer(self.config)
        
        # Load person detector
        logger.info(f"Loading detector model: {detector_model}")
        self.detector = YOLO(detector_model)
        
        # Create output directories
        self._setup_directories()
        
        # Statistics
        self.stats: Dict[str, int] = {
            "total_frames": 0,
            "esp_frames": 0,
            "clean_frames": 0,
            "total_overlays": 0,
        }
    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        for split in ["train", "val"]:
            (SYNTHETIC_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (SYNTHETIC_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {SYNTHETIC_DIR}")
    
    def _get_split(self) -> str:
        """Randomly determine train/val split for current frame."""
        return "train" if random.random() < self.train_split else "val"
    
    def _box_to_normalized(
        self,
        box: Tuple[int, int, int, int],
        img_w: int,
        img_h: int
    ) -> ESPOverlay:
        """
        Convert pixel coordinates to normalized YOLO format.
        
        Args:
            box: Bounding box (x1, y1, x2, y2) in pixels
            img_w: Image width
            img_h: Image height
            
        Returns:
            ESPOverlay with normalized coordinates
        """
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        return ESPOverlay(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=0  # esp_overlay
        )
    
    def _save_frame(
        self,
        frame: np.ndarray,
        labels: List[ESPOverlay],
        frame_id: str,
        split: str
    ) -> None:
        """
        Save frame and corresponding label file.
        
        Args:
            frame: Image frame to save
            labels: List of ESP overlay annotations
            frame_id: Unique frame identifier
            split: 'train' or 'val'
        """
        # Save image
        img_path = SYNTHETIC_DIR / "images" / split / f"{frame_id}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save labels (empty file for clean frames)
        label_path = SYNTHETIC_DIR / "labels" / split / f"{frame_id}.txt"
        with open(label_path, "w") as f:
            for label in labels:
                f.write(label.to_yolo_format() + "\n")
    
    def process_video(
        self,
        video_path: Path,
        frame_skip: int = 5,
        max_frames: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process a video file to generate synthetic training data.
        
        This is the main entry point for data generation. It:
        1. Iterates through video frames (with optional skipping)
        2. Detects persons using YOLO
        3. Applies ESP overlays to 50% of frames
        4. Saves images and YOLO-format labels
        
        Args:
            video_path: Path to input video file
            frame_skip: Process every Nth frame (reduces dataset size)
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            Dictionary with processing statistics
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        
        # Validate input
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Processing: {video_path.name}")
        logger.info(f"Video info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Calculate frames to process
        frames_to_process = total_frames // frame_skip
        if max_frames:
            frames_to_process = min(frames_to_process, max_frames)
        
        video_name = video_path.stem
        frame_count = 0
        processed_count = 0
        
        # Progress bar
        pbar = tqdm(
            total=frames_to_process,
            desc=f"Generating ESP data",
            unit="frames",
            ncols=100
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            if max_frames and processed_count >= max_frames:
                break
            
            frame_id = f"{video_name}_{frame_count:06d}"
            split = self._get_split()
            labels: List[ESPOverlay] = []
            
            # Decide whether to apply ESP (50% probability)
            apply_esp = random.random() < self.config.esp.apply_probability
            
            if apply_esp:
                # Detect persons in frame
                results = self.detector(frame, classes=[0], verbose=False)  # class 0 = person
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw ESP overlay
                            frame, drawn_box = self.esp_drawer.draw_overlay(
                                frame, (x1, y1, x2, y2)
                            )
                            
                            # Create label for the DRAWN overlay (not the person!)
                            label = self._box_to_normalized(drawn_box, width, height)
                            labels.append(label)
                            self.stats["total_overlays"] += 1
                
                self.stats["esp_frames"] += 1
            else:
                self.stats["clean_frames"] += 1
            
            # Save frame and labels
            self._save_frame(frame, labels, frame_id, split)
            
            self.stats["total_frames"] += 1
            processed_count += 1
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        logger.info(f"Generation complete! Stats: {self.stats}")
        return self.stats
    
    def process_directory(
        self,
        input_dir: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, int]:
        """
        Process all videos in a directory.
        
        Args:
            input_dir: Directory containing videos (defaults to data/raw)
            **kwargs: Additional arguments passed to process_video
            
        Returns:
            Aggregated statistics
        """
        input_dir = Path(input_dir) if input_dir else RAW_DIR
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        videos = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]
        
        if not videos:
            logger.warning(f"No videos found in {input_dir}")
            return self.stats
        
        logger.info(f"Found {len(videos)} videos to process")
        
        for video_path in videos:
            try:
                self.process_video(video_path, **kwargs)
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {e}")
                continue
        
        return self.stats


def main():
    """
    Main entry point for synthetic data generation.
    
    Usage:
        python -m src.generator
        
    Or with custom video:
        from src.generator import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        gen.process_video("path/to/video.mp4")
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="G-Vision Sentinel - Synthetic ESP Data Generator"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to specific video file (optional)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing videos (default: data/raw)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process per video"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    if args.video:
        # Process single video
        generator.process_video(
            Path(args.video),
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
    else:
        # Process all videos in directory
        generator.process_directory(
            input_dir=Path(args.input_dir) if args.input_dir else None,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
    
    logger.info("Synthetic data generation complete!")
    logger.info(f"Output saved to: {SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()

