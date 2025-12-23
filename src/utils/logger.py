"""
Logging Module for G-Vision Sentinel

Provides consistent, formatted logging across all components
of the anti-cheat detection system.

Features:
    - Colored console output for different log levels
    - File logging for debugging and audit trails
    - Timestamps for tracking processing times
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import PROJECT_ROOT


# ANSI color codes for terminal output
class LogColors:
    """ANSI escape codes for colored terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels.
    
    Makes it easy to visually distinguish between
    INFO, WARNING, ERROR, and DEBUG messages.
    """
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.MAGENTA + LogColors.BOLD,
    }
    
    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and record.levelno in self.LEVEL_COLORS:
            record.levelname = (
                f"{self.LEVEL_COLORS[record.levelno]}"
                f"{record.levelname}"
                f"{LogColors.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str = "gvision",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        use_colors: Whether to use colored console output
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("generator", logging.DEBUG)
        >>> logger.info("Processing frame 100/1000")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = ColoredFormatter(
        fmt="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        use_colors=use_colors
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Capture everything in file
        file_fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "gvision") -> logging.Logger:
    """
    Get an existing logger or create a new one with defaults.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Create default project logger
default_log_path = PROJECT_ROOT / "logs" / f"gvision_{datetime.now():%Y%m%d}.log"
project_logger = setup_logger("gvision", log_file=default_log_path)

