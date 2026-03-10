import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/utils/logging.py
"""
Logging utilities for SentinelML.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Setup logging for SentinelML.

    Parameters
    ----------
    level : int or str
        Logging level.
    log_file : str, optional
        File to log to (console only if None).
    format_string : str, optional
        Custom format string.
    propagate : bool
        Whether to propagate to root logger.

    Returns
    -------
    logger : logging.Logger
        Configured logger.
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - " "%(filename)s:%(lineno)d - %(message)s"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Get logger
    logger = logging.getLogger("sentinelml")
    logger.setLevel(level)
    logger.propagate = propagate

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns root SentinelML logger.

    Returns
    -------
    logger : logging.Logger
    """
    if name is None:
        return logging.getLogger("sentinelml")
    return logging.getLogger(f"sentinelml.{name}")


class ProgressLogger:
    """
    Context manager for logging progress of long operations.

    Usage:
        with ProgressLogger("Training", total=100) as progress:
            for i in range(100):
                # ... do work ...
                progress.update(1)
    """

    def __init__(
        self,
        description: str,
        total: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ):
        self.description = description
        self.total = total
        self.logger = logger or get_logger()
        self.log_interval = log_interval

        self.current = 0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.description} in {elapsed:.2f}s")
        else:
            self.logger.error(f"Failed: {self.description} after {elapsed:.2f}s - {exc_val}")
        return False

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n

        if self.current % self.log_interval == 0:
            if self.total:
                pct = 100 * self.current / self.total
                self.logger.info(f"{self.description}: {self.current}/{self.total} ({pct:.1f}%)")
            else:
                self.logger.info(f"{self.description}: {self.current} completed")


def log_system_info(logger: Optional[logging.Logger] = None):
    """Log system information for debugging."""
    import platform

    import psutil

    logger = logger or get_logger()

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU: {psutil.cpu_count()} cores")
    logger.info(f"  Memory: {psutil.virtual_memory().total / 1e9:.2f} GB")

    # Optional dependencies
    optional_deps = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("transformers", "Transformers"),
        ("sklearn", "scikit-learn"),
    ]

    logger.info("Optional Dependencies:")
    for module, name in optional_deps:
        try:
            __import__(module)
            logger.info(f"  {name}: installed")
        except ImportError:
            logger.info(f"  {name}: not installed")
