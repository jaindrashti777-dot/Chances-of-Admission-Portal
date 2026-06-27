"""
Structured logging setup for the Admission Prediction project.

Provides rotating file handlers and console output. Every module should use:
    from src.logger import get_logger
    logger = get_logger(__name__)

Zero print() statements — all output through logging.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
    max_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    log_file : str, optional
        Path to the log file. If provided, a rotating file handler is added.
    level : int
        Logging level (default: INFO).
    max_bytes : int
        Maximum log file size before rotation (default: 5 MB).
    backup_count : int
        Number of backup log files to keep (default: 3).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always active
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — only if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_training_logger() -> logging.Logger:
    """Return a logger configured for training operations."""
    return get_logger(
        name="training",
        log_file="logs/training.log",
    )


def get_prediction_logger() -> logging.Logger:
    """Return a logger configured for prediction operations."""
    return get_logger(
        name="prediction",
        log_file="logs/prediction.log",
    )
