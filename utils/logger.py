"""Simple logger factory without sensitive data logging.

Emits formatted records to stdout and, when possible, to a rotating
log file under ``<repo>/logs/etl.log`` (10 MB × 5 backups). File logging
failures (permission / filesystem) degrade gracefully to stdout-only.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def _ensure_log_dir() -> Path | None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        return _LOG_DIR
    except OSError:
        return None


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level.upper())
    formatter = logging.Formatter(_FMT)

    # Always log to stdout
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    # Rotating file handler (best-effort)
    log_dir = _ensure_log_dir()
    if log_dir is not None:
        try:
            fh = RotatingFileHandler(
                log_dir / "etl.log",
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError:
            # Disk full, permissions, etc. — keep stdout logging and move on.
            pass

    logger.propagate = False
    return logger
