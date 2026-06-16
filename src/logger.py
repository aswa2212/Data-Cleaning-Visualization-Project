"""
logger.py
---------
Centralised logging factory for the retail sales pipeline.
Every module should import get_logger() and use it instead of print().

Features:
  - Console handler  : INFO level, coloured with timestamps
  - File handler     : DEBUG level, rotating (5 MB × 3 backups)
  - Singleton pattern: same name always returns the same logger instance
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_registry: dict = {}


def get_logger(name: str = "retail_pipeline",
               log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger with console + optional rotating-file handler.

    Parameters
    ----------
    name     : Logger name — use __name__ in calling modules.
    log_file : Absolute path for the log file.  None = console only.
    """
    if name in _registry:
        return _registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)-20s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────────────────────
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,   # 5 MB per file
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _registry[name] = logger
    return logger
