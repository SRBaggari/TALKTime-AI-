"""Logging setup. Logs contain operational events only — never transcript text
or recording content."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(config.LOG_DIR / "talktime.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stream)
    for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "numba", "speechbrain", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
