"""Filesystem helpers with path-traversal protection and safe deletion."""
from __future__ import annotations

import logging
import re
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Strip directories and unsafe characters from a user-supplied filename."""
    name = Path(str(name).replace("\\", "/")).name  # drop any directory part
    name = _SAFE_CHARS.sub("_", name).strip("._")
    if not name:
        name = "recording"
    stem, dot, ext = name.rpartition(".")
    if dot and len(name) > max_len:
        name = stem[: max_len - len(ext) - 1] + "." + ext
    return name[:max_len]


def unique_filename(original: str) -> str:
    """Return ``<uuid>_<sanitized name>`` so uploads never collide."""
    return f"{uuid.uuid4().hex}_{sanitize_filename(original)}"


def is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_join(base: Path, *parts: str) -> Path:
    """Join paths and refuse anything that escapes ``base``."""
    candidate = base.joinpath(*parts).resolve()
    if not is_within(base, candidate):
        raise ValueError("Path escapes the storage directory")
    return candidate


def safe_delete(path: str | Path | None, allowed_roots: list[Path]) -> bool:
    """Delete a file only if it lives inside one of ``allowed_roots``."""
    if not path:
        return False
    p = Path(path)
    if not any(is_within(root, p) for root in allowed_roots):
        logger.warning("Refusing to delete file outside data directories")
        return False
    try:
        if p.is_file():
            p.unlink()
            return True
        if p.is_dir():
            shutil.rmtree(p)
            return True
    except OSError as exc:
        logger.error("Failed to delete file: %s", exc.__class__.__name__)
    return False


def free_disk_mb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 * 1024)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
