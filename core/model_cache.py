"""Locate locally cached Hugging Face models without any network access.

``snapshot_download(local_files_only=True)`` in recent huggingface_hub versions
rejects snapshots that lack non-essential repo files (README, .gitattributes),
which faster-whisper intentionally never downloads. We therefore check for the
files each model actually needs.
"""
from __future__ import annotations

import os
from pathlib import Path


def hub_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:  # pragma: no cover
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def cached_snapshot(repo_id: str, required: list[str], cache_dir: Path | None = None) -> Path | None:
    """Return the snapshot directory containing all ``required`` files, or None."""
    repo_dir = (cache_dir or hub_cache_dir()) / ("models--" + repo_id.replace("/", "--"))
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    candidates: list[Path] = []
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        candidates.append(snaps / ref.read_text(encoding="utf-8").strip())
    candidates += sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in candidates:
        if snap.is_dir() and all((snap / f).is_file() for f in required):
            return snap
    return None
