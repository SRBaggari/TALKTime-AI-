"""Local file storage for uploads, processed audio and reports."""
from __future__ import annotations

import logging
from pathlib import Path

import config
from core.audio_validator import ValidationResult, validate_upload
from database import repository
from utils.file_utils import dir_size_mb, free_disk_mb, safe_delete, safe_join, unique_filename

logger = logging.getLogger(__name__)


class StorageError(Exception):
    pass


def allowed_roots() -> list[Path]:
    return [config.UPLOAD_DIR, config.PROCESSED_DIR, config.REPORTS_DIR, config.TEMP_DIR]


def save_upload(filename: str, data: bytes, max_mb: int) -> tuple[Path, ValidationResult]:
    """Validate and persist an uploaded file under a unique, sanitised name."""
    result = validate_upload(filename, len(data), data[:64], max_mb=max_mb)
    if not result.ok:
        return Path(), result
    needed_mb = len(data) / 1048576 * 3 + config.MIN_FREE_DISK_MB  # original + processed wav headroom
    if free_disk_mb(config.UPLOAD_DIR) < needed_mb:
        raise StorageError("Insufficient disk space to store and process this recording.")
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = safe_join(config.UPLOAD_DIR, unique_filename(filename))
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)  # atomic on same volume
    logger.info("Stored upload (%d bytes)", len(data))
    return target, result


def processed_path_for(recording_id: str) -> Path:
    return safe_join(config.PROCESSED_DIR, f"{recording_id}.wav")


def delete_recording_files(paths: list[str]) -> int:
    return sum(1 for p in paths if safe_delete(p, allowed_roots()))


def delete_recording(recording_id: str) -> int:
    """Remove a recording's DB rows *and* its audio/report files."""
    return delete_recording_files(repository.delete_recording(recording_id))


def delete_all_data() -> int:
    n = delete_recording_files(repository.delete_all_recordings())
    for d in (config.TEMP_DIR,):
        for f in d.glob("*"):
            safe_delete(f, allowed_roots())
    return n


def apply_retention(days: int) -> int:
    """Delete recordings older than ``days`` (0 disables)."""
    if not days or days <= 0:
        return 0
    ids = repository.recordings_older_than(days)
    for rid in ids:
        delete_recording(rid)
    return len(ids)


def storage_usage() -> dict:
    return {
        "uploads_mb": dir_size_mb(config.UPLOAD_DIR),
        "processed_mb": dir_size_mb(config.PROCESSED_DIR),
        "reports_mb": dir_size_mb(config.REPORTS_DIR),
        "database_mb": dir_size_mb(config.DATABASE_DIR),
        "free_mb": free_disk_mb(config.DATA_DIR),
    }


def check_writable() -> tuple[bool, str]:
    try:
        config.ensure_directories()
        probe = config.TEMP_DIR / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True, f"Writable ({free_disk_mb(config.DATA_DIR) / 1024:.1f} GB free)"
    except OSError as exc:
        return False, f"Cannot write to data directory ({exc.__class__.__name__})"
