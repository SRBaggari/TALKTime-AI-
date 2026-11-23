"""Upload validation: extension allow-list, size limits and magic-byte sniffing.

Uploaded files are only ever *decoded* (by FFmpeg/libsndfile), never executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detected_format: str | None = None


def sniff_format(header: bytes) -> str | None:
    """Identify common audio containers from their first bytes."""
    if len(header) < 4:
        return None
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:4] == b"fLaC":
        return "flac"
    if header[:4] == b"OggS":
        return "ogg"
    if header[:3] == b"ID3":
        return "mp3"
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:  # MPEG frame sync (MP3 / ADTS AAC)
        return "aac" if (header[1] & 0x06) == 0 else "mp3"
    if header[4:8] == b"ftyp":
        return "mp4"  # m4a / mp4 / aac in ISO-BMFF
    if header[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    if header[:16] == bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C"):
        return "wma"
    return None


_EXT_TO_FORMATS = {
    ".wav": {"wav"},
    ".flac": {"flac"},
    ".mp3": {"mp3"},
    ".ogg": {"ogg"},
    ".opus": {"ogg"},
    ".m4a": {"mp4"},
    ".mp4": {"mp4"},
    ".aac": {"aac", "mp4"},
    ".webm": {"webm"},
    ".wma": {"wma"},
}

_DANGEROUS_HEADERS = (b"MZ", b"\x7fELF", b"#!", b"PK\x03\x04", b"%PDF", b"<script", b"<html", b"<?php")


def validate_upload(filename: str, size_bytes: int, header: bytes, max_mb: int | None = None) -> ValidationResult:
    """Validate an uploaded file before it is written to permanent storage."""
    result = ValidationResult(ok=True)
    max_mb = max_mb or config.DEFAULT_MAX_UPLOAD_MB
    ext = Path(filename or "").suffix.lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        result.errors.append(
            f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )
    if size_bytes <= 0:
        result.errors.append("The file is empty.")
    elif size_bytes > max_mb * 1024 * 1024:
        result.errors.append(f"File is {size_bytes / 1048576:.1f} MB — the limit is {max_mb} MB.")

    if any(header.startswith(sig) for sig in _DANGEROUS_HEADERS):
        result.errors.append("File content is not audio (executable/document signature detected).")

    detected = sniff_format(header)
    result.detected_format = detected
    if size_bytes > 0 and not result.errors:
        expected = _EXT_TO_FORMATS.get(ext)
        if detected is None:
            result.warnings.append("Could not recognise the audio container from its header; decoding will be attempted.")
        elif expected and detected not in expected:
            result.warnings.append(
                f"Extension '{ext}' does not match detected content ({detected}); decoding will be attempted."
            )

    result.ok = not result.errors
    return result
