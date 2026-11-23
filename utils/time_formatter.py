"""Human-friendly time formatting helpers."""
from __future__ import annotations

import math


def format_timestamp(seconds: float | None, show_ms: bool = False) -> str:
    """Format seconds as ``M:SS`` / ``H:MM:SS`` (optionally with tenths)."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    seconds = max(0.0, float(seconds))
    if show_ms:
        tenths = int(round(seconds * 10))
        whole, frac = divmod(tenths, 10)
    else:
        whole, frac = int(round(seconds)), None
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return f"{base}.{frac}" if frac is not None else base


def format_duration(seconds: float | None) -> str:
    """Format a duration compactly, e.g. ``1h 02m``, ``3m 07s``, ``4.2s``."""
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    seconds = max(0.0, float(seconds))
    if seconds < 10:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_percent(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"
