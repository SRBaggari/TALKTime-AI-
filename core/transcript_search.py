"""Transcript search: keywords/phrases, speaker and timestamp filters, highlighting.

Query syntax
------------
* ``budget report``     — all words must appear (any order, case-insensitive)
* ``"budget report"``   — exact phrase
* ``1:23`` / ``01:02:03`` / ``83s`` — a timestamp: finds the line being spoken at that moment
Works for any script (Telugu, Hindi, …) because matching is plain Unicode
case-folded substring search.
"""
from __future__ import annotations

import re
from html import escape

from core.types import TranscriptSegment

_TS = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?\s*$|^\s*(\d+(?:\.\d+)?)\s*s\s*$")


def parse_timestamp(text: str) -> float | None:
    """'1:23' → 83.0, '1:02:03' → 3723.0, '83s' → 83.0. Returns None if not a timestamp."""
    m = _TS.match(text or "")
    if not m:
        return None
    if m.group(5) is not None:
        return float(m.group(5))
    h = int(m.group(1) or 0)
    mins, secs = int(m.group(2)), int(m.group(3))
    if secs >= 60:
        return None
    frac = float("0." + m.group(4)) if m.group(4) else 0.0
    return h * 3600 + mins * 60 + secs + frac


def parse_query(query: str) -> list[str]:
    """Split into terms; quoted phrases stay together. Case-folded."""
    terms = [p.casefold() for p in re.findall(r'"([^"]+)"', query or "")]
    rest = re.sub(r'"[^"]*"', " ", query or "")
    terms += [w.casefold() for w in rest.split() if w.strip()]
    return [t for t in terms if t]


def matches(text: str, terms: list[str]) -> bool:
    low = (text or "").casefold()
    return all(t in low for t in terms)


def highlight(text: str, terms: list[str]) -> str:
    """HTML-escaped text with each term wrapped in <mark>."""
    if not terms:
        return escape(text)
    pattern = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.IGNORECASE)
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(escape(text[last:m.start()]))
        out.append(f"<mark>{escape(m.group(0))}</mark>")
        last = m.end()
    out.append(escape(text[last:]))
    return "".join(out)


def segment_at(segments: list[TranscriptSegment], t: float, tolerance: float = 1.0) -> int | None:
    """Index of the line spoken at time ``t`` (or the nearest one within ``tolerance`` s)."""
    best, best_gap = None, tolerance
    for i, s in enumerate(segments):
        if s.start <= t < s.end:
            return i
        gap = min(abs(s.start - t), abs(s.end - t))
        if gap <= best_gap:
            best, best_gap = i, gap
    return best


def search_segments(segments: list[TranscriptSegment], query: str = "", speakers: list[str] | None = None,
                    start: float | None = None, end: float | None = None) -> dict:
    """Return matching segment indices plus the parsed query.

    If ``query`` is a timestamp, the result is the segment at that time (``mode='timestamp'``).
    """
    ts = parse_timestamp(query)
    if ts is not None:
        idx = segment_at(segments, ts)
        hits = [idx] if idx is not None and (not speakers or segments[idx].speaker in speakers) else []
        return {"mode": "timestamp", "time": ts, "terms": [], "indices": hits}
    terms = parse_query(query)
    hits = []
    for i, s in enumerate(segments):
        if speakers and s.speaker not in speakers:
            continue
        if start is not None and s.end < start:
            continue
        if end is not None and s.start > end:
            continue
        if terms and not matches(s.text, terms):
            continue
        hits.append(i)
    return {"mode": "keyword" if terms else "filter", "time": None, "terms": terms, "indices": hits}
