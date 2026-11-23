"""Exact interval arithmetic on half-open [start, end) intervals.

Every speaking-time figure in TalkTime AI is derived from these functions, so
they are kept small, pure and heavily tested.
"""
from __future__ import annotations

from typing import Iterable

Interval = tuple[float, float]


def normalize(intervals: Iterable[Interval]) -> list[Interval]:
    """Drop empty/inverted intervals and sort."""
    return sorted((float(s), float(e)) for s, e in intervals if e > s)


def merge(intervals: Iterable[Interval], gap: float = 0.0) -> list[Interval]:
    """Union of intervals. Intervals separated by <= ``gap`` seconds are joined."""
    out: list[list[float]] = []
    for s, e in normalize(intervals):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total(intervals: Iterable[Interval]) -> float:
    """Total length of the *union* of intervals (overlaps counted once)."""
    return sum(e - s for s, e in merge(intervals))


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    """Intersection of two interval sets (each merged first)."""
    a, b = merge(a), merge(b)
    i = j = 0
    out: list[Interval] = []
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    """Parts of ``a`` not covered by ``b``."""
    b = merge(b)
    out: list[Interval] = []
    for s, e in merge(a):
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if cur >= e:
                break
        if cur < e:
            out.append((cur, e))
    return out


def complement(intervals: Iterable[Interval], start: float, end: float) -> list[Interval]:
    """Gaps inside [start, end) not covered by ``intervals``."""
    return subtract([(start, end)], intervals)


def overlap_length(a: Interval, b: Interval) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def multi_coverage(groups: dict[str, list[Interval]], min_count: int = 2) -> list[Interval]:
    """Regions where at least ``min_count`` distinct groups are active (sweep line)."""
    events: list[tuple[float, int]] = []
    for ivs in groups.values():
        for s, e in merge(ivs):
            events.append((s, 1))
            events.append((e, -1))
    # Process ends before starts at the same instant (half-open intervals).
    events.sort(key=lambda x: (x[0], x[1]))
    out: list[Interval] = []
    active = 0
    region_start = None
    for t, delta in events:
        before = active
        active += delta
        if before < min_count <= active:
            region_start = t
        elif before >= min_count > active and region_start is not None:
            if t > region_start:
                out.append((region_start, t))
            region_start = None
    return merge(out)
