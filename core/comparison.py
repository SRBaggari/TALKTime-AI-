"""Cross-recording comparison and historical trends.

Only *conversation-level* metrics are compared. Speaker shares are compared as
ranked distributions (largest share, 2nd, …) because "Speaker 1" in one
recording is not the same person as "Speaker 1" in another. No voice matching
across recordings is performed (that would require storing voiceprints —
biometric data — which TalkTime AI deliberately does not do).
"""
from __future__ import annotations

import statistics

TREND_METRICS = {
    "duration_min": ("Duration", "min"),
    "balance": ("Participation balance (0–1)", ""),
    "top_share_pct": ("Most active speaker's share", "%"),
    "avg_turn_s": ("Average turn", "s"),
    "turns_per_min": ("Turns per minute", ""),
    "interruptions_per_min": ("Possible interruptions per minute", ""),
    "silence_pct": ("Silence (share of recording)", "%"),
    "avg_pause_s": ("Average pause within a speaker", "s"),
    "median_response_s": ("Median response delay", "s"),
    "quality_score": ("Audio quality score", ""),
}


def recording_summary(rec: dict, m: dict) -> dict:
    sp = m.get("speaking", {})
    spk = sp.get("speakers", {})
    turns = sp.get("turns", [])
    pa = m.get("participation", {})
    si = m.get("silence", {})
    se = m.get("sentiment", {})
    dur = rec.get("duration") or sp.get("recording_duration") or 0.0
    pauses = list((si.get("within_speaker_pauses") or {}).values())
    pause_total = sum(p["total"] for p in pauses)
    pause_count = sum(p["count"] for p in pauses)
    n_seg = sum((se.get("overall_distribution") or {}).values()) if se.get("available") else 0
    dist = se.get("overall_distribution") or {}
    return {
        "id": rec["id"], "title": rec["title"], "created_at": rec["created_at"],
        "duration_min": dur / 60, "speakers": sp.get("num_speakers", 0),
        "balance": pa.get("balance_score"), "gini": pa.get("gini"),
        "top_share_pct": (pa.get("dominant_share") or 0) * 100,
        "ranked_shares_pct": sorted((v["share_of_speaking_time"] * 100 for v in spk.values()), reverse=True),
        "avg_turn_s": (sum(t["end"] - t["start"] for t in turns) / len(turns)) if turns else 0.0,
        "turns_per_min": (len(turns) / (dur / 60)) if dur else 0.0,
        "interruptions_per_min": m.get("interruptions", {}).get("interruptions_per_minute", 0.0),
        "overlap_pct": sp.get("summary", {}).get("overlap_ratio_of_speech", 0) * 100,
        "silence_s": si.get("total_silence", 0.0),
        "silence_pct": (si.get("total_silence", 0.0) / dur * 100) if dur else 0.0,
        "avg_pause_s": (pause_total / pause_count) if pause_count else 0.0,
        "median_response_s": (si.get("response_delay_stats") or {}).get("median", 0.0),
        "questions": m.get("questions", {}).get("total_questions", 0),
        "positive_pct": dist.get("positive", 0) / n_seg * 100 if n_seg else None,
        "neutral_pct": dist.get("neutral", 0) / n_seg * 100 if n_seg else None,
        "negative_pct": dist.get("negative", 0) / n_seg * 100 if n_seg else None,
        "quality_score": (m.get("quality") or {}).get("score"),
        "language": (m.get("language") or {}).get("detected_name") or rec.get("language"),
    }


def compare(items: list[dict]) -> list[dict]:
    """``items``: [{'recording': ..., 'metrics': ...}] → one summary row per recording."""
    return [recording_summary(i["recording"], i["metrics"]) for i in items]


def trend(rows: list[dict], metric: str) -> dict:
    """Chronological series for one metric plus a simple first-half vs second-half change."""
    ordered = sorted((r for r in rows if r.get(metric) is not None), key=lambda r: r["created_at"])
    values = [r[metric] for r in ordered]
    out = {"metric": metric, "label": TREND_METRICS.get(metric, (metric, ""))[0],
           "points": [(r["created_at"], r[metric], r["title"]) for r in ordered], "change": None}
    if len(values) >= 4:
        half = len(values) // 2
        out["change"] = statistics.fmean(values[half:]) - statistics.fmean(values[:half])
    return out
