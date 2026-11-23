"""CSV / JSON exports of transcripts and analytics."""
from __future__ import annotations

import io
import json

import pandas as pd

from utils.time_formatter import format_timestamp


def _name(names: dict[str, str], label: str | None) -> str:
    return names.get(label, label or "Unknown")


def transcript_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    rows = []
    for t in data["transcript"]:
        rows.append({
            "start": round(t.start, 2), "end": round(t.end, 2),
            "start_hms": format_timestamp(t.start, show_ms=True), "end_hms": format_timestamp(t.end, show_ms=True),
            "speaker_label": t.speaker, "speaker_name": _name(names, t.speaker), "text": t.text,
            "speaker_confidence": round(t.speaker_confidence, 3), "uncertain_speaker": t.uncertain,
            "overlapping_speakers": ";".join(t.overlapping_speakers),
            "sentiment": t.sentiment, "sentiment_score": t.sentiment_score,
        })
    return pd.DataFrame(rows)


def speakers_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    sp = data["metrics"].get("speaking", {}).get("speakers", {})
    sent = data["metrics"].get("sentiment", {}).get("per_speaker", {})
    q = data["metrics"].get("questions", {}).get("per_speaker", {})
    rows = []
    for label, st in sp.items():
        rows.append({
            "speaker_label": label, "speaker_name": _name(names, label),
            "total_seconds": round(st["total_time"], 2),
            "share_of_speaking_time_pct": round(st["share_of_speaking_time"] * 100, 2),
            "share_of_recording_pct": round(st["share_of_recording"] * 100, 2),
            "turns": st["num_turns"], "avg_turn_seconds": round(st["avg_turn"], 2),
            "longest_turn_seconds": round(st["longest_turn"], 2), "shortest_turn_seconds": round(st["shortest_turn"], 2),
            "first_spoke_seconds": round(st["first_spoke"], 2), "last_spoke_seconds": round(st["last_spoke"], 2),
            "overlap_seconds": round(st["overlap_time"], 2),
            "possible_interruptions_initiated": st["interruptions_initiated"],
            "possible_interruptions_received": st["interruptions_received"],
            "questions_asked": q.get(label, {}).get("asked", 0),
            "mean_sentiment": sent.get(label, {}).get("mean_score"),
        })
    return pd.DataFrame(rows)


def segments_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    return pd.DataFrame([{"speaker_label": d.speaker, "speaker_name": _name(names, d.speaker),
                          "start": round(d.start, 3), "end": round(d.end, 3), "duration": round(d.duration, 3)}
                         for d in data["diarization"]])


def events_dataframe(data: dict) -> pd.DataFrame:
    m = data["metrics"]
    rows = [{"type": e["label"], "start": e["start"], "end": e["end"], "duration": e["duration"],
             "speakers": ";".join(e["speakers"]), "ongoing": e.get("ongoing_speaker"), "incoming": e.get("incoming_speaker")}
            for e in m.get("interruptions", {}).get("events", [])]
    rows += [{"type": f"silence ({s['kind']})", "start": s["start"], "end": s["end"], "duration": s["duration"],
              "speakers": "", "ongoing": s.get("before"), "incoming": s.get("after")}
             for s in m.get("silence", {}).get("silences", [])]
    return pd.DataFrame(rows).sort_values("start") if rows else pd.DataFrame(columns=["type", "start", "end", "duration"])


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel opens UTF-8 correctly


def metrics_json_bytes(data: dict) -> bytes:
    payload = {"recording": {k: (str(v) if k.endswith("_at") else v) for k, v in data["recording"].items()},
               "speakers": data["speakers"], "metrics": data["metrics"]}
    return json.dumps(payload, indent=2, default=str).encode("utf-8")
