"""CSV, JSON, XLSX, speaker-wise TXT and interactive HTML exports, all built from stored analysis results."""
from __future__ import annotations

import io
import json
import re

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


# --------------------------------------------------------------------------- additional tables

def questions_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    rows = [{"time": format_timestamp(q["start"]), "start": q["start"], "asked_by": _name(names, q["speaker"]),
             "question": q["text"], "detection_confidence": q["confidence"], "detection_method": q["method"],
             "likely_answered": q["answered"], "answered_by": _name(names, q["answered_by"]) if q["answered"] else "",
             "response_delay_s": q["response_delay"], "response": q["answer_text"] or "",
             "match_confidence": q["answer_confidence"]}
            for q in data["metrics"].get("questions", {}).get("questions", [])]
    return pd.DataFrame(rows)


def insights_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    rows = []
    for kind, items in (data["metrics"].get("insights") or {}).items():
        for x in items:
            rows.append({"type": kind.replace("_", " "), "time": format_timestamp(x["start"]), "start": x["start"],
                         "speaker": _name(names, x.get("speaker")), "text": x["text"],
                         "deadline": x.get("deadline") or "", "confidence": x.get("confidence"),
                         "method": x.get("method", "rule-based")})
    for t in data["metrics"].get("topic_recurrence") or []:
        first = t["episodes"][0]["start"] if t["episodes"] else None
        rows.append({"type": "topic", "time": format_timestamp(first) if first is not None else "", "start": first,
                     "speaker": "", "text": t["topic"], "deadline": "", "confidence": None,
                     "method": f"{len(t['episodes'])} discussion episode(s)" + (" (revisited)" if t["revisited"] else "")})
    return pd.DataFrame(rows).sort_values("start", na_position="last") if rows else pd.DataFrame()


def participation_dataframe(data: dict) -> pd.DataFrame:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    per = data["metrics"].get("participation", {}).get("per_speaker", {})
    conf = data["metrics"].get("alignment", {}).get("per_speaker", {})
    return pd.DataFrame([{
        "speaker": _name(names, k), "time_share_pct": round(v["time_share"] * 100, 2),
        "turn_share_pct": round(v["turn_share"] * 100, 2), "turns_per_minute": round(v["turns_per_minute"], 2),
        "avg_turn_s": round(v["avg_turn"], 2), "longest_turn_s": round(v["longest_turn"], 2),
        "avg_response_time_s": None if v["avg_response_time"] is None else round(v["avg_response_time"], 2),
        "interruption_balance": v["interruption_balance"], "questions_asked": v["questions_asked"],
        "mean_assignment_confidence": conf.get(k, {}).get("mean_confidence"),
        "uncertain_lines": conf.get(k, {}).get("uncertain"),
    } for k, v in per.items()])


def summary_rows(data: dict) -> pd.DataFrame:
    rec, m = data["recording"], data["metrics"]
    sp, pa = m.get("speaking", {}), m.get("participation", {})
    q, lang, perf = m.get("quality") or {}, m.get("language") or {}, m.get("processing") or {}
    rows = [
        ("Title", rec["title"]), ("File", rec["original_filename"]), ("Analysed", str(rec["created_at"])),
        ("Duration (s)", round(rec["duration"], 2)), ("Speakers", sp.get("num_speakers")),
        ("Language", lang.get("detected_name") or rec.get("language")),
        ("Language probability", rec.get("language_probability")),
        ("Code-switching detected", lang.get("code_switching")),
        ("Speech coverage (%)", round(sp.get("summary", {}).get("speech_ratio", 0) * 100, 2)),
        ("Overlapping speech (s)", sp.get("summary", {}).get("overlap_duration")),
        ("Balance score (entropy)", pa.get("balance_score")), ("Gini (speaking time)", pa.get("gini")),
        ("Dominance ratio", pa.get("dominance_ratio")), ("Turns per minute", pa.get("turns_per_minute")),
        ("Possible interruptions", m.get("interruptions", {}).get("possible_interruptions")),
        ("Total silence (s)", m.get("silence", {}).get("total_silence")),
        ("Questions", m.get("questions", {}).get("total_questions")),
        ("Audio quality score", q.get("score")), ("Audio quality", q.get("label")),
        ("Processing time (s)", perf.get("total_seconds")), ("Real-time factor", perf.get("realtime_factor")),
        ("Summary", m.get("summary")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def quality_dataframe(data: dict) -> pd.DataFrame:
    q = data["metrics"].get("quality") or {}
    if not q:
        return pd.DataFrame()
    rows = [("Score", q["score"]), ("Label", q["label"]), ("Sample rate (original)", q["format"].get("sample_rate")),
            ("Channels", q["format"].get("channels")), ("Codec", q["format"].get("codec")),
            ("Speech level (dBFS)", q["speech_level_dbfs"]), ("Noise floor (dBFS)", q["noise_floor_dbfs"]),
            ("SNR estimate (dB)", q["snr_db"]), ("Peak (dBFS)", q["peak_dbfs"]),
            ("Clipping ratio", q["clipping_ratio"]), ("Clipping events", q["clipping_events"]),
            ("Bandwidth (Hz)", q["bandwidth_hz"])]
    rows += [(f"Issue ({i['severity']})", f"{i['message']} Recommendation: {i['recommendation']}") for i in q["issues"]]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def performance_dataframe(data: dict) -> pd.DataFrame:
    p = data["metrics"].get("processing") or {}
    if not p:
        return pd.DataFrame()
    rows = [("Total processing (s)", round(p["total_seconds"], 2)),
            ("Audio duration (s)", round(p.get("audio_duration") or 0, 2)),
            ("Real-time factor", round(p["realtime_factor"], 3) if p.get("realtime_factor") else None)]
    rows += [(f"Stage: {k} (s)", round(v, 2)) for k, v in p.get("stages", {}).items()]
    rows += [(f"Model load: {k} (s)", round(v, 2)) for k, v in (p.get("model_load_seconds") or {}).items()]
    r = p.get("resources") or {}
    if r.get("available"):
        rows += [("Peak memory (MB)", round(r["peak_rss_mb"], 1)),
                 ("Mean CPU (% of machine)",
                  round(r["mean_cpu_percent"], 1) if r.get("mean_cpu_percent") is not None else None)]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# --------------------------------------------------------------------------- XLSX

def xlsx_bytes(data: dict) -> bytes:
    """Multi-sheet Excel workbook with every analysis table, plus one transcript sheet per speaker."""
    sheets = {
        "Summary": summary_rows(data),
        "Speakers": speakers_dataframe(data),
        "Participation": participation_dataframe(data),
        "Transcript": transcript_dataframe(data),
        "Speaker segments": segments_dataframe(data),
        "Overlaps & silence": events_dataframe(data),
        "Questions": questions_dataframe(data),
        "Insights & topics": insights_dataframe(data),
        "Audio quality": quality_dataframe(data),
        "Performance": performance_dataframe(data),
    }
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    full = transcript_dataframe(data)
    for label in [s["label"] for s in data["speakers"]][:20]:
        if not full.empty:
            sheets[f"Transcript - {_name(names, label)}"] = full[full["speaker_label"] == label]
    buf = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
            while safe in used:
                safe = safe[:28] + f"_{len(used)}"
            used.add(safe)
            (df if not df.empty else pd.DataFrame({"info": ["No data"]})).to_excel(xw, sheet_name=safe, index=False)
            ws = xw.sheets[safe]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(80, max(10, width + 2))
    return buf.getvalue()


# --------------------------------------------------------------------------- speaker-wise TXT

def speaker_transcript_text(data: dict) -> str:
    """Plain-text transcript: chronological, then one section per speaker."""
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    rec = data["recording"]
    lines = [f"{rec['title']} - transcript",
             f"File: {rec['original_filename']} | Duration: {format_timestamp(rec['duration'])}",
             "(?) = uncertain speaker assignment", "", "=== Full conversation ===", ""]
    for t in data["transcript"]:
        flag = " (?)" if t.uncertain else ""
        lines.append(f"[{format_timestamp(t.start)}] {_name(names, t.speaker)}{flag}: {t.text}")
    for s in data["speakers"]:
        own = [t for t in data["transcript"] if t.speaker == s["label"]]
        lines += ["", f"=== {s['display_name']} ({len(own)} lines) ===", ""]
        lines += [f"[{format_timestamp(t.start)}] {t.text}" for t in own]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- interactive HTML

def interactive_html(data: dict) -> str:
    """Self-contained HTML report (plotly.js embedded, so it opens offline) with interactive charts."""
    from html import escape

    import plotly.graph_objects as go
    from plotly.io import to_html

    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    colors = {s["label"]: s["color"] for s in data["speakers"]}
    m = data["metrics"]
    spk = m.get("speaking", {}).get("speakers", {})
    base = dict(template="plotly_white", height=360, margin=dict(l=70, r=20, t=50, b=40))
    figs = []
    if spk:
        figs.append(go.Figure(go.Pie(labels=[names.get(k, k) for k in spk], values=[v["total_time"] for v in spk.values()],
                                     marker=dict(colors=[colors.get(k) for k in spk]), hole=0.55, sort=False),
                              layout=dict(title="Share of speaking time", **base)))
        tl = go.Figure(layout=dict(title="Conversation timeline (seconds)", barmode="overlay", **base))
        for label in spk:
            segs = [d for d in data["diarization"] if d.speaker == label]
            tl.add_bar(y=[names.get(label, label)] * len(segs), x=[d.duration for d in segs],
                       base=[d.start for d in segs], orientation="h", name=names.get(label, label),
                       marker_color=colors.get(label))
        figs.append(tl)
        figs.append(go.Figure(go.Bar(x=[names.get(k, k) for k in spk], y=[v["num_turns"] for v in spk.values()],
                                     marker_color=[colors.get(k) for k in spk]),
                              layout=dict(title="Turns per speaker", **base)))
    se = m.get("sentiment", {})
    if se.get("available") and se.get("timeline"):
        pts = se["timeline"]
        figs.append(go.Figure(go.Scatter(x=[p["time"] for p in pts], y=[p["rolling"] for p in pts], mode="lines"),
                              layout=dict(title="Language sentiment, rolling average (text-based estimate)",
                                          yaxis_range=[-1, 1], **base)))
    parts = [to_html(f, include_plotlyjs=(i == 0), full_html=False) for i, f in enumerate(figs)]
    title = escape(data["recording"]["title"])
    head = ("<!doctype html><html><head><meta charset=\"utf-8\"><title>TalkTime AI - " + title + "</title>"
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;max-width:1100px;margin:24px auto;"
            "padding:0 16px;color:#0f172a}p{color:#475569}</style></head><body>")
    intro = ("<h1>" + title + "</h1><p>Interactive charts exported from TalkTime AI. All values come from this "
             "recording's analysis. The file is self-contained and works offline.</p>")
    return head + intro + "".join(parts) + "</body></html>"
