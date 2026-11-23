"""PDF report generation with ReportLab (charts drawn natively — no browser)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

import config
from core.conversation_summary import with_display_names
from utils.time_formatter import format_duration, format_percent, format_timestamp

INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#64748B")
ACCENT = colors.HexColor("#6366F1")
LINE = colors.HexColor("#E2E8F0")
PANEL = colors.HexColor("#F8FAFC")

LIMITATIONS = [
    "Speaker diarization is automatic and can merge similar voices or split one voice into two; labels are "
    "anonymous (Speaker 1, 2…) unless renamed by a user.",
    "Speaking times are measured from diarized intervals. Overlapping speech is credited to every active speaker, "
    "so per-speaker shares of the recording do not sum to 100%.",
    "Transcription accuracy depends on audio quality, accents, crosstalk and the selected Whisper model size.",
    "'Possible interruptions' are timing-based overlap events, not judgements of intent.",
    "Questions, answers, topics, decisions and action items are extracted with heuristics and may be incomplete or wrong.",
    "Sentiment is a text-only lexicon estimate (English only) and does not measure emotion or mental state.",
]


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=10, textColor=MUTED, spaceAfter=12),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=INK,
                             spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11, textColor=INK,
                             spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=INK),
    }


def _table(rows, col_widths, header=True, zebra=True):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("BACKGROUND", (0, 0), (-1, 0), PANEL),
                  ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
    if zebra:
        for r in range(2 if header else 1, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FBFCFE")))
    t.setStyle(TableStyle(style))
    return t


def _pie(labels, values, cols, size=150):
    d = Drawing(size + 170, size + 10)
    p = Pie()
    p.x, p.y, p.width, p.height = 5, 5, size, size
    p.data = values if sum(values) > 0 else [1]
    p.labels = None
    p.strokeColor = colors.white
    p.strokeWidth = 1
    for i, c in enumerate(cols):
        p.slices[i].fillColor = colors.HexColor(c)
    d.add(p)
    total = sum(values) or 1
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = size - 10 - i * 16
        d.add(Rect(size + 20, y, 9, 9, fillColor=colors.HexColor(cols[i]), strokeColor=None))
        d.add(String(size + 34, y + 1, f"{lab[:22]}  {v / total * 100:.1f}%", fontName="Helvetica", fontSize=8.5, fillColor=INK))
    return d


def _bars(labels, values, cols, width=440, unit="s"):
    h = max(60, 22 * len(labels) + 30)
    d = Drawing(width, h)
    bc = HorizontalBarChart()
    bc.x, bc.y, bc.width, bc.height = 110, 15, width - 140, h - 25
    bc.data = [values]
    bc.categoryAxis.categoryNames = [l[:18] for l in labels]
    bc.categoryAxis.labels.fontSize = 8
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.bars.strokeColor = None
    for i, c in enumerate(cols):
        bc.bars[(0, i)].fillColor = colors.HexColor(c)
    d.add(bc)
    return d


def _timeline(data, names, width=480):
    """Compact speaker-lane timeline of diarized speech."""
    speakers = [s["label"] for s in data["speakers"]]
    dur = data["recording"]["duration"] or 1
    lane = 14
    h = lane * len(speakers) + 24
    d = Drawing(width, h)
    left = 90
    span = width - left - 10
    for i, spk in enumerate(speakers):
        y = h - 14 - (i + 1) * lane
        d.add(String(0, y + 3, names.get(spk, spk)[:16], fontName="Helvetica", fontSize=7.5, fillColor=INK))
        d.add(Rect(left, y + 2, span, lane - 4, fillColor=PANEL, strokeColor=None))
        col = colors.HexColor(next(s["color"] for s in data["speakers"] if s["label"] == spk))
        for seg in data["diarization"]:
            if seg.speaker == spk:
                d.add(Rect(left + seg.start / dur * span, y + 2, max(0.4, seg.duration / dur * span), lane - 4,
                           fillColor=col, strokeColor=None))
    for k in range(5):
        x = left + span * k / 4
        d.add(String(x - 8, 2, format_timestamp(dur * k / 4), fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return d


def generate_pdf(data: dict, output_path: Path, include_transcript: bool = True) -> Path:
    st = _styles()
    rec, m = data["recording"], data["metrics"]
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    cols = [s["color"] for s in data["speakers"]]
    sp = m.get("speaking", {})
    spk = sp.get("speakers", {})
    P = lambda text, style="body": Paragraph(escape(str(text)), st[style])  # noqa: E731

    story = [
        Paragraph(f"{config.APP_NAME} — Conversation Analysis Report", st["title"]),
        Paragraph(escape(f"{rec['title']}  ·  generated {datetime.now():%d %b %Y %H:%M}  ·  processed locally"), st["sub"]),
    ]

    # Recording details
    story.append(Paragraph("Recording details", st["h2"]))
    lang = rec.get("language") or "—"
    if rec.get("language_probability"):
        lang += f" ({rec['language_probability'] * 100:.0f}% confidence)"
    details = [
        ["File", rec["original_filename"], "Duration", format_duration(rec["duration"])],
        ["Analysed", f"{rec['created_at']:%d %b %Y %H:%M}" if rec.get("created_at") else "—", "Speakers", str(sp.get("num_speakers", 0))],
        ["Language", lang, "Processing time", format_duration(rec.get("processing_seconds"))],
        ["Transcription", (rec.get("model_info") or {}).get("transcription", "—"),
         "Diarization", (rec.get("model_info") or {}).get("diarization", "—")],
    ]
    story.append(_table([[P(c, "cell") for c in r] for r in details], [28 * mm, 60 * mm, 30 * mm, 62 * mm], header=False, zebra=False))

    # Summary
    story.append(Paragraph("Analysis summary", st["h2"]))
    story.append(P(with_display_names(m.get("summary", ""), names)))
    if (m.get("llm_summary") or {}).get("text"):
        story.append(Paragraph("Local LLM summary (Ollama — generated, may contain errors)", st["h3"]))
        story.append(P(m["llm_summary"]["text"]))

    # Speaker statistics
    story.append(Paragraph("Speaker statistics", st["h2"]))
    rows = [["Speaker", "Time", "Share of speech", "Share of recording", "Turns", "Avg turn", "Longest", "Int. init/recv"]]
    for label, s in spk.items():
        rows.append([names.get(label, label), format_duration(s["total_time"]), format_percent(s["share_of_speaking_time"]),
                     format_percent(s["share_of_recording"]), str(s["num_turns"]), format_duration(s["avg_turn"]),
                     format_duration(s["longest_turn"]), f"{s['interruptions_initiated']}/{s['interruptions_received']}"])
    story.append(_table(rows, [30 * mm, 18 * mm, 24 * mm, 26 * mm, 14 * mm, 20 * mm, 20 * mm, 24 * mm]))
    story.append(Spacer(1, 4))
    for key, note in sp.get("denominators", {}).items():
        story.append(P(f"{key.replace('_', ' ').capitalize()}: {note}", "small"))
    if spk:
        labels = [names.get(l, l) for l in spk]
        story.append(Spacer(1, 6))
        story.append(KeepTogether([Paragraph("Speaking-time distribution", st["h3"]),
                                   _pie(labels, [s["total_time"] for s in spk.values()], cols)]))
        story.append(KeepTogether([Paragraph("Speaking time per speaker (seconds)", st["h3"]),
                                   _bars(labels, [round(s["total_time"], 1) for s in spk.values()], cols)]))

    # Timeline summary
    story.append(Paragraph("Conversation timeline", st["h2"]))
    summ = sp.get("summary", {})
    tl = [["Speech (union)", format_duration(summ.get("speech_union")), "Non-speech", format_duration(summ.get("non_speech_time"))],
          ["Overlapping speech", format_duration(summ.get("overlap_duration")), "Total turns", str(summ.get("total_turns", 0))]]
    story.append(_table([[P(c, "cell") for c in r] for r in tl], [40 * mm, 50 * mm, 40 * mm, 50 * mm], header=False, zebra=False))
    if data["diarization"]:
        story.append(Spacer(1, 6))
        story.append(_timeline(data, names))

    # Interruptions
    it = m.get("interruptions", {})
    story.append(Paragraph("Overlaps and possible interruptions", st["h2"]))
    story.append(P(f"{it.get('total_overlap_events', 0)} overlap event(s); {it.get('possible_interruptions', 0)} classified as "
                   f"possible interruptions ({it.get('interruptions_per_minute', 0):.2f} per minute)."))
    story.append(P(it.get("note", ""), "small"))
    events = [e for e in it.get("events", []) if e["kind"] == "possible_interruption"][:25]
    if events:
        rows = [["Time", "Already speaking", "Started speaking", "Overlap"]]
        rows += [[format_timestamp(e["start"]), names.get(e["ongoing_speaker"], "—"), names.get(e["incoming_speaker"], "—"),
                  format_duration(e["duration"])] for e in events]
        story.append(_table(rows, [25 * mm, 50 * mm, 50 * mm, 30 * mm]))

    # Silence
    si = m.get("silence", {})
    story.append(Paragraph("Silence and pauses", st["h2"]))
    rows = [["Metric", "Value"],
            [f"Silences ≥ {si.get('thresholds', {}).get('min_silence', 1)}s", str(si.get("num_silences", 0))],
            ["Total silence (≥ threshold)", format_duration(si.get("total_silence"))],
            ["Longest silence", format_duration(si.get("longest_silence"))],
            ["Average silence", format_duration(si.get("average_silence"))],
            ["Median response delay at speaker changes", format_duration(si.get("response_delay_stats", {}).get("median"))]]
    story.append(_table(rows, [90 * mm, 50 * mm]))

    # Participation
    pa = m.get("participation", {})
    story.append(Paragraph("Participation balance", st["h2"]))
    score = pa.get("balance_score")
    story.append(P(f"Balance score: {score:.2f} — {pa.get('balance_label')}" if score is not None else pa.get("balance_label", "")))
    for w in pa.get("warnings", []):
        story.append(P("• " + with_display_names(w, names)))
    story.append(P(pa.get("method", ""), "small"))
    story.append(P(pa.get("caveat", ""), "small"))

    per = pa.get("per_speaker") or {}
    if per:
        story.append(Paragraph("Conversation dynamics", st["h3"]))
        g, dr = pa.get("gini"), pa.get("dominance_ratio")
        story.append(P(f"Gini {'—' if g is None else f'{g:.2f}'} · dominance ratio {'—' if dr is None else f'{dr:.2f}x'} · "
                       f"{pa.get('turns_per_minute', 0):.1f} turns per minute"
                       + (f" · average response time {pa['avg_response_time']:.1f}s" if pa.get("avg_response_time") is not None else "")))
        rows = [["Speaker", "Time share", "Turn share", "Turns/min", "Avg response", "Int. balance", "Questions"]]
        for label, v in per.items():
            rows.append([names.get(label, label), f"{v['time_share'] * 100:.0f}%", f"{v['turn_share'] * 100:.0f}%",
                         f"{v['turns_per_minute']:.1f}",
                         "—" if v["avg_response_time"] is None else f"{v['avg_response_time']:.1f}s",
                         f"{v['interruption_balance']:+d}", str(v["questions_asked"])])
        story.append(_table(rows, [32 * mm, 22 * mm, 22 * mm, 20 * mm, 26 * mm, 24 * mm, 22 * mm]))
        for ind in pa.get("dominance_indicators") or []:
            story.append(P(f"• {ind['indicator']}: {names.get(ind['speaker'], ind['speaker'])} ({ind['value']})", "small"))

    # Sentiment
    se = m.get("sentiment", {})
    story.append(Paragraph("Language sentiment (approximate)", st["h2"]))
    if se.get("available"):
        dist = se.get("overall_distribution", {})
        story.append(P(f"Segments — positive: {dist.get('positive', 0)}, neutral: {dist.get('neutral', 0)}, negative: {dist.get('negative', 0)}."))
        rows = [["Speaker", "Mean score", "Label", "+ / 0 / −"]]
        for label, v in se.get("per_speaker", {}).items():
            d = v["distribution"]
            rows.append([names.get(label, label), f"{v['mean_score']:+.2f}", v["label"],
                         f"{d.get('positive', 0)} / {d.get('neutral', 0)} / {d.get('negative', 0)}"])
        story.append(_table(rows, [50 * mm, 30 * mm, 30 * mm, 40 * mm]))
    else:
        story.append(P(se.get("reason", "Not available.")))
    story.append(P(se.get("disclaimer", ""), "small"))

    # Key points, topics, decisions, actions
    story.append(Paragraph("Key discussion points", st["h2"]))
    topics = m.get("topics", [])
    if topics:
        story.append(P("Topics: " + ", ".join(f"{t['topic']} ({t['mentions']:g})" for t in topics)))
    for kp in m.get("key_points", []):
        story.append(P(f"• [{format_timestamp(kp['start'])}] {names.get(kp['speaker'], kp['speaker'])}: {kp['text']}"))
    ins = m.get("insights", {})
    for kind, title in (("decision", "Possible decisions"), ("action_item", "Possible action items"),
                        ("commitment", "Commitments"), ("important", "Important statements")):
        items = ins.get(kind, [])[:15]
        if items:
            story.append(Paragraph(title, st["h3"]))
            for x in items:
                extra = f" (deadline: {x['deadline']})" if x.get("deadline") else ""
                story.append(P(f"• [{format_timestamp(x['start'])}] {names.get(x['speaker'], x['speaker'])}: {x['text']}{extra}"))

    # Q&A
    qa = m.get("questions", {})
    story.append(Paragraph("Questions and responses", st["h2"]))
    story.append(P(f"{qa.get('total_questions', 0)} question(s) detected; {qa.get('likely_answered', 0)} followed by another "
                   f"speaker's response; {qa.get('possibly_unanswered', 0)} possibly unanswered."))
    rows = [["Time", "Asked by", "Question", "Response"]]
    for q in qa.get("questions", [])[:40]:
        resp = f"{names.get(q['answered_by'], q['answered_by'])} (+{q['response_delay']:.1f}s)" if q["answered"] else "Possibly unanswered"
        rows.append([format_timestamp(q["start"]), names.get(q["speaker"], q["speaker"]), P(q["text"], "cell"), resp])
    if len(rows) > 1:
        story.append(_table(rows, [16 * mm, 28 * mm, 96 * mm, 40 * mm]))
    story.append(P(qa.get("method", ""), "small"))

    # Audio quality, VAD and language
    qa_ = m.get("quality") or {}
    if qa_:
        story.append(Paragraph("Audio quality", st["h2"]))
        fmt = qa_.get("format") or {}
        story.append(P(f"Score {qa_['score']}/100 ({qa_['label']}). Original format: "
                       f"{fmt.get('sample_rate') or '?'} Hz, {fmt.get('channels') or '?'} channel(s), {fmt.get('codec') or '?'}. "
                       f"Speech level {qa_['speech_level_dbfs'] if qa_['speech_level_dbfs'] is not None else '—'} dBFS, "
                       f"noise floor {qa_['noise_floor_dbfs']} dBFS, SNR estimate "
                       f"{qa_['snr_db'] if qa_['snr_db'] is not None else '—'} dB, {qa_['clipping_events']} clipping events."))
        for i in qa_.get("issues", []):
            story.append(P(f"• [{i['severity']}] {i['message']} — {i['recommendation']}", "small"))
    vad_ = m.get("vad") or {}
    if vad_:
        cmp_ = vad_.get("comparison") or {}
        story.append(P(f"Voice activity ({vad_['method']}): {format_duration(vad_['speech_duration'])} of speech "
                       f"({vad_['speech_ratio'] * 100:.0f}% of the recording); agreement with diarized speech "
                       f"{cmp_.get('agreement', 0) * 100:.0f}%; {format_duration(cmp_.get('unattributed_total'))} of speech "
                       "was not assigned to any speaker.", "small"))
    lang_ = m.get("language") or {}
    if lang_:
        story.append(Paragraph("Language", st["h2"]))
        tp = ", ".join(f"{c} {p_ * 100:.0f}%" for c, p_ in (lang_.get("top_probabilities") or [])[:3])
        mix = ", ".join(f"{k} {v * 100:.0f}%" for k, v in (lang_.get("script_mix") or {}).items())
        story.append(P(f"Detected: {lang_.get('detected_name')} ({tp or 'no probabilities stored'}). "
                       f"Scripts in transcript: {mix or '—'}. Code-switching: {'yes' if lang_.get('code_switching') else 'no'}."))
        off = [k for k, v in (lang_.get("feature_support") or {}).items() if not v]
        if off:
            story.append(P("English-only analyses switched off for this language: " + ", ".join(k.replace("_", " ") for k in off) + ".", "small"))

    # Processing performance
    pr = m.get("processing") or {}
    if pr:
        story.append(Paragraph("Processing", st["h2"]))
        res = pr.get("resources") or {}
        stages = ", ".join(f"{k} {v:.1f}s" for k, v in (pr.get("stages") or {}).items())
        line = (f"Processed {format_duration(pr.get('audio_duration'))} of audio in {format_duration(pr['total_seconds'])}"
                + (f" (real-time factor {pr['realtime_factor']:.2f})" if pr.get("realtime_factor") else "") + f". Stages: {stages}.")
        if res.get("available"):
            line += f" Peak memory {res['peak_rss_mb']:.0f} MB"
            if res.get("mean_cpu_percent") is not None:
                line += f", mean CPU {res['mean_cpu_percent']:.0f}% of the machine"
            line += "."
        story.append(P(line, "small"))

    # Limitations
    story.append(Paragraph("Limitations and confidence", st["h2"]))
    al = m.get("alignment", {})
    for label, c in (al.get("per_speaker") or {}).items():
        story.append(P(f"• {names.get(label, label)}: mean speaker-assignment confidence {c['mean_confidence']:.0%}, "
                       f"{c['uncertain']} of {c['segments']} lines uncertain.", "small"))
    story.append(P(f"{al.get('uncertain_segments', 0)} of {al.get('segments', 0)} transcript segments have an uncertain speaker "
                   f"assignment (confidence below {al.get('threshold', 0.6):.0%} or overlapping speech)."))
    for lim in LIMITATIONS:
        story.append(P("• " + lim, "small"))

    # Transcript
    if include_transcript and data["transcript"]:
        story.append(PageBreak())
        story.append(Paragraph("Speaker-wise transcript", st["h2"]))
        rows = [["Time", "Speaker", "Text"]]
        for t in data["transcript"]:
            spk_name = names.get(t.speaker, t.speaker or "Unknown") + (" (?)" if t.uncertain else "")
            rows.append([format_timestamp(t.start), P(spk_name, "cell"), P(t.text, "cell")])
        story.append(_table(rows, [16 * mm, 32 * mm, 132 * mm]))
        story.append(P("(?) = uncertain speaker assignment.", "small"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"{config.APP_NAME} · generated offline · confidential")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title=f"{config.APP_NAME} report", author=config.APP_NAME)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_path
