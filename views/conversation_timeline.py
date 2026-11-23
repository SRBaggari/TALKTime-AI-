"""Interactive timeline with synchronised playback and speaker-wise transcript."""
from __future__ import annotations

from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from core.transcription_engine import is_low_confidence
from services.analysis_service import reassign_segment
from ui.components import (PLOTLY_CONFIG, chip, load, names_and_colors, note, page_header, recording_picker, section,
                           style_fig)
from utils.time_formatter import format_timestamp

page_header("Analysis", "Conversation timeline",
            "Click a bar on the timeline or a transcript line to play the audio from that moment.")
rid = recording_picker()
if not rid:
    st.stop()
data = load(rid)
rec, m = data["recording"], data["metrics"]
names, colors = names_and_colors(data)
duration = rec["duration"] or 1.0
labels = [s["label"] for s in data["speakers"]]

# ---------------------------------------------------------------- player
seek = st.session_state.get("seek")
if not seek or seek.get("rid") != rid:
    seek = {"rid": rid, "start": 0.0, "end": None, "autoplay": False}
audio_path = rec.get("processed_path")
with st.container(border=True):
    if audio_path and Path(audio_path).is_file():
        label = f"Playing from {format_timestamp(seek['start'], True)}"
        if seek["end"]:
            label += f" to {format_timestamp(seek['end'], True)} (segment)"
        st.caption(label)
        st.audio(audio_path, format="audio/wav", start_time=float(seek["start"]),
                 end_time=float(seek["end"]) if seek["end"] else None, autoplay=seek["autoplay"])
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("Restart", icon=":material/replay:", width="stretch"):
            st.session_state.seek = {"rid": rid, "start": 0.0, "end": None, "autoplay": True}
            st.rerun()
        if seek["end"] and c2.button("Play to end", icon=":material/fast_forward:", width="stretch"):
            st.session_state.seek = {**seek, "end": None, "autoplay": True}
            st.rerun()
    else:
        st.warning("The processed audio file for this recording is not available, so playback is disabled. "
                   "Analytics remain available.")


def set_seek(start: float, end: float | None = None) -> None:
    st.session_state.seek = {"rid": rid, "start": max(0.0, float(start)), "end": end, "autoplay": True}


# ---------------------------------------------------------------- timeline chart
section("Timeline", "Speaker lanes show diarized speech. Overlap and silence lanes are derived from those intervals.", tag="measured")
show_overlap = st.toggle("Show overlap and silence lanes", value=True)
lookup: dict[tuple[int, int], tuple[float, float]] = {}
fig = go.Figure()
lane_names = [names[l] for l in labels]
for label in labels:
    segs = [d for d in data["diarization"] if d.speaker == label]
    lookup.update({(len(fig.data), i): (d.start, d.end) for i, d in enumerate(segs)})
    fig.add_bar(
        y=[names[label]] * len(segs), x=[d.duration for d in segs], base=[d.start for d in segs], orientation="h",
        name=names[label], marker=dict(color=colors[label], line=dict(width=0)),
        customdata=[[format_timestamp(d.start, True), format_timestamp(d.end, True), d.duration] for d in segs],
        hovertemplate="<b>" + escape(names[label]) + "</b><br>%{customdata[0]} → %{customdata[1]} (%{customdata[2]:.1f}s)<extra>click to play</extra>")
if show_overlap:
    ov = m.get("speaking", {}).get("overlap_regions", [])
    if ov:
        lookup.update({(len(fig.data), i): (s, e) for i, (s, e) in enumerate(ov)})
        fig.add_bar(y=["Overlap"] * len(ov), x=[e - s for s, e in ov], base=[s for s, _ in ov], orientation="h", name="Overlap",
                    marker=dict(color="#E2E8F0", pattern=dict(shape="/", fgcolor="#0B1120", size=5)),
                    customdata=[[format_timestamp(s, True), format_timestamp(e, True)] for s, e in ov],
                    hovertemplate="<b>Overlapping speech</b><br>%{customdata[0]} → %{customdata[1]}<extra></extra>")
        lane_names.append("Overlap")
    sil = m.get("silence", {}).get("silences", [])
    if sil:
        lookup.update({(len(fig.data), i): (x["start"], x["end"]) for i, x in enumerate(sil)})
        fig.add_bar(y=["Silence"] * len(sil), x=[x["duration"] for x in sil], base=[x["start"] for x in sil], orientation="h",
                    name="Silence", marker=dict(color="#475569"),
                    customdata=[[format_timestamp(x["start"], True), x["duration"], x["kind"].replace("_", " ")] for x in sil],
                    hovertemplate="<b>Silence</b> (%{customdata[2]})<br>%{customdata[0]} · %{customdata[1]:.1f}s<extra></extra>")
        lane_names.append("Silence")
fig.update_layout(barmode="overlay", bargap=0.3)
fig.update_yaxes(automargin=True, ticklabelstandoff=8)
fig.update_xaxes(automargin=True)
fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(lane_names)))
tickvals = [duration * k / 8 for k in range(9)]
fig.update_xaxes(range=[0, duration], tickvals=tickvals, ticktext=[format_timestamp(t) for t in tickvals], showgrid=True)
if seek["start"] > 0:
    fig.add_vline(x=seek["start"], line=dict(color="#F1F5F9", width=1.5, dash="dot"))
event = st.plotly_chart(style_fig(fig, 90 + 46 * len(lane_names)), width="stretch", theme=None, config=PLOTLY_CONFIG,
                        on_select="rerun", selection_mode="points", key=f"timeline_{rid}")
points = (event or {}).get("selection", {}).get("points", []) if event else []
if points:
    p = points[0]
    hit = lookup.get((p.get("curve_number"), p.get("point_index")))
    if hit and st.session_state.get("last_click") != (rid, hit):
        st.session_state.last_click = (rid, hit)
        set_seek(hit[0], hit[1])
        st.rerun()

# ---------------------------------------------------------------- transcript
section("Transcript", "Speaker-wise transcript aligned to diarization by timestamp overlap.", tag="measured")
al = m.get("alignment", {})
if al.get("uncertain_segments"):
    note(f"{al['uncertain_segments']} of {al['segments']} segments are marked “uncertain speaker” (assignment confidence "
         f"below {al.get('threshold', 0.6):.0%}, overlapping speech, or no diarized speech under the words). "
         "Use “Correct” to fix any line; analytics are recalculated.")
transcript = data["transcript"]
if not transcript:
    st.info("No speech was transcribed for this recording.")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 1])
spk_filter = c1.multiselect("Speakers", labels + (["Unknown"] if any(t.speaker == "Unknown" for t in transcript) else []),
                            format_func=lambda l: names.get(l, l), placeholder="All speakers")
query = c2.text_input("Search transcript", placeholder="Find words…")
only_uncertain = c3.toggle("Uncertain only")
rows = [(i, t) for i, t in enumerate(transcript)
        if (not spk_filter or t.speaker in spk_filter) and (not query or query.lower() in t.text.lower())
        and (not only_uncertain or t.uncertain)]
PAGE = 40
pages = max(1, (len(rows) + PAGE - 1) // PAGE)
page = st.number_input(f"Page (of {pages}) · {len(rows)} lines", 1, pages, 1) if pages > 1 else 1
current_t = seek["start"]

with st.container(border=True):
    for i, t in rows[(page - 1) * PAGE: page * PAGE]:
        color = colors.get(t.speaker, "#64748B")
        playing = t.start <= current_t < t.end and seek["start"] > 0
        c1, c2, c3 = st.columns([1.5, 8.5, 1.5], vertical_alignment="top")
        if c1.button(format_timestamp(t.start), key=f"seek_{rid}_{i}", icon=":material/play_arrow:", width="stretch", help="Play this segment",
                     type="primary" if playing else "secondary"):
            set_seek(t.start, t.end)
            st.rerun()
        flags = ""
        if t.uncertain:
            flags += f'<span class="tt-uncertain">⚠ uncertain speaker ({t.speaker_confidence:.0%})</span>'
        if t.overlapping_speakers:
            flags += f' <span class="tt-row-sub">· overlaps {escape(", ".join(names.get(o, o) for o in t.overlapping_speakers))}</span>'
        if is_low_confidence(t):
            flags += ' <span class="tt-row-sub">· low transcription confidence</span>'
        if data["transcript_corrected"][i]:
            flags += ' <span class="tt-row-sub">· corrected</span>'
        c2.markdown(
            f'<div class="tt-seg" style="border-left-color:{color}"><div class="tt-seg-head">'
            f'{chip(names.get(t.speaker, t.speaker or "Unknown"), color)}<span>{format_timestamp(t.start, True)} – '
            f'{format_timestamp(t.end, True)}</span>{flags}</div><div class="tt-seg-text">{escape(t.text)}</div></div>',
            unsafe_allow_html=True)
        with c3.popover("Correct", icon=":material/edit:", width="stretch"):
            st.caption("Attribute this line (and its time span) to another speaker.")
            target = st.selectbox("Speaker", labels, index=labels.index(t.speaker) if t.speaker in labels else 0,
                                  format_func=lambda l: names[l], key=f"fix_{rid}_{i}")
            if st.button("Apply", key=f"apply_{rid}_{i}", type="primary", width="stretch", disabled=target == t.speaker):
                with st.spinner("Recalculating…"):
                    reassign_segment(rid, i, target)
                st.toast("Speaker corrected — analytics updated")
                st.rerun()
