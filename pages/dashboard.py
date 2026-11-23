"""Dashboard — portfolio-level statistics, all read from the database."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from database import repository
from ui.components import (PLOTLY_CONFIG, empty_state, metrics_row, page_header, section, status_badge, style_fig)
from utils.time_formatter import format_duration

page_header("Overview", "Dashboard", "Every figure on this page is computed from recordings analysed on this computer.")

stats = repository.dashboard_stats()
recordings = repository.list_recordings()

if stats["recordings"] == 0:
    empty_state(
        "Welcome to TalkTime AI",
        "Analyse calls, interviews and meetings entirely offline. Find out who spoke, for how long, and what was discussed.",
        icon="◉",
        steps=[
            ("Check setup", "Settings → System status confirms models, FFmpeg and storage are ready."),
            ("Upload a recording", "MP3, WAV, M4A, FLAC and more. Files stay on this computer."),
            ("Explore the analysis", "Speaker time, timeline with playback, questions, topics and reports."),
        ],
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Upload recording", type="primary", icon=":material/upload:", use_container_width=True):
        st.switch_page("pages/upload_audio.py")
    if c2.button("Check setup", icon=":material/settings:", use_container_width=True):
        st.switch_page("pages/settings.py")
    failed = [r for r in recordings if r["status"] == "failed"]
    if failed:
        st.warning(f"{len(failed)} recording(s) failed to analyse. Open Upload Recording to retry or Settings to delete them.")
    st.stop()

metrics_row([
    ("Recordings analysed", str(stats["recordings"]), f"{stats['failed']} failed" if stats["failed"] else "All successful", "#199E70"),
    ("Speakers detected", str(stats["total_speakers"]), f"{stats['total_speakers'] / stats['recordings']:.1f} per recording", "#9085E9"),
    ("Audio processed", format_duration(stats["total_duration"]), f"Processing time {format_duration(stats['total_processing'])}", "#D95926"),
    ("Average call duration", format_duration(stats["avg_duration"]), "Completed recordings", "#3987E5"),
])

history = repository.all_metrics()
rows = []
for item in history:
    r, m = item["recording"], item["metrics"]
    sp = m.get("speaking", {})
    rows.append({
        "Recording": r["title"], "Date": r["created_at"], "Duration (min)": r["duration"] / 60,
        "Speakers": sp.get("num_speakers", 0),
        "Balance": m.get("participation", {}).get("balance_score"),
        "Top speaker share": m.get("participation", {}).get("dominant_share", 0) * 100,
        "Speech %": sp.get("summary", {}).get("speech_ratio", 0) * 100,
    })
df = pd.DataFrame(rows)

left, right = st.columns(2, gap="large")
with left:
    section("Speaking-time trend", "Share of speaking time held by the most active speaker in each recording.", tag="measured")
    fig = go.Figure(go.Scatter(
        x=df["Date"], y=df["Top speaker share"], mode="lines+markers", line=dict(color="#3987E5", width=2),
        marker=dict(size=9, color="#3987E5", line=dict(color="#0B1120", width=2)), customdata=df[["Recording"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d %b %H:%M}<br>Top speaker: %{y:.0f}%<extra></extra>"))
    fig.add_hline(y=50, line=dict(color="#475569", dash="dot", width=1))
    fig.update_yaxes(range=[0, 105], ticksuffix="%")
    st.plotly_chart(style_fig(fig, 280, legend=False), use_container_width=True, config=PLOTLY_CONFIG)
with right:
    section("Recording durations", "Length of each analysed recording.", tag="measured")
    fig = go.Figure(go.Bar(
        x=df["Recording"].str.slice(0, 24), y=df["Duration (min)"], marker=dict(color="#199E70", cornerradius=4),
        customdata=df[["Recording", "Speakers"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{y:.1f} min · %{customdata[1]} speakers<extra></extra>"))
    fig.update_yaxes(ticksuffix=" min")
    st.plotly_chart(style_fig(fig, 280, legend=False), use_container_width=True, config=PLOTLY_CONFIG)

left, right = st.columns([3, 2], gap="large")
with left:
    section("Recent analyses")
    with st.container(border=True):
        for r in recordings[:8]:
            c1, c2 = st.columns([5, 1.3])
            c1.markdown(
                f'<div class="tt-row-title">{escape(r["title"])}</div>'
                f'<div class="tt-row-sub">{r["created_at"]:%d %b %Y %H:%M} · {format_duration(r["duration"])} · '
                f'{r["num_speakers"]} speakers · {status_badge(r["status"])}</div>', unsafe_allow_html=True)
            if r["status"] == "completed" and c2.button("Open", key=f"open_{r['id']}", use_container_width=True):
                st.session_state.current_recording = r["id"]
                st.switch_page("pages/speaker_analysis.py")
with right:
    section("Recent activity")
    with st.container(border=True):
        for r in recordings[:8]:
            when = r["updated_at"] or r["created_at"]
            verb = {"completed": "Analysis completed", "failed": "Analysis failed", "processing": "Processing",
                    "pending": "Uploaded"}.get(r["status"], r["status"])
            extra = f" in {format_duration(r['processing_seconds'])}" if r["status"] == "completed" and r["processing_seconds"] else ""
            st.markdown(f'<div class="tt-row"><div><div class="tt-row-title">{verb}{extra}</div>'
                        f'<div class="tt-row-sub">{escape(r["title"])} · {when:%d %b %H:%M}</div></div></div>',
                        unsafe_allow_html=True)

st.caption(f"Data stored locally in {config.DATA_DIR}")
