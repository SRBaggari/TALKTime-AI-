"""Processing performance, resource usage and error/recovery information."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import repository
from services.analysis_service import JobManager
from services.resource_monitor import system_snapshot
from ui.components import PLOTLY_CONFIG, badge, metrics_row, note, page_header, section, style_fig
from utils.time_formatter import format_duration

page_header("Workspace", "Processing monitor",
            "Measured processing time, model loading, CPU and memory use, and errors — all from this computer.")


@st.fragment(run_every=2.0)
def live() -> None:
    snap = system_snapshot()
    active = JobManager.active()
    if not snap.get("available"):
        st.info("psutil is not installed — live CPU/memory figures are unavailable.")
        return
    metrics_row([
        ("CPU (machine)", f"{snap['cpu_percent']:.0f}%", f"{snap['cpu_count']} logical cores", None),
        ("Memory used", f"{snap['memory_percent']:.0f}%",
         f"{snap['memory_total_gb'] - snap['memory_available_gb']:.1f} of {snap['memory_total_gb']:.1f} GB", None),
        ("TalkTime AI process", f"{snap['process_rss_mb']:.0f} MB", "Resident memory", None),
        ("Analysis job", "Running" if active else "Idle",
         f"{JobManager.status(active)['progress'] * 100:.0f}% · {JobManager.status(active)['message'][:40]}" if active else "Ready", None),
    ])
    if snap["memory_available_gb"] < 2.0:
        st.warning("Less than 2 GB of memory is free. Close other programs or choose a smaller Whisper model.")


section("Live system usage", "Refreshes every 2 seconds. Inference runs on the CPU — no GPU is assumed.", tag="measured")
live()

items = repository.all_metrics()
rows = []
for it in items:
    r, p = it["recording"], (it["metrics"].get("processing") or {})
    res = p.get("resources") or {}
    rows.append({
        "id": r["id"], "Recording": r["title"][:40], "Date": r["created_at"], "Audio (s)": r["duration"],
        "Processing (s)": p.get("total_seconds") or r["processing_seconds"] or 0,
        "Real-time factor": p.get("realtime_factor"),
        "Transcription model load (s)": (p.get("model_load_seconds") or {}).get("transcription"),
        "Diarization model load (s)": (p.get("model_load_seconds") or {}).get("diarization"),
        "Peak memory (MB)": res.get("peak_rss_mb"), "Mean CPU (%)": res.get("mean_cpu_percent"),
        "stages": p.get("stages") or {}, "model": (p.get("settings") or {}).get("whisper_model", ""),
    })
df = pd.DataFrame(rows)

if df.empty:
    st.info("No completed analyses yet. Performance figures appear here after the first recording is processed.")
else:
    total_audio = df["Audio (s)"].sum()
    total_proc = df["Processing (s)"].sum()
    metrics_row([
        ("Recordings processed", str(len(df)), "Completed analyses", None),
        ("Audio processed", format_duration(total_audio), "Total duration", None),
        ("Processing time", format_duration(total_proc), "Wall-clock, measured", None),
        ("Average real-time factor", f"{total_proc / total_audio:.2f}×" if total_audio else "—",
         "Processing time ÷ audio duration (lower is faster)", None),
    ])
    left, right = st.columns(2, gap="large")
    with left:
        section("Audio duration vs processing time", tag="measured")
        fig = go.Figure(go.Scatter(x=df["Audio (s)"], y=df["Processing (s)"], mode="markers",
                                   marker=dict(size=11, color="#3987E5", line=dict(color="#0B1120", width=2)),
                                   customdata=df[["Recording", "Real-time factor"]],
                                   hovertemplate="<b>%{customdata[0]}</b><br>%{x:.0f}s audio → %{y:.0f}s processing"
                                                 "<br>RTF %{customdata[1]:.2f}<extra></extra>"))
        mx = max(df["Audio (s)"].max(), df["Processing (s)"].max()) * 1.05
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="#475569", dash="dot", width=1),
                        hoverinfo="skip", name="real time (1×)")
        fig.update_xaxes(title="audio duration (s)", rangemode="tozero")
        fig.update_yaxes(title="processing time (s)", rangemode="tozero")
        st.plotly_chart(style_fig(fig, 320, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)
        note("Points below the dotted line were processed faster than real time.")
    with right:
        section("Time per processing stage", tag="measured")
        stage_order = ["decode", "vad", "quality", "transcription", "diarization", "analytics"]
        palette = {"decode": "#64748B", "vad": "#9085E9", "quality": "#C98500", "transcription": "#3987E5",
                   "diarization": "#199E70", "analytics": "#D95926"}
        fig = go.Figure()
        recent = df.tail(10)
        for stg in stage_order:
            vals = [r.get(stg, 0) for r in recent["stages"]]
            if any(vals):
                fig.add_bar(y=recent["Recording"], x=vals, name=stg.capitalize(), orientation="h",
                            marker=dict(color=palette[stg], line=dict(color="#0B1120", width=2)),
                            hovertemplate="%{y}<br>" + stg + ": %{x:.1f}s<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="seconds")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 320), width="stretch", theme=None, config=PLOTLY_CONFIG)

    section("Per-recording details")
    show = df.drop(columns=["id", "stages"]).copy()
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Date": st.column_config.DatetimeColumn(format="D MMM YYYY HH:mm"),
        "Audio (s)": st.column_config.NumberColumn(format="%.1f"),
        "Processing (s)": st.column_config.NumberColumn(format="%.1f"),
        "Real-time factor": st.column_config.NumberColumn(format="%.2f"),
        "Transcription model load (s)": st.column_config.NumberColumn(format="%.1f"),
        "Diarization model load (s)": st.column_config.NumberColumn(format="%.1f"),
        "Peak memory (MB)": st.column_config.NumberColumn(format="%.0f"),
        "Mean CPU (%)": st.column_config.NumberColumn(format="%.0f"),
        "model": st.column_config.TextColumn("Whisper model"),
    })
    note("A model load time of 0 means the model was already in memory from a previous analysis. "
         "CPU % is relative to the whole machine (100% = all cores busy).")

section("Errors and recovery")
failed = repository.list_recordings(status="failed")
if not failed:
    st.markdown(badge("No failed analyses", "ok"), unsafe_allow_html=True)
for r in failed[:20]:
    info = r.get("model_info") or {}
    with st.container(border=True):
        a, b = st.columns([5, 1], vertical_alignment="center")
        stage = info.get("failed_stage")
        a.markdown(f"**{escape(r['title'])}** · {r['created_at']:%d %b %Y %H:%M}" +
                   (f" · failed during **{stage}**" if stage else ""))
        a.caption((r["error"] or "Unknown error")[:400])
        if info.get("hint"):
            a.markdown(f"Recovery: {info['hint']}")
        if b.button("Retry", key=f"retry_{r['id']}", width="stretch", disabled=not r["stored_path"]):
            st.session_state.active_job = None
            st.switch_page("views/upload_audio.py")
