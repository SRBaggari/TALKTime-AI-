"""Speaker analysis — per-speaker speaking time, turns and participation."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import repository
from services.analysis_service import merge_speakers
from ui.components import (PLOTLY_CONFIG, chip, load, metrics_row, names_and_colors, note, page_header,
                           recording_picker, section, style_fig)
from utils.time_formatter import format_duration, format_percent, format_timestamp

page_header("Analysis", "Speaker analysis", "Speaking time, turns and participation for each detected speaker.")
rid = recording_picker()
if not rid:
    st.stop()
data = load(rid)
m = data["metrics"]
sp = m.get("speaking", {})
speakers = sp.get("speakers", {})
names, colors = names_and_colors(data)
summary = sp.get("summary", {})
part = m.get("participation", {})

if data["recording"]["diarization_mode"] == "single":
    st.info("This recording was analysed in single-speaker mode — all speech is attributed to one speaker.")

metrics_row([
    ("Recording length", format_duration(sp.get("recording_duration")), data["recording"]["original_filename"], None),
    ("Speakers detected", str(sp.get("num_speakers", 0)), "Automatic diarization" if data["recording"]["diarization_mode"] == "pyannote" else "Single-speaker mode", None),
    ("Speech coverage", format_percent(summary.get("speech_ratio")), f"{format_duration(summary.get('non_speech_time'))} without speech", None),
    ("Overlapping speech", format_duration(summary.get("overlap_duration")), f"{format_percent(summary.get('overlap_ratio_of_speech'))} of speech time", None),
])

if not speakers:
    st.warning("No speaker segments were detected in this recording.")
    st.stop()

# ---------------------------------------------------------------- speaker cards
section("Speakers", "Rename speakers to real names — names are saved and used across all pages and reports.", tag="measured")
labels = list(speakers)
per_row = 3 if len(labels) != 4 else 2
for i in range(0, len(labels), per_row):
    cols = st.columns(per_row, gap="medium")
    for col, label in zip(cols, labels[i:i + per_row]):
        st_ = speakers[label]
        with col.container(border=True):
            st.markdown(chip(names[label], colors[label]) + f'&nbsp;<span class="tt-row-sub">{escape(label)}</span>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="tt-metric-value">{format_duration(st_["total_time"])}</div>'
                f'<div class="tt-metric-sub">{format_percent(st_["share_of_speaking_time"])} of speaking time · '
                f'{format_percent(st_["share_of_recording"])} of recording</div>', unsafe_allow_html=True)
            st.progress(min(1.0, st_["share_of_speaking_time"]))
            a, b = st.columns(2)
            a.markdown(f'<div class="tt-metric-label">Turns</div><b>{st_["num_turns"]}</b>', unsafe_allow_html=True)
            b.markdown(f'<div class="tt-metric-label">Average turn</div><b>{format_duration(st_["avg_turn"])}</b>', unsafe_allow_html=True)
            a.markdown(f'<div class="tt-metric-label">Longest turn</div><b>{format_duration(st_["longest_turn"])}</b>', unsafe_allow_html=True)
            b.markdown(f'<div class="tt-metric-label">Shortest turn</div><b>{format_duration(st_["shortest_turn"])}</b>', unsafe_allow_html=True)
            a.markdown(f'<div class="tt-metric-label">First spoke</div><b>{format_timestamp(st_["first_spoke"])}</b>', unsafe_allow_html=True)
            b.markdown(f'<div class="tt-metric-label">Last spoke</div><b>{format_timestamp(st_["last_spoke"])}</b>', unsafe_allow_html=True)
            with st.form(f"rename_{rid}_{label}", border=False):
                new = st.text_input("Display name", value=names[label], max_chars=100, key=f"name_{rid}_{label}")
                if st.form_submit_button("Save name", width="stretch"):
                    repository.rename_speaker(rid, label, new)
                    st.toast(f"Renamed {label} to {new.strip() or label}")
                    st.rerun()

for key, text in sp.get("denominators", {}).items():
    note(f"{key.replace('_', ' ').capitalize()}: {text}")

# ---------------------------------------------------------------- charts
disp = [names[l] for l in labels]
cols = [colors[l] for l in labels]
left, right = st.columns(2, gap="large")
with left:
    section("Share of speaking time", tag="measured")
    fig = go.Figure(go.Pie(labels=disp, values=[speakers[l]["total_time"] for l in labels], hole=0.62, sort=False,
                           marker=dict(colors=cols, line=dict(color="#0B1120", width=2)), textinfo="label+percent",
                           textfont=dict(color="#F1F5F9"),
                           hovertemplate="<b>%{label}</b><br>%{value:.1f}s · %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{format_duration(summary.get('combined_speaker_time'))}</b><br><span style='font-size:11px;color:#94A3B8'>combined</span>",
                       showarrow=False, font=dict(size=17, color="#F1F5F9"))
    st.plotly_chart(style_fig(fig, 320), width="stretch", theme=None, config=PLOTLY_CONFIG)
with right:
    section("Speaking time and solo vs overlapping", tag="measured")
    fig = go.Figure()
    fig.add_bar(y=disp, x=[speakers[l]["solo_time"] for l in labels], orientation="h", name="Speaking alone",
                marker=dict(color=cols, cornerradius=4), hovertemplate="<b>%{y}</b><br>Alone: %{x:.1f}s<extra></extra>")
    fig.add_bar(y=disp, x=[speakers[l]["overlap_time"] for l in labels], orientation="h", name="Overlapping others",
                marker=dict(color=cols, pattern=dict(shape="/", fgcolor="#0B1120", size=6), cornerradius=4),
                hovertemplate="<b>%{y}</b><br>While others spoke: %{x:.1f}s<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(title="seconds")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 320), width="stretch", theme=None, config=PLOTLY_CONFIG)

left, right = st.columns(2, gap="large")
with left:
    section("Turns per speaker", tag="measured")
    fig = go.Figure(go.Bar(x=disp, y=[speakers[l]["num_turns"] for l in labels], marker=dict(color=cols, cornerradius=4),
                           hovertemplate="<b>%{x}</b><br>%{y} turns<extra></extra>"))
    st.plotly_chart(style_fig(fig, 280, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)
with right:
    section("Turn length distribution", tag="measured")
    turns = pd.DataFrame(sp.get("turns", []))
    fig = go.Figure()
    for l in labels:
        d = turns[turns["speaker"] == l]["end"] - turns[turns["speaker"] == l]["start"] if not turns.empty else []
        fig.add_box(x=d, name=names[l], marker=dict(color=colors[l], size=6), line=dict(color=colors[l], width=2),
                    boxpoints="all", jitter=0.4, pointpos=0, fillcolor="rgba(0,0,0,0)",
                    hovertemplate="%{x:.1f}s<extra>" + escape(names[l]) + "</extra>")
    fig.update_xaxes(title="turn length (seconds)")
    st.plotly_chart(style_fig(fig, 280, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------- participation
section("Participation balance", part.get("method", ""), tag="measured")
c1, c2 = st.columns([1, 2])
score = part.get("balance_score")
c1.markdown(
    f'<div class="tt-card"><div class="tt-metric-label">Balance score</div>'
    f'<div class="tt-metric-value">{"—" if score is None else f"{score:.2f}"}</div>'
    f'<div class="tt-metric-sub">{escape(part.get("balance_label", ""))}</div></div>', unsafe_allow_html=True)
with c2:
    if part.get("dominant_speaker"):
        st.markdown(f"Most active: {chip(names.get(part['dominant_speaker'], part['dominant_speaker']), colors.get(part['dominant_speaker'], '#64748B'))} "
                    f"with {format_percent(part.get('dominant_share'))} of speaking time.", unsafe_allow_html=True)
    if part.get("turn_balance_score") is not None:
        st.markdown(f"Turn-taking balance: **{part['turn_balance_score']:.2f}**")
    for w in part.get("warnings", []):
        st.warning(w, icon=":material/info:")
    note(part.get("caveat", ""))

# ---------------------------------------------------------------- table
section("All speaker statistics")
table = pd.DataFrame([{
    "Speaker": names[l], "Total": format_duration(s["total_time"]), "Share of speech": format_percent(s["share_of_speaking_time"]),
    "Share of recording": format_percent(s["share_of_recording"]), "Turns": s["num_turns"], "Avg turn": format_duration(s["avg_turn"]),
    "Longest": format_duration(s["longest_turn"]), "Shortest": format_duration(s["shortest_turn"]),
    "Overlap": format_duration(s["overlap_time"]), "Possible interruptions made": s["interruptions_initiated"],
    "Possible interruptions received": s["interruptions_received"],
} for l, s in speakers.items()])
st.dataframe(table, hide_index=True, width="stretch")

# ---------------------------------------------------------------- corrections
with st.expander("Correct speaker detection", icon=":material/merge:"):
    st.caption("If the model split one person into two speakers, merge them. All statistics are recalculated from the "
               "stored segments (no re-transcription). To fix individual lines, use the Conversation Timeline page.")
    if len(labels) < 2:
        st.write("Only one speaker detected — nothing to merge.")
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        src = c1.selectbox("Merge this speaker", labels, format_func=lambda l: names[l], key="merge_src")
        dst = c2.selectbox("into", [l for l in labels if l != src], format_func=lambda l: names[l], key="merge_dst")
        c3.write("")
        if c3.button("Merge", type="primary", width="stretch"):
            with st.spinner("Recalculating analytics…"):
                merge_speakers(rid, src, dst)
            st.toast(f"Merged {names[src]} into {names[dst]}")
            st.rerun()
