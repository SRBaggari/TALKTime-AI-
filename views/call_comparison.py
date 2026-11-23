"""Compare several analysed recordings side by side."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import repository
from ui.components import PLOTLY_CONFIG, empty_state, note, page_header, section, style_fig
from utils.time_formatter import format_duration

page_header("Analysis", "Call comparison",
            "Compare conversation-level metrics across recordings. Speakers are never matched between recordings.")

history = repository.all_metrics()
if len(history) < 2:
    empty_state("Analyse at least two recordings to compare",
                "Comparison shows duration, participation balance, turn length, interruptions, silence and sentiment side by side.",
                icon="⇄")
    st.stop()

by_id = {h["recording"]["id"]: h for h in history}
label = lambda rid: f"{by_id[rid]['recording']['title']} · {by_id[rid]['recording']['created_at']:%d %b %H:%M}"  # noqa: E731
default = [h["recording"]["id"] for h in history[-3:]]
chosen = st.multiselect("Recordings to compare", list(by_id), default=default, format_func=label, max_selections=8)
if len(chosen) < 2:
    st.info("Select at least two recordings.")
    st.stop()

rows = []
for rid in chosen:
    rec, m = by_id[rid]["recording"], by_id[rid]["metrics"]
    sp = m.get("speaking", {})
    spk = sp.get("speakers", {})
    turns = sp.get("turns", [])
    se = m.get("sentiment", {})
    n_seg = sum(se.get("overall_distribution", {}).values()) if se.get("available") else 0
    rows.append({
        "id": rid, "Recording": rec["title"][:40], "Duration": rec["duration"], "Speakers": sp.get("num_speakers", 0),
        "Balance": m.get("participation", {}).get("balance_score"),
        "Top share %": m.get("participation", {}).get("dominant_share", 0) * 100,
        "Avg turn (s)": (sum(t["end"] - t["start"] for t in turns) / len(turns)) if turns else 0,
        "Interruptions / min": m.get("interruptions", {}).get("interruptions_per_minute", 0),
        "Overlap %": sp.get("summary", {}).get("overlap_ratio_of_speech", 0) * 100,
        "Silence (s)": m.get("silence", {}).get("total_silence", 0),
        "Silence %": (m.get("silence", {}).get("total_silence", 0) / rec["duration"] * 100) if rec["duration"] else 0,
        "Questions": m.get("questions", {}).get("total_questions", 0),
        "Positive %": se.get("overall_distribution", {}).get("positive", 0) / n_seg * 100 if n_seg else None,
        "Neutral %": se.get("overall_distribution", {}).get("neutral", 0) / n_seg * 100 if n_seg else None,
        "Negative %": se.get("overall_distribution", {}).get("negative", 0) / n_seg * 100 if n_seg else None,
        "shares": [(repository.speaker_names(rid).get(k, k), v["share_of_speaking_time"] * 100) for k, v in spk.items()],
    })
df = pd.DataFrame(rows)

section("Overview", tag="measured")
table = df[["Recording", "Duration", "Speakers", "Balance", "Top share %", "Avg turn (s)", "Interruptions / min", "Overlap %", "Silence %", "Questions"]].copy()
table["Duration"] = table["Duration"].map(format_duration)
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "Balance": st.column_config.ProgressColumn("Balance (0–1)", min_value=0, max_value=1, format="%.2f"),
    "Top share %": st.column_config.NumberColumn(format="%.0f%%"),
    "Avg turn (s)": st.column_config.NumberColumn(format="%.1f"),
    "Interruptions / min": st.column_config.NumberColumn(format="%.2f"),
    "Overlap %": st.column_config.NumberColumn(format="%.1f%%"),
    "Silence %": st.column_config.NumberColumn(format="%.1f%%"),
})

section("Speaking-time distribution within each recording", tag="measured")
fig = go.Figure()
max_n = max(len(r["shares"]) for r in rows)
palette = ["#3987E5", "#7FAEEB", "#B4CFF2", "#D7E5F7", "#E8EFF8", "#F0F4FA", "#F5F7FB", "#F8FAFC"]
for k in range(max_n):
    fig.add_bar(y=df["Recording"], x=[sorted(r["shares"], key=lambda s: -s[1])[k][1] if k < len(r["shares"]) else 0 for r in rows],
                orientation="h", name=["Largest share", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"][k],
                marker=dict(color=palette[k], line=dict(color="#0B1120", width=2)),
                customdata=[escape(sorted(r["shares"], key=lambda s: -s[1])[k][0]) if k < len(r["shares"]) else "" for r in rows],
                hovertemplate="%{y}<br><b>%{customdata}</b>: %{x:.1f}%<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_xaxes(range=[0, 100], ticksuffix="%")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, 90 + 46 * len(rows)), width="stretch", theme=None, config=PLOTLY_CONFIG)
note("Segments are ordered from the most to least active speaker within each recording. Speaker labels are local to each "
     "recording — 'Speaker 1' in one call is not the same person as 'Speaker 1' in another.")


def metric_chart(col, title, ycol, suffix="", fmt=".1f", color="#3987E5"):
    with col:
        section(title)
        vals = df[ycol].fillna(0)
        f = go.Figure(go.Bar(x=df["Recording"], y=vals, marker=dict(color=color, cornerradius=4),
                             hovertemplate="%{x}<br>%{y:" + fmt + "}" + suffix + "<extra></extra>"))
        f.update_yaxes(ticksuffix=suffix)
        st.plotly_chart(style_fig(f, 260, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)


c1, c2 = st.columns(2, gap="large")
metric_chart(c1, "Participation balance (0–1)", "Balance", fmt=".2f", color="#199E70")
metric_chart(c2, "Average turn duration", "Avg turn (s)", suffix="s", color="#9085E9")
c1, c2 = st.columns(2, gap="large")
metric_chart(c1, "Possible interruptions per minute", "Interruptions / min", fmt=".2f", color="#D95926")
metric_chart(c2, "Silence (share of recording)", "Silence %", suffix="%", color="#64748B")

section("Language sentiment distribution", "Share of transcript segments per class (English recordings only).", tag="inferred")
sdf = df.dropna(subset=["Positive %"])
if sdf.empty:
    st.caption("Sentiment is not available for the selected recordings.")
else:
    fig = go.Figure()
    for cat, col in (("Negative %", "#D55181"), ("Neutral %", "#64748B"), ("Positive %", "#199E70")):
        fig.add_bar(y=sdf["Recording"], x=sdf[cat], name=cat.replace(" %", ""), orientation="h",
                    marker=dict(color=col, line=dict(color="#0B1120", width=2)),
                    hovertemplate="%{y}<br>" + cat.replace(" %", "") + ": %{x:.0f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 90 + 46 * len(sdf)), width="stretch", theme=None, config=PLOTLY_CONFIG)
