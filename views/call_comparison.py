"""Compare analysed recordings and view historical trends."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import comparison
from database import repository
from ui.components import PLOTLY_CONFIG, empty_state, note, page_header, section, style_fig
from utils.time_formatter import format_duration

page_header("Analysis", "Call comparison",
            "Compare conversation-level metrics across recordings and follow trends over time.")

history = repository.all_metrics()
if len(history) < 2:
    empty_state("Analyse at least two recordings to compare",
                "Comparison covers duration, speaking-time distribution, participation, turn-taking, interruptions, "
                "silence and pauses, sentiment and audio quality.", icon="⇄")
    st.stop()

all_rows = comparison.compare(history)
by_id = {r["id"]: r for r in all_rows}
label = lambda rid: f"{by_id[rid]['title']} · {by_id[rid]['created_at']:%d %b %H:%M}"  # noqa: E731
chosen = st.multiselect("Recordings to compare", list(by_id), default=[r["id"] for r in all_rows[-3:]],
                        format_func=label, max_selections=10)
st.markdown('<div class="tt-note">Speakers are never matched across recordings — “Speaker 1” in one call is not '
            'assumed to be the same person in another. Distributions are compared by rank (largest share, 2nd, …). '
            'Cross-recording voice matching is not offered because it would require storing voiceprints (biometric data).</div>',
            unsafe_allow_html=True)
if len(chosen) < 2:
    st.info("Select at least two recordings.")
    st.stop()
rows = [by_id[r] for r in chosen]
df = pd.DataFrame(rows)
df["Recording"] = df["title"].str.slice(0, 32)

section("Overview", tag="measured")
table = df[["Recording", "duration_min", "speakers", "language", "balance", "gini", "top_share_pct", "avg_turn_s",
            "turns_per_min", "interruptions_per_min", "silence_pct", "avg_pause_s", "median_response_s", "questions",
            "quality_score"]]
st.dataframe(table, hide_index=True, width="stretch", column_config={
    "duration_min": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
    "speakers": "Speakers", "language": "Language",
    "balance": st.column_config.ProgressColumn("Balance", min_value=0, max_value=1, format="%.2f"),
    "gini": st.column_config.NumberColumn("Gini", format="%.2f"),
    "top_share_pct": st.column_config.NumberColumn("Top share", format="%.0f%%"),
    "avg_turn_s": st.column_config.NumberColumn("Avg turn (s)", format="%.1f"),
    "turns_per_min": st.column_config.NumberColumn("Turns/min", format="%.1f"),
    "interruptions_per_min": st.column_config.NumberColumn("Interruptions/min", format="%.2f"),
    "silence_pct": st.column_config.NumberColumn("Silence", format="%.1f%%"),
    "avg_pause_s": st.column_config.NumberColumn("Avg pause (s)", format="%.1f"),
    "median_response_s": st.column_config.NumberColumn("Median response (s)", format="%.1f"),
    "questions": "Questions",
    "quality_score": st.column_config.NumberColumn("Audio quality", format="%d"),
})

section("Speaking-time distribution within each recording", tag="measured")
fig = go.Figure()
max_n = max(len(r["ranked_shares_pct"]) for r in rows)
ramp = ["#3987E5", "#6FA5EA", "#9CC0EF", "#C3D8F4", "#DCE8F8", "#E8EFF8", "#F0F4FA", "#F5F7FB"]
rank_names = ["Largest share", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
for k in range(min(max_n, 8)):
    fig.add_bar(y=df["Recording"], x=[r["ranked_shares_pct"][k] if k < len(r["ranked_shares_pct"]) else 0 for r in rows],
                orientation="h", name=rank_names[k], marker=dict(color=ramp[k], line=dict(color="#0B1120", width=2)),
                hovertemplate="%{y}<br>" + rank_names[k] + " speaker: %{x:.1f}%<extra></extra>")
fig.update_layout(barmode="stack")
fig.update_xaxes(range=[0, 100], ticksuffix="%")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(style_fig(fig, 90 + 46 * len(rows)), width="stretch", theme=None, config=PLOTLY_CONFIG)


def metric_chart(col, key, color, fmt=".1f"):
    title, unit = comparison.TREND_METRICS[key]
    with col:
        section(title)
        f = go.Figure(go.Bar(x=df["Recording"], y=df[key].fillna(0), marker=dict(color=color, cornerradius=4),
                             hovertemplate="%{x}<br>%{y:" + fmt + "}" + unit + "<extra></extra>"))
        if unit:
            f.update_yaxes(ticksuffix=unit if unit == "%" else f" {unit}")
        st.plotly_chart(style_fig(f, 240, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)


pairs = [("duration_min", "#64748B"), ("balance", "#199E70"), ("avg_turn_s", "#9085E9"), ("turns_per_min", "#9085E9"),
         ("interruptions_per_min", "#D95926"), ("silence_pct", "#64748B"), ("avg_pause_s", "#C98500"),
         ("median_response_s", "#C98500")]
for i in range(0, len(pairs), 2):
    c1, c2 = st.columns(2, gap="large")
    metric_chart(c1, *pairs[i], fmt=".2f" if pairs[i][0] in ("balance", "interruptions_per_min") else ".1f")
    metric_chart(c2, *pairs[i + 1], fmt=".2f" if pairs[i + 1][0] in ("balance", "interruptions_per_min") else ".1f")

section("Language sentiment distribution", "Share of transcript segments per class (English recordings only).", tag="inferred")
sdf = df.dropna(subset=["positive_pct"])
if sdf.empty:
    st.caption("Sentiment is not available for the selected recordings.")
else:
    fig = go.Figure()
    for cat, col in (("negative_pct", "#D55181"), ("neutral_pct", "#64748B"), ("positive_pct", "#199E70")):
        nm = cat.split("_")[0].capitalize()
        fig.add_bar(y=sdf["Recording"], x=sdf[cat], name=nm, orientation="h",
                    marker=dict(color=col, line=dict(color="#0B1120", width=2)),
                    hovertemplate="%{y}<br>" + nm + ": %{x:.0f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 90 + 46 * len(sdf)), width="stretch", theme=None, config=PLOTLY_CONFIG)

# ---------------------------------------------------------------- trends (all recordings)
section("Historical trends", f"All {len(all_rows)} analysed recordings in chronological order.", tag="measured")
metric = st.selectbox("Metric", list(comparison.TREND_METRICS), format_func=lambda k: comparison.TREND_METRICS[k][0])
tr = comparison.trend(all_rows, metric)
if len(tr["points"]) < 2:
    st.caption("Not enough recordings with this metric yet.")
else:
    xs, ys, titles = zip(*tr["points"])
    unit = comparison.TREND_METRICS[metric][1]
    fig = go.Figure(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", line=dict(color="#3987E5", width=2),
                               marker=dict(size=9, color="#3987E5", line=dict(color="#0B1120", width=2)),
                               customdata=list(titles),
                               hovertemplate="<b>%{customdata}</b><br>%{x|%d %b %Y %H:%M}<br>%{y:.2f}" + unit + "<extra></extra>"))
    st.plotly_chart(style_fig(fig, 280, legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)
    if tr["change"] is not None:
        note(f"Second half of the history vs first half: {tr['change']:+.2f}{unit} on average. "
             "Recordings differ in purpose and participants, so treat trends as descriptive only.")
    else:
        note("At least four recordings are needed to summarise the direction of a trend.")
