"""Reusable UI building blocks. All user-provided strings are HTML-escaped."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

import config
from database import repository
from utils.time_formatter import format_duration

CHART_SURFACE = "rgba(0,0,0,0)"
GRID = "#1F2A3D"
AXIS_TEXT = "#94A3B8"
TEXT = "#E5E7EB"


@st.cache_data(show_spinner=False)
def _css() -> str:
    return (config.ASSETS_DIR / "styles.css").read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def settings() -> config.AppSettings:
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = repository.load_app_settings()
    return st.session_state.app_settings


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="tt-header"><div class="tt-eyebrow">{escape(eyebrow)}</div>'
        f'<div class="tt-title">{escape(title)}</div><div class="tt-subtitle">{escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    bar = f'<div class="tt-accent-bar" style="background:{accent}"></div>' if accent else ""
    sub_html = f'<div class="tt-metric-sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="tt-card">{bar}<div class="tt-metric-label">{escape(label)}</div>'
            f'<div class="tt-metric-value">{escape(value)}</div>{sub_html}</div>')


def metrics_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(metric_card(*item), unsafe_allow_html=True)


def section(title: str, caption: str = "", tag: str | None = None) -> None:
    tag_html = ""
    if tag == "measured":
        tag_html = '<span class="tt-tag tt-tag-measured">Measured from audio</span>'
    elif tag == "inferred":
        tag_html = '<span class="tt-tag tt-tag-inferred">Inferred · NLP estimate</span>'
    elif tag == "optional":
        tag_html = '<span class="tt-tag tt-tag-optional">Optional · local LLM</span>'
    st.markdown(f'<div class="tt-section"><h3>{escape(title)}</h3>{tag_html}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="tt-section-caption">{escape(caption)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="tt-note">{escape(text)}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    icon = {"ok": "✓", "warn": "!", "bad": "✕", "info": "•"}.get(kind, "•")
    return f'<span class="tt-badge tt-badge-{kind}">{icon} {escape(text)}</span>'


def chip(name: str, color: str) -> str:
    return f'<span class="tt-chip"><span class="tt-dot" style="background:{color}"></span>{escape(name)}</span>'


def empty_state(title: str, text: str, icon: str = "◎", steps: list[tuple[str, str]] | None = None) -> None:
    steps_html = ""
    if steps:
        steps_html = '<div class="tt-steps">' + "".join(
            f'<div class="tt-step"><div class="tt-step-n">STEP {i}</div><b>{escape(a)}</b><span>{escape(b)}</span></div>'
            for i, (a, b) in enumerate(steps, 1)) + "</div>"
    st.markdown(f'<div class="tt-empty"><div class="tt-empty-icon">{icon}</div><h3>{escape(title)}</h3>'
                f'<p>{escape(text)}</p>{steps_html}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    return badge(status.capitalize(), {"completed": "ok", "failed": "bad", "processing": "info", "pending": "warn"}.get(status, "info"))


# --------------------------------------------------------------------------- recordings

def completed_recordings() -> list[dict]:
    return repository.list_recordings(status="completed")


def recording_picker(key: str = "rec_picker") -> str | None:
    """Selectbox of analysed recordings; the selection is shared across pages."""
    recs = completed_recordings()
    if not recs:
        empty_state("No analysed recordings yet",
                    "Upload a recording to see speaker analytics, timelines and conversation insights here.",
                    icon="♪")
        if st.button("Upload a recording", type="primary", icon=":material/upload:"):
            st.switch_page("pages/upload_audio.py")
        return None
    ids = [r["id"] for r in recs]
    current = st.session_state.get("current_recording")
    index = ids.index(current) if current in ids else 0
    labels = {r["id"]: f"{r['title']}  ·  {format_duration(r['duration'])}  ·  {r['created_at']:%d %b %Y %H:%M}" for r in recs}
    rid = st.selectbox("Recording", ids, index=index, format_func=labels.get, key=key, label_visibility="collapsed")
    st.session_state.current_recording = rid
    return rid


def load(recording_id: str) -> dict | None:
    return repository.load_analysis(recording_id)


def names_and_colors(data: dict) -> tuple[dict, dict]:
    names = {s["label"]: s["display_name"] for s in data["speakers"]}
    colors = {s["label"]: s["color"] for s in data["speakers"]}
    colors.setdefault("Unknown", "#64748B")
    names.setdefault("Unknown", "Unknown")
    return names, colors


# --------------------------------------------------------------------------- charts

def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=CHART_SURFACE, plot_bgcolor=CHART_SURFACE,
        font=dict(family="Segoe UI, Inter, system-ui, sans-serif", color=TEXT, size=12.5),
        margin=dict(l=8, r=8, t=12, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=AXIS_TEXT)),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#2B3850", font=dict(color=TEXT, size=12.5)),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=AXIS_TEXT), title_font=dict(color=AXIS_TEXT))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=AXIS_TEXT), title_font=dict(color=AXIS_TEXT))
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "toImageButtonOptions": {"scale": 2}}
