"""Search transcripts across all analysed recordings and jump to the audio."""
from __future__ import annotations

from html import escape

import streamlit as st

import config
from core.transcript_search import highlight, parse_query, parse_timestamp
from database import repository
from ui.components import chip, completed_recordings, empty_state, page_header, section
from utils.time_formatter import format_timestamp

page_header("Analysis", "Transcript search",
            "Search every analysed recording by keyword, phrase, speaker or timestamp. Click a result to play it.")

recs = completed_recordings()
if not recs:
    empty_state("Nothing to search yet", "Analyse a recording first — its transcript becomes searchable here.", icon="⌕")
    st.stop()

with st.form("search", border=False):
    c1, c2, c3 = st.columns([4, 2, 1], vertical_alignment="bottom")
    query = c1.text_input("Search", placeholder='e.g.  budget   ·   "action item"   ·   1:23   ·   రేపు',
                          value=st.session_state.get("search_q", ""))
    speaker = c2.text_input("Speaker name contains", value=st.session_state.get("search_spk", ""),
                            placeholder="e.g. Priya")
    submitted = c3.form_submit_button("Search", type="primary", icon=":material/search:", width="stretch")
    rec_ids = st.multiselect("Recordings", [r["id"] for r in recs], format_func=lambda i: next(r["title"] for r in recs if r["id"] == i),
                             placeholder="All recordings")
st.caption("Words must all appear (any order). Quotes match an exact phrase. A timestamp like 1:23 finds the line being "
           "spoken at that moment in each selected recording.")

if submitted:
    st.session_state.search_q, st.session_state.search_spk = query, speaker

q = st.session_state.get("search_q", "")
spk = st.session_state.get("search_spk", "")
if not q and not spk:
    st.stop()


def play(rid: str, start: float, end: float | None) -> None:
    st.session_state.current_recording = rid
    st.session_state.seek = {"rid": rid, "start": max(0.0, start), "end": end, "autoplay": True}
    st.switch_page("views/conversation_timeline.py")


ts = parse_timestamp(q)
if ts is not None:
    section(f"Lines spoken at {format_timestamp(ts, True)}")
    from core.transcript_search import segment_at

    targets = rec_ids or [r["id"] for r in recs]
    found = 0
    for r in recs:
        if r["id"] not in targets:
            continue
        data = repository.load_analysis(r["id"])
        idx = segment_at(data["transcript"], ts)
        if idx is None:
            continue
        t = data["transcript"][idx]
        names = {s["label"]: s["display_name"] for s in data["speakers"]}
        if spk and spk.casefold() not in names.get(t.speaker, "").casefold():
            continue
        found += 1
        with st.container(border=True):
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.markdown(f"**{escape(r['title'])}** · {chip(names.get(t.speaker, t.speaker or 'Unknown'), config.speaker_color(int(t.speaker.split()[-1]) - 1 if t.speaker and t.speaker.split()[-1].isdigit() else 7))}"
                       f' <span class="tt-row-sub">{format_timestamp(t.start, True)} – {format_timestamp(t.end, True)}</span>'
                       f'<div class="tt-seg-text">{escape(t.text)}</div>', unsafe_allow_html=True)
            if b.button("Play", key=f"ts_{r['id']}", icon=":material/play_arrow:", width="stretch"):
                play(r["id"], ts, None)
    if not found:
        st.info("No speech at that time in the selected recordings.")
    st.stop()

hits = repository.search_transcripts(q, recording_ids=rec_ids or None, speaker_name=spk or None)
terms = parse_query(q)
section(f"{len(hits)} matching line{'s' if len(hits) != 1 else ''}" + (" (showing first 300)" if len(hits) >= 300 else ""))
if not hits:
    st.info("No matches. Try fewer words, check spelling, or remove the speaker filter.")
    st.stop()

by_rec: dict[str, list] = {}
for h in hits:
    by_rec.setdefault(h["recording_id"], []).append(h)
for rid, items in by_rec.items():
    with st.expander(f"{items[0]['title']} · {items[0]['created_at']:%d %b %Y} · {len(items)} match(es)", expanded=len(by_rec) <= 3):
        for i, h in enumerate(items):
            idx = int(h["speaker_label"].split()[-1]) - 1 if h["speaker_label"] and h["speaker_label"].split()[-1].isdigit() else 7
            color = config.speaker_color(idx)
            a, b = st.columns([1.3, 8], vertical_alignment="top")
            if a.button(format_timestamp(h["start"]), key=f"hit_{rid}_{i}", icon=":material/play_arrow:", width="stretch"):
                play(rid, h["start"], h["end"])
            flag = ' <span class="tt-uncertain">⚠ uncertain speaker</span>' if h["uncertain"] else ""
            b.markdown(f'<div class="tt-seg" style="border-left-color:{color}"><div class="tt-seg-head">'
                       f'{chip(h["speaker_name"], color)}{flag}</div><div class="tt-seg-text">{highlight(h["text"], terms)}</div></div>',
                       unsafe_allow_html=True)
