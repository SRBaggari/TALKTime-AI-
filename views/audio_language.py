"""Audio quality, voice activity detection and language identification."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.language_utils import language_name
from ui.components import (PLOTLY_CONFIG, badge, load, metric_card, metrics_row, names_and_colors, note, page_header,
                           recording_picker, section, style_fig)
from utils.time_formatter import format_duration, format_timestamp

page_header("Analysis", "Audio & language",
            "Recording quality, where speech occurs, and which languages were spoken — measured locally.")
rid = recording_picker()
if not rid:
    st.stop()
data = load(rid)
m, rec = data["metrics"], data["recording"]
names, colors = names_and_colors(data)
q, vad, lang = m.get("quality"), m.get("vad"), m.get("language") or {}

if not q and not vad:
    st.info("This recording was analysed before audio-quality and VAD analysis existed. Re-upload it to see these measurements.")

# ---------------------------------------------------------------- quality
if q:
    section("Audio quality", q.get("method", ""), tag="measured")
    score_kind = "ok" if q["score"] >= 70 else "warn" if q["score"] >= 50 else "bad"
    fmt = q["format"]
    c1, c2 = st.columns([1, 3], gap="medium")
    c1.markdown(
        f'<div class="tt-card"><div class="tt-metric-label">Quality score</div>'
        f'<div class="tt-metric-value">{q["score"]}<span class="tt-metric-sub"> / 100</span></div>'
        f'<div style="margin-top:8px">{badge(q["label"], score_kind)}</div></div>', unsafe_allow_html=True)
    with c2:
        fmt_bits = [f"{fmt['sample_rate']:,} Hz" if fmt.get("sample_rate") else None,
                    {1: "mono", 2: "stereo"}.get(fmt.get("channels"), f"{fmt.get('channels')} channels") if fmt.get("channels") else None,
                    (fmt.get("codec") or "").upper() or None,
                    f"{fmt['bit_rate'] // 1000} kbps" if fmt.get("bit_rate") else None]
        metrics_row([
            ("Speech level", "—" if q["speech_level_dbfs"] is None else f"{q['speech_level_dbfs']:.0f} dBFS", "RMS during speech", None),
            ("Noise floor", f"{q['noise_floor_dbfs']:.0f} dBFS", "RMS without speech", None),
            ("Signal-to-noise", "—" if q["snr_db"] is None else f"{q['snr_db']:.0f} dB", "Estimate", None),
            ("Clipping", f"{q['clipping_ratio'] * 100:.3f}%", f"{q['clipping_events']} events", None),
        ])
        st.caption("Original format: " + " · ".join(b for b in fmt_bits if b) +
                   f" · peak {q['peak_dbfs']:.1f} dBFS" +
                   (f" · speech bandwidth ≈ {q['bandwidth_hz'] / 1000:.1f} kHz" if q.get("bandwidth_hz") else ""))

    section("Potential processing issues and recommendations")
    if not q["issues"]:
        st.success("No significant recording problems detected.", icon=":material/check_circle:")
    for i in q["issues"]:
        kind = {"serious": "bad", "warning": "warn"}.get(i["severity"], "info")
        with st.container(border=True):
            st.markdown(f'{badge(i["severity"].capitalize(), kind)} &nbsp; **{escape(i["message"])}**', unsafe_allow_html=True)
            st.caption(f"{i['recommendation']}  (score −{i['penalty']})")
    note("Low volume is judged on speech regions only, so long silences never count as 'quiet speech'; "
         "background sound in gaps is reported separately and is never treated as speech.")

# ---------------------------------------------------------------- VAD
if vad:
    section("Voice activity", f"Speech vs non-speech using {'Silero VAD (local neural model)' if vad['method'] == 'silero' else 'adaptive energy detection'}. "
            "Timestamps are on the original recording timeline.", tag="measured")
    cmp_ = vad.get("comparison") or {}
    sil = m.get("silence", {})
    metrics_row([
        ("Speech (VAD)", format_duration(vad["speech_duration"]), f"{vad['speech_ratio'] * 100:.0f}% of recording", None),
        ("VAD ↔ speaker agreement", f"{cmp_.get('agreement', 0) * 100:.0f}%", "Overlap of VAD and diarized speech", None),
        ("Speech without a speaker", format_duration(cmp_.get("unattributed_total")), "Heard by VAD, not diarized", None),
        ("Background sound in gaps", format_duration(sil.get("background_sound_total")), "Non-speech sound, not silence", None),
    ])
    duration = rec["duration"] or 1
    fig = go.Figure()
    lanes = []

    def lane(name, spans, color, hover, pattern=None):
        if not spans:
            return
        lanes.append(name)
        fig.add_bar(y=[name] * len(spans), x=[e - s for s, e in spans], base=[s for s, _ in spans], orientation="h",
                    name=name, marker=dict(color=color, pattern=pattern or {}),
                    customdata=[[format_timestamp(s, True), format_timestamp(e, True)] for s, e in spans],
                    hovertemplate=f"<b>{hover}</b><br>%{{customdata[0]}} → %{{customdata[1]}}<extra></extra>")

    lane("Speech (VAD)", vad["speech"], "#3987E5", "Speech detected by VAD")
    lane("Speakers", [(d.start, d.end) for d in data["diarization"]], "#199E70", "Diarized speech")
    quiet = [(x["start"], x["end"]) for x in sil.get("silences", []) if x.get("acoustic") == "quiet"]
    noisy = [(x["start"], x["end"]) for x in sil.get("silences", []) if x.get("acoustic") == "background_sound"]
    lane("Quiet gaps", quiet, "#475569", "Silence (quiet)")
    lane("Background sound", noisy, "#C98500", "Non-speech sound", dict(shape="/", fgcolor="#0B1120", size=5))
    lane("Speech without speaker", cmp_.get("unattributed_speech", []), "#D55181", "VAD speech not assigned to a speaker")
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(lanes)))
    tv = [duration * k / 8 for k in range(9)]
    fig.update_xaxes(range=[0, duration], tickvals=tv, ticktext=[format_timestamp(t) for t in tv])
    st.plotly_chart(style_fig(fig, 80 + 44 * len(lanes)), width="stretch", theme=None, config=PLOTLY_CONFIG)
    p = vad.get("params", {})
    note(f"Pauses shorter than {p.get('min_silence_ms', 300)} ms stay inside a speech region (normal pauses between words). "
         "Transcription also uses VAD to skip non-speech audio.")

# ---------------------------------------------------------------- language
section("Language identification", lang.get("method", ""), tag="measured")
if not lang:
    st.caption("No language information stored for this recording.")
else:
    left, right = st.columns([2, 3], gap="large")
    with left:
        prob = rec.get("language_probability")
        st.markdown(metric_card("Detected language", lang.get("detected_name") or "Unknown",
                                f"{prob * 100:.0f}% confidence" if prob else "", None), unsafe_allow_html=True)
        sel = lang.get("selected")
        st.write("")
        st.markdown(badge("Code-switching detected" if lang.get("code_switching") else "Single language", "info"),
                    unsafe_allow_html=True)
        st.caption(("Language was set manually" if sel and sel != "auto" else "Automatic detection") +
                   (" · per-segment (multilingual) mode" if lang.get("multilingual_mode") else ""))
        if lang.get("model_advice"):
            st.warning(lang["model_advice"], icon=":material/tips_and_updates:")
        if lang.get("transliterated_segments"):
            st.caption(f"{lang['transliterated_segments']} line(s) were written by Whisper in Devanagari and converted to "
                       "the native script (original text kept).")
        if prob is not None and prob < 0.6:
            st.warning("Language confidence is low. If the transcript looks wrong, re-run with the language set manually.")
    with right:
        tp = lang.get("top_probabilities") or []
        if tp:
            fig = go.Figure(go.Bar(y=[language_name(c) for c, _ in tp], x=[p * 100 for _, p in tp], orientation="h",
                                   marker=dict(color="#3987E5", cornerradius=4),
                                   hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Whisper language probability")
            st.plotly_chart(style_fig(fig, 60 + 34 * len(tp), legend=False), width="stretch", theme=None, config=PLOTLY_CONFIG)
    spoken = lang.get("languages_spoken") or {}
    if len(spoken) > 1 or lang.get("multilingual_mode"):
        st.markdown("**Languages spoken** (seconds of transcribed speech per language, detected per speech region)")
        fig = go.Figure(go.Bar(y=[v["name"] for v in spoken.values()], x=[v["share"] * 100 for v in spoken.values()],
                               orientation="h", marker=dict(color="#199E70", cornerradius=4),
                               customdata=[v["seconds"] for v in spoken.values()],
                               hovertemplate="%{y}: %{x:.0f}% (%{customdata:.0f}s)<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(style_fig(fig, 60 + 34 * len(spoken), legend=False), width="stretch", theme=None,
                        config=PLOTLY_CONFIG)
    mix = lang.get("script_mix") or {}
    if mix:
        st.markdown("**Writing scripts in the transcript** (each word's Unicode script)")
        st.dataframe(pd.DataFrame([{"Script": k, "Likely language(s)": lang.get("script_languages", {}).get(k, k),
                                    "Share of words": f"{v * 100:.1f}%"} for k, v in mix.items()]),
                     hide_index=True, width="stretch")
        st.caption(f"{lang.get('code_switch_segments', 0)} transcript line(s) mix scripts. Languages sharing a script "
                   "(e.g. Hindi/Marathi, English/Spanish) cannot be separated this way.")
    fs = lang.get("feature_support") or {}
    if fs:
        labels = {"transcription": "Transcription", "speaker_analytics": "Speaker & timing analytics",
                  "questions_punctuation": "Questions (from '?')", "questions_grammar": "Questions (grammar patterns)",
                  "sentiment": "Sentiment", "topics_spacy": "Topic extraction (spaCy)",
                  "insights_rules": "Decisions / action items"}
        st.markdown("**Feature availability for this language**")
        cols = st.columns(4)
        for i, (k, v) in enumerate(fs.items()):
            cols[i % 4].markdown(badge(labels.get(k, k), "ok" if v else "warn"), unsafe_allow_html=True)
        if not all(fs.values()):
            note("Unavailable features are English-only components. They are switched off rather than producing "
                 "unreliable results; keyword topics, questions marked with '?' and all audio analytics still work.")
