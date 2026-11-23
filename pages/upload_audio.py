"""Upload a recording and run the analysis in the background."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import streamlit as st

import config
from core.diarization_engine import SETUP_INSTRUCTIONS, is_diarization_available
from core.transcription_engine import is_whisper_available
from database import repository
from services import storage_service
from services.analysis_service import AnalysisError, AnalysisOptions, JobManager
from ui.components import badge, page_header, section, settings
from utils.time_formatter import format_duration
from utils.validators import SUPPORTED_LANGUAGES, validate_speaker_bounds

page_header("Analyse", "Upload recording",
            "Recordings are stored and processed on this computer only. Processing starts only when you press Start.")
s = settings()


@st.fragment(run_every=1.0)
def job_monitor(recording_id: str) -> None:
    job = JobManager.status(recording_id)
    rec = repository.get_recording(recording_id)
    if job is None:  # app restarted mid-job, or finished in a previous session
        if rec and rec["status"] == "completed":
            st.success("Analysis complete.")
        elif rec and rec["status"] == "failed":
            st.error(rec["error"] or "Analysis failed.")
        else:
            st.info("This analysis is no longer running (the app may have restarted). Retry below.")
        return
    elapsed = (job["finished"] or time.time()) - job["started"]
    title = rec["title"] if rec else "Recording"
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if job["state"] == "running":
            st.progress(job["progress"], text=f"{job['message']}  ·  {job['progress'] * 100:.0f}%")
            st.caption(f"Elapsed {format_duration(elapsed)}. You can switch pages; analysis continues in the background. "
                       "CPU processing of long recordings can take several minutes.")
        elif job["state"] == "completed":
            st.progress(1.0, text="Analysis complete")
            st.success(f"Finished in {format_duration(elapsed)} "
                       f"({format_duration(rec['duration'])} of audio, {rec['num_speakers']} speaker(s) detected).")
            c1, c2, c3 = st.columns(3)
            if c1.button("Speaker analysis", type="primary", use_container_width=True, icon=":material/record_voice_over:"):
                st.session_state.current_recording = recording_id
                st.switch_page("pages/speaker_analysis.py")
            if c2.button("Timeline & transcript", use_container_width=True, icon=":material/view_timeline:"):
                st.session_state.current_recording = recording_id
                st.switch_page("pages/conversation_timeline.py")
            if c3.button("Analyse another", use_container_width=True, icon=":material/add:"):
                st.session_state.pop("active_job", None)
                st.rerun(scope="app")
        else:
            st.error("Analysis failed.")
            if job.get("model_missing"):
                st.code(job["error"], language=None)
            else:
                st.write(job["error"])
            c1, c2 = st.columns(2)
            if c1.button("Retry analysis", type="primary", use_container_width=True, icon=":material/refresh:"):
                try:
                    JobManager.start(recording_id, st.session_state.get("job_options", AnalysisOptions(settings=s)))
                except AnalysisError as exc:
                    st.warning(str(exc))
                st.rerun(scope="fragment")
            if c2.button("Discard recording", use_container_width=True, icon=":material/delete:"):
                storage_service.delete_recording(recording_id)
                st.session_state.pop("active_job", None)
                st.rerun(scope="app")


# Resume monitoring of a running/finished job in this session.
active = st.session_state.get("active_job") or JobManager.active()
if active:
    st.session_state.active_job = active
    job_monitor(active)
    st.stop()

# ---------------------------------------------------------------- model readiness
whisper_ok = is_whisper_available(s.whisper_model)
diar_ok = is_diarization_available()
section("Model readiness", "Checked against the local model cache — no network access.")
c1, c2, c3 = st.columns(3)
c1.markdown(badge(f"Whisper {s.whisper_model} ready" if whisper_ok else f"Whisper {s.whisper_model} missing",
                  "ok" if whisper_ok else "bad"), unsafe_allow_html=True)
c2.markdown(badge("Speaker diarization ready" if diar_ok else "Diarization model missing", "ok" if diar_ok else "warn"),
            unsafe_allow_html=True)
c3.markdown(badge("FFmpeg found" if __import__("shutil").which("ffmpeg") else "FFmpeg missing (WAV/FLAC only)",
                  "ok" if __import__("shutil").which("ffmpeg") else "warn"), unsafe_allow_html=True)
if not whisper_ok:
    st.error(f"The Whisper '{s.whisper_model}' model is not downloaded. Go to Settings → Models to download it "
             "(one-time, needs internet) or choose another size that is already cached.")
if not diar_ok:
    with st.expander("Speaker diarization is not set up — how to enable it", icon=":material/info:"):
        st.code(SETUP_INSTRUCTIONS, language=None)
        st.caption("Until then you can analyse recordings in single-speaker mode (all speech attributed to one speaker).")

# ---------------------------------------------------------------- form
section("Recording")
uploaded = st.file_uploader("Drag and drop an audio file", type=sorted(e.lstrip(".") for e in config.ALLOWED_EXTENSIONS),
                            help=f"Maximum {s.max_upload_mb} MB. The file is decoded locally with FFmpeg; it is never executed.")

c1, c2 = st.columns(2)
default_title = Path(uploaded.name).stem.replace("_", " ")[:80] if uploaded else ""
title = c1.text_input("Title", value=default_title, max_chars=200, placeholder="e.g. Customer onboarding call")
lang_codes = list(SUPPORTED_LANGUAGES)
language = c2.selectbox("Language", lang_codes, index=lang_codes.index(s.default_language) if s.default_language in lang_codes else 0,
                        format_func=SUPPORTED_LANGUAGES.get, help="Auto-detect works well; fixing the language avoids misdetection on short clips.")
description = st.text_area("Notes (optional)", max_chars=1000, height=70)

section("Speakers")
modes = ["pyannote", "single"]
mode = st.radio("Diarization", modes, horizontal=True, index=0 if diar_ok else 1,
                format_func={"pyannote": "Detect speakers automatically (pyannote)",
                             "single": "Single speaker (no diarization)"}.get,
                disabled=not diar_ok)
count_mode, num, lo, hi = "auto", None, None, None
if mode == "pyannote":
    c1, c2, c3 = st.columns([2, 1, 1])
    count_mode = c1.radio("Expected number of speakers", ["auto", "exact", "range"], horizontal=True,
                          format_func={"auto": "Automatic", "exact": "Exact", "range": "Range"}.get)
    if count_mode == "exact":
        num = c2.number_input("Speakers", 1, 20, 2)
    elif count_mode == "range":
        lo = c2.number_input("Minimum", 1, 20, max(1, s.min_speakers or 1))
        hi = c3.number_input("Maximum", 1, 20, max(2, s.max_speakers or 4))
    elif s.min_speakers or s.max_speakers:
        lo, hi = s.min_speakers or None, s.max_speakers or None
        st.caption(f"Using default bounds from Settings: min {lo or 'auto'}, max {hi or 'auto'}.")

ready = uploaded is not None and whisper_ok and (mode == "single" or diar_ok)
st.write("")
start = st.button("Start analysis", type="primary", icon=":material/play_arrow:", disabled=not ready)

if start and uploaded is not None:
    try:
        lo_v, hi_v = validate_speaker_bounds(lo, hi)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    data = uploaded.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get("last_submitted") == digest:
        st.warning("This file was just submitted. Open the analysis pages or upload a different file.")
        st.stop()
    try:
        path, result = storage_service.save_upload(uploaded.name, data, s.max_upload_mb)
    except storage_service.StorageError as exc:
        st.error(str(exc))
        st.stop()
    if not result.ok:
        for e in result.errors:
            st.error(e)
        st.stop()
    for w in result.warnings:
        st.warning(w)
    rid = repository.create_recording(
        title=(title or Path(uploaded.name).stem)[:200], description=description or None,
        original_filename=Path(uploaded.name).name[:255], stored_path=str(path), file_size=len(data),
        diarization_mode=mode, status="pending")
    options = AnalysisOptions(language=None if language == "auto" else language, num_speakers=int(num) if num else None,
                              min_speakers=lo_v, max_speakers=hi_v, diarization_mode=mode, settings=s)
    try:
        JobManager.start(rid, options)
    except AnalysisError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state.last_submitted = digest
    st.session_state.active_job = rid
    st.session_state.job_options = options
    st.rerun()

# ---------------------------------------------------------------- failed recordings
failed = repository.list_recordings(status="failed")
if failed:
    section("Failed analyses", "Retry with current settings or delete.")
    for r in failed[:5]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f"**{r['title']}**")
            c1.caption((r["error"] or "Unknown error").splitlines()[0][:200])
            if c2.button("Retry", key=f"retry_{r['id']}", use_container_width=True, disabled=not r["stored_path"]):
                try:
                    opts = AnalysisOptions(diarization_mode=r["diarization_mode"] if diar_ok else "single", settings=s)
                    JobManager.start(r["id"], opts)
                    st.session_state.active_job = r["id"]
                    st.session_state.job_options = opts
                    st.rerun()
                except AnalysisError as exc:
                    st.warning(str(exc))
            if c3.button("Delete", key=f"del_{r['id']}", use_container_width=True):
                storage_service.delete_recording(r["id"])
                st.rerun()
