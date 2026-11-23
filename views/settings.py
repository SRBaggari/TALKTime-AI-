"""Settings: system status, models, processing, analytics, storage, Ollama, privacy."""
from __future__ import annotations

import os
from dataclasses import replace

import streamlit as st

import config
from core.conversation_summary import ollama_status
from core.diarization_engine import SETUP_INSTRUCTIONS
from database import repository
from services import model_manager, storage_service
from services.analysis_service import JobManager
from ui.components import badge, page_header, section, settings
from utils.validators import SUPPORTED_LANGUAGES, is_local_url

page_header("Workspace", "Settings", "Models, processing, privacy and data management. Everything here stays on this computer.")
s = settings()


def save(new: config.AppSettings) -> None:
    repository.save_app_settings(new)
    st.session_state.app_settings = new
    st.cache_data.clear()
    st.toast("Settings saved")


tabs = st.tabs(["System status", "Models", "Processing", "Analytics", "Storage & retention", "Local LLM", "Privacy & data"])

# ---------------------------------------------------------------- status
with tabs[0]:
    checks = model_manager.startup_checks(s)
    ready = model_manager.offline_ready(checks)
    if ready:
        st.success("Offline ready — all required components are installed locally. The app needs no internet connection.",
                   icon=":material/wifi_off:")
    else:
        st.warning("Setup incomplete — fix the items marked below. Missing components are explained, not faked.", icon=":material/build:")
    for c in checks:
        kind = "ok" if c.ok else ("bad" if c.required else "warn")
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"{badge('OK' if c.ok else ('Missing' if c.required else 'Optional'), kind)} &nbsp; **{c.name}**", unsafe_allow_html=True)
        c2.markdown(c.detail + (f"  \n`{c.fix}`" if not c.ok and c.fix else ""))
    st.caption("Privacy: Streamlit usage statistics, Hugging Face telemetry and pyannote telemetry are disabled. "
               "Recordings are never uploaded anywhere.")

# ---------------------------------------------------------------- models
with tabs[1]:
    status = model_manager.model_status(s.whisper_model)
    section("Transcription — Faster-Whisper")
    st.caption("Approximate download / RAM (int8, CPU): tiny 75 MB / ~0.5 GB · base 145 MB / ~0.7 GB · "
               "small 480 MB / ~1.2 GB (recommended for 16 GB RAM) · medium 1.5 GB / ~2.5 GB (slower, more accurate).")
    for size in config.WHISPER_SIZES:
        c1, c2, c3 = st.columns([1, 2, 1.2])
        c1.markdown(f"**{size}**" + (" · selected" if size == s.whisper_model else ""))
        c2.markdown(badge("Cached locally", "ok") if status["whisper"][size] else badge("Not downloaded", "warn"), unsafe_allow_html=True)
        if not status["whisper"][size]:
            if c3.button("Download", key=f"dl_{size}", width="stretch", help="One-time download from Hugging Face (internet needed)."):
                with st.spinner(f"Downloading faster-whisper-{size}…"):
                    try:
                        model_manager.download_whisper(size)
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Download failed ({exc.__class__.__name__}). Check your internet connection.")
    section("Speaker diarization — pyannote Community-1")
    if status["diarization"]:
        st.markdown(badge("Cached locally — works offline", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(badge("Not downloaded", "warn") + " &nbsp; " +
                    badge("HF token found" if status["hf_token"] else "HF_TOKEN not set", "ok" if status["hf_token"] else "bad"),
                    unsafe_allow_html=True)
        st.code(SETUP_INSTRUCTIONS, language=None)
        if status["hf_token"] and st.button("Download diarization model", type="primary"):
            with st.spinner("Downloading pyannote/speaker-diarization-community-1…"):
                try:
                    model_manager.download_diarization()
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    msg = str(exc)
                    if "403" in msg or "gated" in msg.lower() or "401" in msg:
                        st.error("Access denied — make sure you accepted the model conditions on Hugging Face with the "
                                 "same account that created the token.")
                    else:
                        st.error(f"Download failed ({exc.__class__.__name__}).")
    section("Memory")
    st.markdown(f"Loaded now: Whisper {'yes' if status['whisper_loaded'] else 'no'} · diarization {'yes' if status['diarization_loaded'] else 'no'}")
    if st.button("Release loaded models from memory", disabled=bool(JobManager.active())):
        model_manager.release_models()
        st.toast("Models released")
        st.rerun()

# ---------------------------------------------------------------- processing
with tabs[2]:
    with st.form("processing"):
        c1, c2 = st.columns(2)
        whisper_model = c1.selectbox("Whisper model", config.WHISPER_SIZES, index=config.WHISPER_SIZES.index(s.whisper_model))
        compute_type = c2.selectbox("Quantization", ["int8", "int8_float32", "float32"], index=["int8", "int8_float32", "float32"].index(s.compute_type)
                                    if s.compute_type in ("int8", "int8_float32", "float32") else 0,
                                    help="int8 is fastest and uses the least memory on CPU.")
        cpu_threads = c1.slider("CPU threads", 1, os.cpu_count() or 8, s.cpu_threads, help=f"This computer has {os.cpu_count()} logical cores.")
        beam_size = c2.slider("Beam size", 1, 5, s.beam_size, help="Lower is faster; 5 is most accurate.")
        langs = list(SUPPORTED_LANGUAGES)
        default_language = c1.selectbox("Default language", langs, index=langs.index(s.default_language) if s.default_language in langs else 0,
                                        format_func=SUPPORTED_LANGUAGES.get)
        c2.write("")
        vad_filter = c2.checkbox("Skip non-speech with VAD (faster, fewer hallucinations)", s.vad_filter)
        word_timestamps = c2.checkbox("Word timestamps (word-level speaker alignment)", s.word_timestamps)
        keep_models_loaded = c2.checkbox("Keep models in memory between analyses", s.keep_models_loaded,
                                         help="Faster repeated analyses; uncheck to free RAM after each run.")
        section("Speaker configuration")
        c1, c2 = st.columns(2)
        min_speakers = c1.number_input("Default minimum speakers (0 = auto)", 0, 20, s.min_speakers)
        max_speakers = c2.number_input("Default maximum speakers (0 = auto)", 0, 20, s.max_speakers)
        if st.form_submit_button("Save processing settings", type="primary"):
            if min_speakers and max_speakers and min_speakers > max_speakers:
                st.error("Minimum speakers cannot exceed maximum.")
            else:
                save(replace(s, whisper_model=whisper_model, compute_type=compute_type, cpu_threads=int(cpu_threads),
                             beam_size=int(beam_size), default_language=default_language, vad_filter=vad_filter,
                             word_timestamps=word_timestamps, keep_models_loaded=keep_models_loaded,
                             min_speakers=int(min_speakers), max_speakers=int(max_speakers)))

# ---------------------------------------------------------------- analytics
with tabs[3]:
    st.caption("Thresholds apply to new analyses and when analytics are recalculated after corrections.")
    with st.form("analytics"):
        c1, c2 = st.columns(2)
        min_silence = c1.number_input("Minimum silence (s)", 0.2, 10.0, s.min_silence, 0.1, help="Shorter gaps are normal pauses.")
        long_silence = c2.number_input("Long silence (s)", 1.0, 60.0, s.long_silence, 0.5)
        min_overlap = c1.number_input("Minimum overlap to count (s)", 0.05, 3.0, s.min_overlap, 0.05)
        turn_merge_gap = c2.number_input("Join same-speaker pauses up to (s)", 0.0, 3.0, s.turn_merge_gap, 0.1)
        uncertainty = c1.slider("Speaker-assignment confidence threshold", 0.5, 0.95, s.uncertainty_threshold, 0.05)
        qa_window = c2.number_input("Question response window (s)", 5.0, 120.0, s.question_answer_window, 5.0)
        if st.form_submit_button("Save analytics settings", type="primary"):
            if long_silence < min_silence:
                st.error("Long silence must be at least the minimum silence.")
            else:
                save(replace(s, min_silence=min_silence, long_silence=long_silence, min_overlap=min_overlap,
                             turn_merge_gap=turn_merge_gap, uncertainty_threshold=uncertainty, question_answer_window=qa_window))

# ---------------------------------------------------------------- storage
with tabs[4]:
    usage = storage_service.storage_usage()
    c = st.columns(5)
    for col, (k, v) in zip(c, [("Uploads", usage["uploads_mb"]), ("Processed audio", usage["processed_mb"]),
                               ("Reports", usage["reports_mb"]), ("Database", usage["database_mb"]), ("Free disk", usage["free_mb"])]):
        col.metric(k, f"{v / 1024:.2f} GB" if v > 1024 else f"{v:.1f} MB")
    st.caption(f"Data folder: {config.DATA_DIR}")
    with st.form("storage"):
        keep_original = st.checkbox("Keep original uploaded audio after analysis", s.keep_original_audio,
                                    help="The processed 16 kHz WAV used for playback is always kept until you delete the recording.")
        retention = st.number_input("Automatically delete recordings older than (days, 0 = never)", 0, 3650, s.retention_days)
        max_mb = st.number_input("Maximum upload size (MB)", 10, 2000, s.max_upload_mb)
        if st.form_submit_button("Save storage settings", type="primary"):
            save(replace(s, keep_original_audio=keep_original, retention_days=int(retention), max_upload_mb=int(max_mb)))
    if s.retention_days and st.button(f"Apply retention now (delete recordings older than {s.retention_days} days)"):
        n = storage_service.apply_retention(s.retention_days)
        st.success(f"Deleted {n} recording(s).")

# ---------------------------------------------------------------- ollama
with tabs[5]:
    st.caption("Optional. When enabled, a model running in your local Ollama installation can write an abstractive summary. "
               "Only localhost URLs are accepted. Core features never depend on it.")
    with st.form("ollama"):
        enabled = st.toggle("Enable Ollama summaries", s.ollama_enabled)
        url = st.text_input("Ollama URL", s.ollama_url)
        model = st.text_input("Model name", s.ollama_model, help="e.g. llama3.2, qwen2.5:7b, mistral")
        if st.form_submit_button("Save", type="primary"):
            if not is_local_url(url):
                st.error("Only local URLs (localhost / 127.0.0.1) are allowed.")
            else:
                save(replace(s, ollama_enabled=enabled, ollama_url=url.strip(), ollama_model=model.strip()))
    if st.button("Test connection"):
        r = ollama_status(s.ollama_url)
        if r["available"]:
            st.success("Connected. Installed models: " + (", ".join(r["models"]) or "none"))
            if s.ollama_model not in r["models"] and f"{s.ollama_model}:latest" not in r["models"]:
                st.warning(f"Model '{s.ollama_model}' is not installed. Run: ollama pull {s.ollama_model}")
        else:
            st.error(r["reason"])

# ---------------------------------------------------------------- privacy
@st.dialog("Confirm permanent deletion")
def confirm_delete(action: str, rid: str | None, title: str) -> None:
    texts = {
        "recording": "the recording, its transcript, speakers, analytics, original and processed audio, and all reports",
        "analysis": "the transcript, speakers, analytics and reports (the audio is kept so it can be re-analysed)",
        "reports": "all generated report files (PDF) for this recording",
        "original": "the original uploaded file (the processed audio used for playback is kept)",
        "all": "EVERY recording, transcript, analysis and report stored by TalkTime AI",
    }
    st.markdown(f"This permanently deletes **{texts[action]}**" + (f" for **{title}**." if rid else "."))
    st.caption("Deleted data cannot be recovered.")
    phrase = "DELETE ALL" if action == "all" else "DELETE"
    typed = st.text_input(f"Type {phrase} to confirm")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Delete permanently", type="primary", width="stretch", disabled=typed != phrase):
        if action == "recording":
            storage_service.delete_recording(rid)
        elif action == "analysis":
            storage_service.delete_transcripts_and_analysis(rid)
        elif action == "reports":
            storage_service.delete_reports(rid)
        elif action == "original":
            storage_service.delete_original_audio(rid)
        elif action == "all":
            storage_service.delete_all_data()
        if action in ("recording", "analysis", "all"):
            if action == "all" or st.session_state.get("current_recording") == rid:
                st.session_state.pop("current_recording", None)
            st.session_state.pop("active_job", None)
            st.session_state.pop("seek", None)
        st.session_state.privacy_msg = "Deleted."
        st.rerun()


with tabs[6]:
    st.markdown(badge("Local processing only", "ok") + " &nbsp; " + badge("No telemetry", "ok") + " &nbsp; " +
                badge("No cloud services", "ok"), unsafe_allow_html=True)
    st.caption("Audio is decoded, transcribed, diarized and analysed on this computer. Optional Ollama summaries are "
               "restricted to localhost. Nothing is uploaded anywhere.")
    if st.session_state.pop("privacy_msg", None):
        st.success("Deleted.", icon=":material/delete:")

    section("Credentials")
    token = config.get_hf_token()
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(f"Hugging Face token: `{config.mask_token(token)}`  \n"
                "Stored only in the local, git-ignored `.env` file. It is needed once to download the diarization "
                "model; analysis never uses it.")
    from core.diarization_engine import is_diarization_available

    if token and c2.button("Remove token", width="stretch", disabled=not is_diarization_available(),
                           help="Available once the diarization model is downloaded."):
        removed = config.remove_hf_token_from_env_file()
        st.success("Token removed from .env." if removed else "Token was not found in .env (it may be set as a system "
                   "environment variable instead).")

    section("Manage a recording")
    recs = repository.list_recordings()
    if recs:
        target = st.selectbox("Recording", [r["id"] for r in recs], key="privacy_rec",
                              format_func=lambda i: next(f"{r['title']} · {r['status']} · {r['created_at']:%d %b %Y %H:%M}"
                                                         for r in recs if r["id"] == i))
        rec = next(r for r in recs if r["id"] == target)
        busy = JobManager.active() == target
        n_reports = len(repository.list_reports(target))
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Delete recording", type="primary", width="stretch", disabled=busy, icon=":material/delete_forever:"):
            confirm_delete("recording", target, rec["title"])
        if c2.button("Delete transcript & analysis", width="stretch", disabled=busy or rec["status"] != "completed"):
            confirm_delete("analysis", target, rec["title"])
        if c3.button(f"Delete reports ({n_reports})", width="stretch", disabled=busy or not n_reports):
            confirm_delete("reports", target, rec["title"])
        if c4.button("Delete original upload", width="stretch", disabled=busy or not rec["stored_path"]):
            confirm_delete("original", target, rec["title"])
        if busy:
            st.caption("This recording is being analysed; deletion is available when it finishes.")
    else:
        st.caption("No recordings stored.")

    section("Delete all local analysis data", "Removes every recording, transcript, analysis and report. "
            "Settings and downloaded models are kept.")
    if st.button("Delete all data…", disabled=bool(JobManager.active()) or not recs, icon=":material/warning:"):
        confirm_delete("all", None, "")
