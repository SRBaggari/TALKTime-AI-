"""End-to-end analysis pipeline and a single-slot background job runner.

Pipeline: probe → decode → VAD → quality → transcribe → diarize → align → analytics → NLP → save.
ML inference runs once per recording; manual corrections only re-run the cheap
deterministic analytics (``recompute``).
"""
from __future__ import annotations

import gc
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

import config
from core import intervals as iv
from core.audio_loader import load_audio, probe_audio
from core.audio_quality import analyze_quality
from core.audio_preprocessor import is_effectively_silent, preprocess, write_wav
from core.conversation_summary import (build_summary, extract_insights, extract_topics, key_points, ollama_summarize,
                                       topic_recurrence)
from core.diarization_engine import DiarizationEngine, ModelUnavailableError
from core.interruption_analysis import analyze_interruptions
from core.language_utils import feature_support, summarize_languages
from core.overlap_detection import detect_overlaps
from core.participation_analysis import analyze_participation
from core.question_analysis import analyze_questions
from core.sentiment_analysis import analyze_sentiment
from core.silence_detection import acoustic_silence, analyze_silence
from core.speaker_alignment import align
from core.speaking_time import build_turns, compute_speaking_time
from core.transcription_engine import TranscriptionEngine
from core.types import SpeakerSegment, TranscriptSegment
from core.vad import compare_with_diarization, detect_speech
from database import repository
from services import storage_service
from services.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    pass


@dataclass
class AnalysisOptions:
    language: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    diarization_mode: str = "pyannote"  # "pyannote" | "single"
    multilingual: bool = False  # per-segment language detection (code-switched speech)
    settings: config.AppSettings = field(default_factory=config.AppSettings)


# --------------------------------------------------------------------------- analytics (pure, no ML)

def _speaker_confidence(transcript: list[TranscriptSegment]) -> dict:
    """Per-speaker assignment confidence (duration-weighted mean of alignment confidence).

    This is the *alignment* confidence (share of diarized speech under a line that
    belongs to the assigned speaker). pyannote does not output a per-segment
    probability, so none is shown or invented.
    """
    acc: dict[str, list] = {}
    for t in transcript:
        a = acc.setdefault(t.speaker or "Unknown", [0.0, 0.0, 0, 0])
        w = max(t.duration, 0.1)
        a[0] += t.speaker_confidence * w
        a[1] += w
        a[2] += 1
        a[3] += int(t.uncertain)
    return {k: {"mean_confidence": round(v[0] / v[1], 3) if v[1] else 0.0, "segments": v[2], "uncertain": v[3]}
            for k, v in acc.items()}


def _annotate_silence_acoustics(silence: dict, audio, sr: int) -> None:
    """Mark each speaker-free gap as truly quiet or containing background sound."""
    from core.audio_preprocessor import rms_envelope
    from core.vad import classify_non_speech

    db, _ = rms_envelope(audio, sr, 0.05)
    if not db.size:
        return
    floor = float(np.percentile(db, 10))
    regions = [(x["start"], x["end"]) for x in silence["silences"]]
    labels = classify_non_speech(audio, sr, regions, floor, min_duration=0.0)
    for x, lab in zip(silence["silences"], labels):
        x["acoustic"] = lab["kind"]
        x["level_db"] = lab["level_db"]
    silence["background_sound_total"] = round(sum(x["duration"] for x in silence["silences"]
                                                  if x.get("acoustic") == "background_sound"), 3)


def compute_metrics(diarization: list[SpeakerSegment], transcript: list[TranscriptSegment], duration: float,
                    settings: config.AppSettings, language: str | None, acoustic: dict | None = None,
                    previous: dict | None = None, vad: dict | None = None, audio=None,
                    sr: int = config.TARGET_SAMPLE_RATE, language_probs: list | None = None) -> dict:
    """All measured + inferred analytics. Mutates transcript sentiment fields.

    ``previous`` carries results that need the audio or models (quality, VAD,
    language probabilities, processing stats) through recomputation.
    """
    prev = previous or {}
    vad = vad or prev.get("vad")
    speaking = compute_speaking_time(diarization, duration, settings.turn_merge_gap)
    turns = build_turns(diarization, settings.turn_merge_gap)
    overlaps = detect_overlaps(diarization, settings.min_overlap)
    interruptions = analyze_interruptions(overlaps, list(speaking["speakers"]), speaking["recording_duration"])
    for spk, c in interruptions["per_speaker"].items():
        speaking["speakers"][spk]["interruptions_initiated"] = c["initiated"]
        speaking["speakers"][spk]["interruptions_received"] = c["received"]
    silence = analyze_silence(diarization, speaking["recording_duration"], settings.min_silence,
                              settings.long_silence, turns)
    if audio is not None:
        _annotate_silence_acoustics(silence, audio, sr)
    questions = analyze_questions(transcript, settings.question_answer_window)
    participation = analyze_participation(speaking, silence, interruptions, questions)
    sentiment = analyze_sentiment(transcript, language)
    topics = extract_topics(transcript, language=language)
    insights = extract_insights(transcript, language)
    points = key_points(transcript, topics)
    uncertain = sum(1 for t in transcript if t.uncertain)
    probs = language_probs if language_probs is not None else (prev.get("language") or {}).get("top_probabilities")
    lang_report = summarize_languages(transcript, language, probs)
    lang_report["feature_support"] = feature_support(language)
    if vad:
        diar_speech = iv.merge((d.start, d.end) for d in diarization)
        vad = {**vad, "comparison": compare_with_diarization([tuple(x) for x in vad["speech"]], diar_speech)}
    metrics = {
        "vad": vad,
        "quality": prev.get("quality"),
        "language": lang_report,
        "topic_recurrence": topic_recurrence(transcript, topics),
        "speaking": speaking,
        "interruptions": interruptions,
        "silence": silence,
        "acoustic_silence": acoustic or (previous or {}).get("acoustic_silence"),
        "participation": participation,
        "questions": questions,
        "sentiment": sentiment,
        "topics": topics,
        "key_points": points,
        "insights": insights,
        "alignment": {
            "segments": len(transcript),
            "uncertain_segments": uncertain,
            "uncertain_ratio": (uncertain / len(transcript)) if transcript else 0.0,
            "threshold": settings.uncertainty_threshold,
            "per_speaker": _speaker_confidence(transcript),
            "method": "Confidence = share of diarized speech under a transcript line that belongs to the assigned "
                      "speaker. The diarization model itself does not output per-segment probabilities.",
        },
        "thresholds": {
            "min_silence": settings.min_silence, "long_silence": settings.long_silence,
            "min_overlap": settings.min_overlap, "turn_merge_gap": settings.turn_merge_gap,
        },
    }
    metrics["summary"] = build_summary(metrics, topics, questions, insights, participation)
    if previous and previous.get("llm_summary"):
        metrics["llm_summary"] = previous["llm_summary"]
    return metrics


def single_speaker_segments(transcript: list[TranscriptSegment]) -> list[SpeakerSegment]:
    """Fallback when diarization is unavailable: all transcribed speech → Speaker 1."""
    spans = []
    for t in transcript:
        if t.words:
            spans += [(w.start, w.end) for w in t.words]
        else:
            spans.append((t.start, t.end))
    return [SpeakerSegment(s, e, "Speaker 1") for s, e in iv.merge(spans, gap=0.3)]


# --------------------------------------------------------------------------- full pipeline

RECOVERY_HINTS = {
    "decode": "The file could not be decoded. Check that it is a valid audio file, install FFmpeg, or convert it to WAV.",
    "vad": "No speech was found. Check the Audio Quality page — the file may be silent, extremely quiet or music only.",
    "transcription": "Transcription failed. Try a smaller Whisper model in Settings, free some memory, or retry.",
    "diarization": "Speaker detection failed. Retry, set an exact speaker count, or use single-speaker mode.",
    "analytics": "Analytics failed after transcription. Please retry; if it persists, check the log file.",
}


def recovery_hint(stage: str, exc: Exception) -> str:
    if isinstance(exc, ModelUnavailableError):
        return "A required model is missing. Open Settings → Models to download it, then retry."
    if isinstance(exc, MemoryError):
        return ("The computer ran out of memory. Close other programs, use a smaller Whisper model, "
                "or untick 'Keep models in memory'.")
    return RECOVERY_HINTS.get(stage, "Please retry the analysis.")


def run_analysis(recording_id: str, options: AnalysisOptions, progress=lambda f, m: None) -> dict:
    rec = repository.get_recording(recording_id)
    if rec is None:
        raise AnalysisError("Recording not found")
    s = options.settings
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    loads: dict[str, float] = {}
    current = "decode"
    repository.update_recording(recording_id, status="processing", error=None)
    monitor = ResourceMonitor()
    monitor.start()

    def stage(lo: float, hi: float, label: str):
        return lambda f, msg="": progress(lo + (hi - lo) * f, f"{label} — {msg}" if msg else label)

    try:
        # ---- Decode (+ original format metadata)
        progress(0.01, "Decoding audio")
        t = time.perf_counter()
        probe = probe_audio(rec["stored_path"])
        raw, sr = load_audio(rec["stored_path"])
        audio = preprocess(raw)
        duration = audio.size / sr
        if is_effectively_silent(raw, threshold_dbfs=-75.0):
            raise AnalysisError("No detectable sound — the recording appears to be silent.")
        processed = storage_service.processed_path_for(recording_id)
        write_wav(audio, sr, processed)
        repository.update_recording(recording_id, duration=duration, sample_rate=sr, processed_path=str(processed))
        timings["decode"] = time.perf_counter() - t

        # ---- Voice activity detection + quality (quality uses original, un-normalised levels)
        current = "vad"
        progress(0.03, "Detecting speech regions (VAD)")
        t = time.perf_counter()
        vad = detect_speech(audio, sr, min_silence_ms=int(max(0.1, min(s.min_silence, 2.0)) * 1000))
        timings["vad"] = time.perf_counter() - t
        t = time.perf_counter()
        quality = analyze_quality(raw, sr, [tuple(x) for x in vad["speech"]], probe,
                                  long_silence=max(10.0, s.long_silence))
        timings["quality"] = time.perf_counter() - t
        del raw
        acoustic = acoustic_silence(audio, sr, s.min_silence)
        if vad["speech_duration"] < 0.3:
            raise AnalysisError(
                "No speech detected by voice activity detection — skipped transcription and diarization to save time. "
                f"Audio quality: {quality['label']} ({quality['score']}/100). "
                + " ".join(i["message"] for i in quality["issues"][:2]))

        # ---- Transcription
        current = "transcription"
        progress(0.05, "Loading transcription model")
        t = time.perf_counter()
        engine = TranscriptionEngine(s.whisper_model, s.compute_type, s.cpu_threads)
        tr = engine.transcribe(audio, sr, language=options.language, beam_size=s.beam_size,
                               vad_filter=s.vad_filter, word_timestamps=s.word_timestamps,
                               progress=stage(0.07, 0.55, "Transcribing"), multilingual=options.multilingual)
        timings["transcription"] = time.perf_counter() - t
        loads["transcription"] = tr.load_seconds
        if not s.keep_models_loaded:
            TranscriptionEngine.unload()
        if not tr.segments:
            logger.info("Transcription returned no speech segments")

        # ---- Diarization
        current = "diarization"
        t = time.perf_counter()
        if options.diarization_mode == "single":
            diar = single_speaker_segments(tr.segments)
            diar_model = "single-speaker mode (no diarization)"
        else:
            progress(0.56, "Loading diarization model")
            d_engine = DiarizationEngine(s.cpu_threads)
            diar = d_engine.diarize(
                audio, sr, num_speakers=options.num_speakers, min_speakers=options.min_speakers,
                max_speakers=options.max_speakers, progress=stage(0.57, 0.92, "Identifying speakers"))
            loads["diarization"] = d_engine.last_load_seconds
            diar_model = config.DIARIZATION_MODEL_ID
            if not s.keep_models_loaded:
                DiarizationEngine.unload()
        timings["diarization"] = time.perf_counter() - t
        if not diar:
            raise AnalysisError("No speech was detected in this recording.")

        # ---- Alignment + analytics
        current = "analytics"
        progress(0.93, "Aligning transcript with speakers")
        t = time.perf_counter()
        aligned = align(tr.segments, diar, s.uncertainty_threshold, word_level=s.word_timestamps)
        progress(0.96, "Computing conversation analytics")
        metrics = compute_metrics(diar, aligned, duration, s, tr.language, acoustic,
                                  previous={"quality": quality}, vad=vad, audio=audio, sr=sr,
                                  language_probs=tr.language_probs)
        metrics["language"]["multilingual_mode"] = tr.multilingual
        metrics["language"]["selected"] = options.language or "auto"
        timings["analytics"] = time.perf_counter() - t
        del audio
        gc.collect()

        if s.ollama_enabled:
            progress(0.98, "Generating local LLM summary (Ollama)")
            try:
                metrics["llm_summary"] = {"model": s.ollama_model,
                                          "text": ollama_summarize(aligned, s.ollama_url, s.ollama_model)}
            except Exception as exc:
                metrics["llm_summary"] = {"model": s.ollama_model, "error": f"Ollama unavailable ({exc.__class__.__name__})"}

        elapsed = time.perf_counter() - t0
        monitor.stop()
        metrics["processing"] = {
            "total_seconds": elapsed, "stages": timings, "model_load_seconds": loads,
            "audio_duration": duration, "realtime_factor": elapsed / duration if duration else None,
            "speech_after_vad": tr.duration_after_vad, "resources": monitor.summary(),
            "settings": {"whisper_model": s.whisper_model, "compute_type": s.compute_type,
                         "cpu_threads": s.cpu_threads, "beam_size": s.beam_size},
        }
        repository.save_analysis(recording_id, diar, aligned, metrics)
        repository.update_recording(
            recording_id, status="completed", processing_seconds=elapsed, language=tr.language,
            language_probability=tr.language_probability, diarization_mode=options.diarization_mode,
            model_info={"transcription": tr.model_name, "diarization": diar_model},
            options={"language": options.language, "num_speakers": options.num_speakers,
                     "min_speakers": options.min_speakers, "max_speakers": options.max_speakers,
                     "multilingual": options.multilingual},
        )
        if not s.keep_original_audio and rec.get("stored_path"):
            storage_service.delete_recording_files([rec["stored_path"]])
            repository.update_recording(recording_id, stored_path=None)
        progress(1.0, "Analysis complete")
        return metrics
    except Exception as exc:
        monitor.stop()
        logger.error("Analysis failed at stage %s: %s", current, exc.__class__.__name__)
        logger.debug(traceback.format_exc())
        msg = str(exc) if isinstance(exc, (ModelUnavailableError, AnalysisError)) else f"{exc.__class__.__name__}: {exc}"
        repository.update_recording(
            recording_id, status="failed", error=msg[:2000], processing_seconds=time.perf_counter() - t0,
            model_info={"failed_stage": current, "hint": recovery_hint(current, exc),
                        "error_type": exc.__class__.__name__, "stages": timings, "resources": monitor.summary()})
        raise
    finally:
        gc.collect()


# --------------------------------------------------------------------------- manual corrections

def recompute(recording_id: str, settings: config.AppSettings | None = None,
              diarization: list[SpeakerSegment] | None = None,
              transcript: list[TranscriptSegment] | None = None, corrected: set[int] | None = None) -> dict:
    """Re-run the deterministic analytics on stored (possibly corrected) data. No ML inference."""
    data = repository.load_analysis(recording_id)
    if data is None:
        raise AnalysisError("Recording not found")
    settings = settings or repository.load_app_settings()
    diar = diarization if diarization is not None else data["diarization"]
    tr = transcript if transcript is not None else data["transcript"]
    if corrected is None:
        corrected = {i for i, c in enumerate(data["transcript_corrected"]) if c}
    prev = data["metrics"]
    audio = None
    path = data["recording"].get("processed_path")
    if path:
        try:
            import soundfile as sf

            audio, _ = sf.read(path, dtype="float32")
        except Exception:
            audio = None  # analytics still work; silences just lack the quiet/background label
    metrics = compute_metrics(diar, tr, data["recording"]["duration"], settings, data["recording"]["language"],
                              previous=prev, audio=audio)
    for k in ("multilingual_mode", "selected"):
        if k in (prev.get("language") or {}):
            metrics["language"][k] = prev["language"][k]
    if prev.get("processing"):
        metrics["processing"] = prev["processing"]
    repository.save_analysis(recording_id, diar, tr, metrics, corrected_ids=corrected)
    return metrics


def merge_speakers(recording_id: str, source: str, target: str) -> dict:
    """Relabel every segment of ``source`` as ``target`` (fixes a speaker split in two)."""
    if source == target:
        raise ValueError("Choose two different speakers")
    data = repository.load_analysis(recording_id)
    diar = [SpeakerSegment(d.start, d.end, target if d.speaker == source else d.speaker) for d in data["diarization"]]
    tr = data["transcript"]
    corrected = {i for i, c in enumerate(data["transcript_corrected"]) if c}
    for i, t in enumerate(tr):
        if t.speaker == source:
            t.speaker = target
            corrected.add(i)
        t.overlapping_speakers = sorted({target if x == source else x for x in t.overlapping_speakers} - {t.speaker})
    return recompute(recording_id, diarization=diar, transcript=tr, corrected=corrected)


def reassign_segment(recording_id: str, index: int, new_speaker: str) -> dict:
    """Attribute one transcript segment's time span to ``new_speaker`` in both
    transcript and diarization, then recompute analytics."""
    data = repository.load_analysis(recording_id)
    tr = data["transcript"]
    if not 0 <= index < len(tr):
        raise IndexError(index)
    seg = tr[index]
    old = seg.speaker
    if old == new_speaker:
        return data["metrics"]
    span = [(seg.start, seg.end)]
    diar: list[SpeakerSegment] = []
    for d in data["diarization"]:
        if d.speaker == old:
            diar += [SpeakerSegment(s, e, old) for s, e in iv.subtract([(d.start, d.end)], span)]
        else:
            diar.append(d)
    new_ivs = iv.merge([(d.start, d.end) for d in diar if d.speaker == new_speaker] + span)
    diar = [d for d in diar if d.speaker != new_speaker] + [SpeakerSegment(s, e, new_speaker) for s, e in new_ivs]
    diar.sort(key=lambda d: (d.start, d.end))
    seg.speaker = new_speaker
    seg.speaker_confidence = 1.0
    seg.uncertain = False
    for w in seg.words:
        w.speaker = new_speaker
    corrected = {i for i, c in enumerate(data["transcript_corrected"]) if c} | {index}
    return recompute(recording_id, diarization=diar, transcript=tr, corrected=corrected)


# --------------------------------------------------------------------------- background jobs

class JobManager:
    """Runs one analysis at a time in a background thread (keeps memory bounded
    on 16 GB machines and keeps the UI responsive)."""

    _lock = threading.Lock()
    _jobs: dict[str, dict] = {}
    _active: str | None = None

    @classmethod
    def start(cls, recording_id: str, options: AnalysisOptions) -> None:
        with cls._lock:
            if cls._active and cls._jobs.get(cls._active, {}).get("state") == "running":
                raise AnalysisError("Another analysis is already running. Please wait for it to finish.")
            cls._jobs[recording_id] = {"state": "running", "progress": 0.0, "message": "Queued",
                                       "error": None, "started": time.time(), "finished": None}
            cls._active = recording_id

        def update(frac: float, msg: str):
            job = cls._jobs[recording_id]
            job["progress"] = max(job["progress"], min(1.0, float(frac)))
            job["message"] = msg

        def worker():
            try:
                run_analysis(recording_id, options, update)
                cls._jobs[recording_id].update(state="completed", progress=1.0, message="Analysis complete")
            except Exception as exc:
                cls._jobs[recording_id].update(state="failed", error=str(exc),
                                               model_missing=isinstance(exc, ModelUnavailableError))
            finally:
                cls._jobs[recording_id]["finished"] = time.time()

        threading.Thread(target=worker, name=f"analysis-{recording_id[:8]}", daemon=True).start()

    @classmethod
    def status(cls, recording_id: str) -> dict | None:
        return cls._jobs.get(recording_id)

    @classmethod
    def active(cls) -> str | None:
        if cls._active and cls._jobs.get(cls._active, {}).get("state") == "running":
            return cls._active
        return None
