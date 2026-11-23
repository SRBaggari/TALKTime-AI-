"""End-to-end analysis pipeline and a single-slot background job runner.

Pipeline: decode → preprocess → transcribe → diarize → align → analytics → NLP → save.
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

import config
from core import intervals as iv
from core.audio_loader import load_audio
from core.audio_preprocessor import is_effectively_silent, preprocess, write_wav
from core.conversation_summary import build_summary, extract_insights, extract_topics, key_points, ollama_summarize
from core.diarization_engine import DiarizationEngine, ModelUnavailableError
from core.interruption_analysis import analyze_interruptions
from core.overlap_detection import detect_overlaps
from core.participation_analysis import analyze_participation
from core.question_analysis import analyze_questions
from core.sentiment_analysis import analyze_sentiment
from core.silence_detection import acoustic_silence, analyze_silence
from core.speaker_alignment import align
from core.speaking_time import build_turns, compute_speaking_time
from core.transcription_engine import TranscriptionEngine
from core.types import SpeakerSegment, TranscriptSegment
from database import repository
from services import storage_service

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
    settings: config.AppSettings = field(default_factory=config.AppSettings)


# --------------------------------------------------------------------------- analytics (pure, no ML)

def compute_metrics(diarization: list[SpeakerSegment], transcript: list[TranscriptSegment], duration: float,
                    settings: config.AppSettings, language: str | None, acoustic: dict | None = None,
                    previous: dict | None = None) -> dict:
    """All measured + inferred analytics. Mutates transcript sentiment fields."""
    speaking = compute_speaking_time(diarization, duration, settings.turn_merge_gap)
    turns = build_turns(diarization, settings.turn_merge_gap)
    overlaps = detect_overlaps(diarization, settings.min_overlap)
    interruptions = analyze_interruptions(overlaps, list(speaking["speakers"]), speaking["recording_duration"])
    for spk, c in interruptions["per_speaker"].items():
        speaking["speakers"][spk]["interruptions_initiated"] = c["initiated"]
        speaking["speakers"][spk]["interruptions_received"] = c["received"]
    silence = analyze_silence(diarization, speaking["recording_duration"], settings.min_silence,
                              settings.long_silence, turns)
    participation = analyze_participation(speaking)
    questions = analyze_questions(transcript, settings.question_answer_window)
    sentiment = analyze_sentiment(transcript, language)
    topics = extract_topics(transcript, language=language)
    insights = extract_insights(transcript, language)
    points = key_points(transcript, topics)
    uncertain = sum(1 for t in transcript if t.uncertain)
    metrics = {
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

def run_analysis(recording_id: str, options: AnalysisOptions, progress=lambda f, m: None) -> dict:
    rec = repository.get_recording(recording_id)
    if rec is None:
        raise AnalysisError("Recording not found")
    s = options.settings
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    repository.update_recording(recording_id, status="processing", error=None)

    def stage(lo: float, hi: float, label: str):
        return lambda f, msg="": progress(lo + (hi - lo) * f, f"{label} — {msg}" if msg else label)

    try:
        progress(0.01, "Decoding audio")
        t = time.perf_counter()
        audio, sr = load_audio(rec["stored_path"])
        audio = preprocess(audio)
        duration = audio.size / sr
        if is_effectively_silent(audio):
            raise AnalysisError("No detectable sound — the recording appears to be silent.")
        processed = storage_service.processed_path_for(recording_id)
        write_wav(audio, sr, processed)
        repository.update_recording(recording_id, duration=duration, sample_rate=sr, processed_path=str(processed))
        acoustic = acoustic_silence(audio, sr, s.min_silence)
        timings["decode"] = time.perf_counter() - t

        # ---- Transcription
        progress(0.05, "Loading transcription model")
        t = time.perf_counter()
        engine = TranscriptionEngine(s.whisper_model, s.compute_type, s.cpu_threads)
        tr = engine.transcribe(audio, sr, language=options.language, beam_size=s.beam_size,
                               vad_filter=s.vad_filter, word_timestamps=s.word_timestamps,
                               progress=stage(0.07, 0.55, "Transcribing"))
        timings["transcription"] = time.perf_counter() - t
        if not s.keep_models_loaded:
            TranscriptionEngine.unload()
        if not tr.segments:
            logger.info("Transcription returned no speech segments")

        # ---- Diarization
        t = time.perf_counter()
        if options.diarization_mode == "single":
            diar = single_speaker_segments(tr.segments)
            diar_model = "single-speaker mode (no diarization)"
        else:
            progress(0.56, "Loading diarization model")
            diar = DiarizationEngine(s.cpu_threads).diarize(
                audio, sr, num_speakers=options.num_speakers, min_speakers=options.min_speakers,
                max_speakers=options.max_speakers, progress=stage(0.57, 0.92, "Identifying speakers"))
            diar_model = config.DIARIZATION_MODEL_ID
            if not s.keep_models_loaded:
                DiarizationEngine.unload()
        timings["diarization"] = time.perf_counter() - t
        del audio
        gc.collect()
        if not diar:
            raise AnalysisError("No speech was detected in this recording.")

        # ---- Alignment + analytics
        progress(0.93, "Aligning transcript with speakers")
        t = time.perf_counter()
        aligned = align(tr.segments, diar, s.uncertainty_threshold, word_level=s.word_timestamps)
        progress(0.96, "Computing conversation analytics")
        metrics = compute_metrics(diar, aligned, duration, s, tr.language, acoustic)
        timings["analytics"] = time.perf_counter() - t

        if s.ollama_enabled:
            progress(0.98, "Generating local LLM summary (Ollama)")
            try:
                metrics["llm_summary"] = {"model": s.ollama_model,
                                          "text": ollama_summarize(aligned, s.ollama_url, s.ollama_model)}
            except Exception as exc:
                metrics["llm_summary"] = {"model": s.ollama_model, "error": f"Ollama unavailable ({exc.__class__.__name__})"}

        elapsed = time.perf_counter() - t0
        metrics["processing"] = {"total_seconds": elapsed, "stages": timings,
                                 "realtime_factor": elapsed / duration if duration else None}
        repository.save_analysis(recording_id, diar, aligned, metrics)
        repository.update_recording(
            recording_id, status="completed", processing_seconds=elapsed, language=tr.language,
            language_probability=tr.language_probability, diarization_mode=options.diarization_mode,
            model_info={"transcription": tr.model_name, "diarization": diar_model},
            options={"language": options.language, "num_speakers": options.num_speakers,
                     "min_speakers": options.min_speakers, "max_speakers": options.max_speakers},
        )
        if not s.keep_original_audio and rec.get("stored_path"):
            storage_service.delete_recording_files([rec["stored_path"]])
            repository.update_recording(recording_id, stored_path=None)
        progress(1.0, "Analysis complete")
        return metrics
    except ModelUnavailableError as exc:
        repository.update_recording(recording_id, status="failed", error=str(exc))
        raise
    except Exception as exc:
        logger.error("Analysis failed: %s", exc.__class__.__name__)
        logger.debug(traceback.format_exc())
        repository.update_recording(recording_id, status="failed", error=str(exc)[:1000])
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
    metrics = compute_metrics(diar, tr, data["recording"]["duration"], settings, data["recording"]["language"],
                              previous=prev)
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
