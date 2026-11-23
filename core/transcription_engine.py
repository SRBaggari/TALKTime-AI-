"""Speech-to-text with faster-whisper (CTranslate2), CPU + int8 by default."""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Callable

import numpy as np

import config
from core.diarization_engine import ModelUnavailableError
from core.types import TranscriptionResult, TranscriptSegment, Word

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Segments below this average log-probability are flagged as low confidence.
LOW_CONFIDENCE_LOGPROB = -1.0
NO_SPEECH_THRESHOLD = 0.6


def whisper_repo(size: str) -> str:
    return config.WHISPER_REPO_TEMPLATE.format(size=size)


WHISPER_REQUIRED_FILES = ["model.bin", "config.json", "tokenizer.json"]


def local_whisper_path(size: str) -> str | None:
    from core.model_cache import cached_snapshot

    snap = cached_snapshot(whisper_repo(size), WHISPER_REQUIRED_FILES)
    return str(snap) if snap else None


def is_whisper_available(size: str) -> bool:
    return local_whisper_path(size) is not None


def is_low_confidence(seg: TranscriptSegment) -> bool:
    return seg.avg_logprob is not None and seg.avg_logprob < LOW_CONFIDENCE_LOGPROB


class TranscriptionEngine:
    _lock = threading.Lock()
    _models: dict[tuple, object] = {}

    def __init__(self, model_size: str = "small", compute_type: str = "int8", cpu_threads: int = 4):
        if model_size not in config.WHISPER_SIZES:
            raise ValueError(f"Unknown Whisper model size: {model_size}")
        self.model_size = model_size
        self.compute_type = compute_type
        self.cpu_threads = int(cpu_threads)

    @property
    def key(self) -> tuple:
        return (self.model_size, self.compute_type, self.cpu_threads)

    last_load_seconds: float = 0.0

    def load(self):
        with self._lock:
            if self.key in self._models:
                self.last_load_seconds = 0.0
                return self._models[self.key]
            t0 = time.perf_counter()
            path = local_whisper_path(self.model_size)
            if path is None:
                raise ModelUnavailableError(
                    f"Faster-Whisper '{self.model_size}' is not downloaded.\n"
                    f"Run: .venv\\Scripts\\python scripts\\download_models.py --whisper {self.model_size}\n"
                    "or use the Settings page → Models → Download."
                )
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ModelUnavailableError("faster-whisper is not installed. Run setup_windows.bat.") from exc
            logger.info("Loading Whisper %s (%s) from local cache", self.model_size, self.compute_type)
            # Only one Whisper model kept in memory at a time.
            self._models.clear()
            gc.collect()
            model = WhisperModel(path, device="cpu", compute_type=self.compute_type,
                                 cpu_threads=self.cpu_threads, local_files_only=True)
            self._models[self.key] = model
            self.last_load_seconds = time.perf_counter() - t0
            return model

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._models.clear()
        gc.collect()

    @classmethod
    def is_loaded(cls) -> bool:
        return bool(cls._models)

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = config.TARGET_SAMPLE_RATE,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        progress: ProgressCallback | None = None,
        multilingual: bool = False,
    ) -> TranscriptionResult:
        """``multilingual=True`` re-detects the language per segment (code-switched speech);
        the original spoken language/script is preserved — nothing is translated."""
        if sr != config.TARGET_SAMPLE_RATE:
            raise ValueError("Whisper expects 16 kHz audio")
        model = self.load()
        duration = audio.size / sr
        seg_iter, info = model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,  # reduces hallucinated repetition loops
            multilingual=multilingual and language is None,
            task="transcribe",  # never translate: keep the original spoken language
        )
        segments: list[TranscriptSegment] = []
        for s in seg_iter:  # generator: decoding happens lazily here
            text = (s.text or "").strip()
            if progress and duration > 0:
                progress(min(1.0, s.end / duration), f"{s.end:.0f}s / {duration:.0f}s")
            if not text:
                continue
            if s.no_speech_prob is not None and s.no_speech_prob > NO_SPEECH_THRESHOLD and (s.avg_logprob or 0) < LOW_CONFIDENCE_LOGPROB:
                text = f"[unclear] {text}"
            words = [
                Word(start=float(w.start), end=float(w.end), word=w.word, probability=float(w.probability))
                for w in (s.words or [])
                if w.end >= w.start
            ]
            segments.append(TranscriptSegment(
                start=float(s.start), end=float(s.end), text=text, words=words,
                avg_logprob=float(s.avg_logprob) if s.avg_logprob is not None else None,
                no_speech_prob=float(s.no_speech_prob) if s.no_speech_prob is not None else None,
            ))
        if progress:
            progress(1.0, "done")
        return TranscriptionResult(
            segments=segments,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            duration=duration,
            model_name=f"faster-whisper-{self.model_size} ({self.compute_type})",
            language_probs=sorted(getattr(info, "all_language_probs", None) or [], key=lambda x: -x[1])[:5],
            multilingual=multilingual and language is None,
            load_seconds=self.last_load_seconds,
            duration_after_vad=getattr(info, "duration_after_vad", None),
        )
