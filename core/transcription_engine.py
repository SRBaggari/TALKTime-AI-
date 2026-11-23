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
from core.transliteration import normalize_script
from core.types import TranscriptionResult, TranscriptSegment, Word

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Segments below this average log-probability are flagged as low confidence.
LOW_CONFIDENCE_LOGPROB = -1.0
NO_SPEECH_THRESHOLD = 0.6
# Whisper's own hallucination signal: highly repetitive output compresses too well.
COMPRESSION_RATIO_THRESHOLD = 2.4
# Upper bound on tokens per second of audio for one decoding attempt (fast speech is ~4-6 tokens/s,
# Indic scripts tokenise longer). Stops repetition loops from running to Whisper's maximum length.
MAX_TOKENS_PER_SECOND = 12
INDIC_LANGS = {"te", "ta", "kn", "ml", "bn", "gu", "pa", "mr", "hi", "ur", "or", "as", "ne", "si"}


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
    return (seg.avg_logprob is not None and seg.avg_logprob < LOW_CONFIDENCE_LOGPROB) or seg.text.startswith("[unclear]")


def model_advice(language: str | None, model_size: str) -> str | None:
    """Honest guidance: Whisper tiny/base/small are weak on Indic languages."""
    if language in INDIC_LANGS and model_size in ("tiny", "base", "small"):
        return (f"Whisper '{model_size}' is unreliable for {language}. Use 'medium' (Settings → Processing) for "
                "noticeably better Telugu/Tamil/Hindi transcripts; it is slower and needs ~2.5 GB RAM.")
    return None


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

    def _segments_from(self, seg_iter, offset: float, lang: str | None, lang_prob: float | None, duration: float,
                       progress: ProgressCallback | None) -> tuple[list[TranscriptSegment], int]:
        """Convert faster-whisper segments, shift them by ``offset`` seconds and restore native script."""
        out: list[TranscriptSegment] = []
        converted = 0
        for s in seg_iter:  # generator: decoding happens lazily here
            text = (s.text or "").strip()
            end = offset + s.end
            if progress and duration > 0:
                progress(min(1.0, end / duration), f"{end:.0f}s / {duration:.0f}s")
            if not text:
                continue
            native, original = normalize_script(text, lang)
            if original is not None:
                converted += 1
            no_speech = s.no_speech_prob is not None and s.no_speech_prob > NO_SPEECH_THRESHOLD and (s.avg_logprob or 0) < LOW_CONFIDENCE_LOGPROB
            repetitive = (getattr(s, "compression_ratio", None) or 0) > COMPRESSION_RATIO_THRESHOLD
            if no_speech or repetitive:
                native = f"[unclear] {native}"
            words = [
                Word(start=offset + float(w.start), end=offset + float(w.end),
                     word=normalize_script(w.word, lang)[0], probability=float(w.probability))
                for w in (s.words or []) if w.end >= w.start
            ]
            out.append(TranscriptSegment(
                start=offset + float(s.start), end=end, text=native, words=words,
                avg_logprob=float(s.avg_logprob) if s.avg_logprob is not None else None,
                no_speech_prob=float(s.no_speech_prob) if s.no_speech_prob is not None else None,
                language=lang, language_probability=lang_prob, original_text=original,
            ))
        return out, converted

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
        regions: list[tuple[float, float]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe in the spoken language (``task='transcribe'`` — never translated).

        ``multilingual=True`` with ``regions`` (VAD speech regions) transcribes each region
        separately with its own language detection — genuine utterance-level code-switching.
        (faster-whisper's built-in multilingual flag only re-detects per 30 s window.)
        """
        if sr != config.TARGET_SAMPLE_RATE:
            raise ValueError("Whisper expects 16 kHz audio")
        model = self.load()
        duration = audio.size / sr
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        common = dict(beam_size=beam_size, word_timestamps=word_timestamps, condition_on_previous_text=False,
                      task="transcribe")

        if multilingual and language is None and regions:
            return self._transcribe_regions(model, audio, sr, regions, duration, common, progress)

        seg_iter, info = model.transcribe(audio, language=language, vad_filter=vad_filter,
                                          multilingual=multilingual and language is None, **common)
        lang = getattr(info, "language", None)
        segments, converted = self._segments_from(seg_iter, 0.0, lang, getattr(info, "language_probability", None),
                                                  duration, progress)
        if progress:
            progress(1.0, "done")
        return TranscriptionResult(
            segments=segments,
            language=lang,
            language_probability=getattr(info, "language_probability", None),
            duration=duration,
            model_name=f"faster-whisper-{self.model_size} ({self.compute_type})",
            language_probs=sorted(getattr(info, "all_language_probs", None) or [], key=lambda x: -x[1])[:5],
            multilingual=multilingual and language is None,
            load_seconds=self.last_load_seconds,
            duration_after_vad=getattr(info, "duration_after_vad", None),
            transliterated_segments=converted,
        )

    def _transcribe_regions(self, model, audio, sr, regions, duration, common, progress) -> TranscriptionResult:
        chunks = build_chunks(regions, duration)
        # Recording-level probabilities (first 30 s), also the fallback for very short regions.
        g_lang, g_prob, g_probs = model.detect_language(audio)
        segments: list[TranscriptSegment] = []
        dist: dict[str, float] = {}
        converted = 0
        for s, e in chunks:
            piece = audio[int(s * sr): int(e * sr)]
            if e - s < MIN_DETECT_SECONDS:
                lang, prob = g_lang, None  # too short to identify reliably → recording language
            else:
                lang, prob, _ = model.detect_language(piece)
            seg_iter, _info = model.transcribe(piece, language=lang, vad_filter=False,
                                               max_new_tokens=int((e - s) * MAX_TOKENS_PER_SECOND) + 16, **common)
            segs, n = self._segments_from(seg_iter, s, lang, prob, duration, progress)
            segments += segs
            converted += n
            dist[lang] = dist.get(lang, 0.0) + (e - s)
        if progress:
            progress(1.0, "done")
        main = max(dist, key=dist.get) if dist else g_lang
        return TranscriptionResult(
            segments=sorted(segments, key=lambda x: x.start),
            language=main,
            language_probability=dict(g_probs).get(main, g_prob),
            duration=duration,
            model_name=f"faster-whisper-{self.model_size} ({self.compute_type})",
            language_probs=sorted(g_probs or [], key=lambda x: -x[1])[:5],
            multilingual=True,
            load_seconds=self.last_load_seconds,
            duration_after_vad=sum(e - s for s, e in chunks),
            language_distribution={k: round(v, 3) for k, v in dist.items()},
            transliterated_segments=converted,
        )


MIN_DETECT_SECONDS = 1.0
MAX_CHUNK_SECONDS = 28.0


def build_chunks(regions: list[tuple[float, float]], duration: float, pad: float = 0.2,
                 max_len: float = MAX_CHUNK_SECONDS) -> list[tuple[float, float]]:
    """Speech regions → padded chunks no longer than Whisper's 30 s window, on the original timeline."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(regions):
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if out and s <= out[-1][1]:  # padding made neighbours touch → keep separate but non-overlapping
            s = out[-1][1]
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        if e - s > 0.05:
            out.append((s, e))
    return out
