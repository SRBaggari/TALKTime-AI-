"""Speaker diarization with pyannote ``speaker-diarization-community-1``.

The pipeline is loaded from the local Hugging Face cache (or a local folder set
via ``TALKTIME_DIARIZATION_PATH``) whenever available, so no network access is
needed after the one-time download. The token is only needed for that download.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

import config
from core.types import SpeakerSegment

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

SETUP_INSTRUCTIONS = (
    "The pyannote Community-1 diarization model is not installed locally.\n"
    "1. Create a free account at https://huggingface.co/join\n"
    "2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions\n"
    "3. Create a read token at https://huggingface.co/settings/tokens\n"
    "4. Put it in the project's .env file as HF_TOKEN=hf_... (never commit this file)\n"
    "5. Run: .venv\\Scripts\\python scripts\\download_models.py --diarization\n"
    "After this one-time download the model works fully offline."
)


class ModelUnavailableError(RuntimeError):
    """A required local model is missing. ``str(exc)`` contains setup steps."""


def local_diarization_path() -> Path | None:
    """Return a local directory holding the pipeline, or None if not installed."""
    override = os.environ.get("TALKTIME_DIARIZATION_PATH")
    if override and (Path(override) / "config.yaml").is_file():
        return Path(override)
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(config.DIARIZATION_MODEL_ID, local_files_only=True))
        return path if (path / "config.yaml").is_file() else None
    except Exception:
        return None


def is_diarization_available() -> bool:
    return local_diarization_path() is not None


def relabel_by_first_appearance(segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
    """Map raw labels (``SPEAKER_00``...) to ``Speaker 1``, ``Speaker 2`` in order of first speech."""
    order: dict[str, str] = {}
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if seg.speaker not in order:
            order[seg.speaker] = f"Speaker {len(order) + 1}"
    return sorted(
        (SpeakerSegment(round(s.start, 3), round(s.end, 3), order[s.speaker]) for s in segments),
        key=lambda s: (s.start, s.end, s.speaker),
    )


class DiarizationEngine:
    _lock = threading.Lock()
    _pipeline = None

    def __init__(self, cpu_threads: int | None = None):
        self.cpu_threads = cpu_threads

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._pipeline is not None

    def load(self):
        with self._lock:
            if DiarizationEngine._pipeline is not None:
                return DiarizationEngine._pipeline
            local = local_diarization_path()
            if local is None:
                raise ModelUnavailableError(SETUP_INSTRUCTIONS)
            try:
                import torch
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise ModelUnavailableError("pyannote.audio / torch are not installed. Run setup_windows.bat.") from exc
            if self.cpu_threads:
                torch.set_num_threads(int(self.cpu_threads))
            logger.info("Loading diarization pipeline from local cache")
            pipeline = Pipeline.from_pretrained(str(local))
            if pipeline is None:
                raise ModelUnavailableError(SETUP_INSTRUCTIONS)
            pipeline.to(torch.device("cpu"))
            DiarizationEngine._pipeline = pipeline
            return pipeline

    @classmethod
    def unload(cls) -> None:
        with cls._lock:
            cls._pipeline = None
        gc.collect()

    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[SpeakerSegment]:
        """Run diarization on an in-memory waveform. Overlapping turns are preserved."""
        import torch

        pipeline = self.load()
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            if progress is None:
                return
            frac = (completed / total) if (total and completed is not None) else None
            weights = {"segmentation": (0.0, 0.45), "speaker_counting": (0.45, 0.5),
                       "embeddings": (0.5, 0.95), "discrete_diarization": (0.95, 1.0)}
            lo, hi = weights.get(step_name, (0.95, 1.0))
            progress(lo + (hi - lo) * (frac if frac is not None else 1.0), step_name.replace("_", " "))

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_speakers:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers:
                kwargs["max_speakers"] = int(max_speakers)

        with torch.inference_mode():
            output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)

        annotation = getattr(output, "speaker_diarization", output)  # pyannote 4 vs 3.x
        raw = [
            SpeakerSegment(float(turn.start), float(turn.end), str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)
            if turn.end > turn.start
        ]
        del waveform, output
        gc.collect()
        return relabel_by_first_appearance(raw)
