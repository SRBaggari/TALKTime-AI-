"""Plain data structures shared by the processing pipeline.

They are intentionally free of ML-library types so analytics can be unit-tested
with synthetic data and serialised to JSON / SQLite easily.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SpeakerSegment:
    """A diarization interval: ``speaker`` is speaking during [start, end)."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float = 1.0
    speaker: str | None = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    # Filled by speaker alignment
    speaker: str | None = None
    speaker_confidence: float = 0.0
    uncertain: bool = False
    overlapping_speakers: list[str] = field(default_factory=list)
    # Filled by NLP analysis
    sentiment: str | None = None
    sentiment_score: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptSegment":
        words = [Word(**w) for w in d.get("words") or []]
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "words"}
        return cls(words=words, **kwargs)


@dataclass
class TranscriptionResult:
    segments: list[TranscriptSegment]
    language: str | None
    language_probability: float | None
    duration: float
    model_name: str = ""
