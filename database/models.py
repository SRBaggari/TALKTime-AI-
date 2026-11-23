"""SQLAlchemy ORM models. Audio binaries live on disk; SQLite stores paths only."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    processed_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    language_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_speakers: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|processing|completed|failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    diarization_mode: Mapped[str] = mapped_column(String(20), default="pyannote")
    model_info: Mapped[dict] = mapped_column(JSON, default=dict)
    options: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    speakers: Mapped[list["Speaker"]] = relationship(back_populates="recording", cascade="all, delete-orphan", passive_deletes=True)
    speaker_segments: Mapped[list["SpeakerSegmentRow"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    transcript_segments: Mapped[list["TranscriptSegmentRow"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    analysis: Mapped["AnalysisResult | None"] = relationship(cascade="all, delete-orphan", passive_deletes=True, uselist=False)
    insights: Mapped[list["ConversationInsight"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    reports: Mapped[list["Report"]] = relationship(cascade="all, delete-orphan", passive_deletes=True)


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(50))  # "Speaker 1" — stable analysis key
    display_name: Mapped[str] = mapped_column(String(100))
    color_index: Mapped[int] = mapped_column(Integer, default=0)
    total_time: Mapped[float] = mapped_column(Float, default=0.0)
    share: Mapped[float] = mapped_column(Float, default=0.0)
    num_turns: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    recording: Mapped[Recording] = relationship(back_populates="speakers")


class SpeakerSegmentRow(Base):
    __tablename__ = "speaker_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    speaker_label: Mapped[str] = mapped_column(String(50))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    speaker_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    speaker_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    uncertain: Mapped[bool] = mapped_column(Boolean, default=False)
    manually_corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    no_speech_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    sentiment: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    words: Mapped[list] = mapped_column(JSON, default=list)
    overlapping_speakers: Mapped[list] = mapped_column(JSON, default=list)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), unique=True, index=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class ConversationInsight(Base):
    __tablename__ = "conversation_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30))  # topic|key_point|decision|action_item|commitment|important|question|llm_summary
    text: Mapped[str] = mapped_column(Text)
    speaker_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    start: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))  # pdf|csv_transcript|csv_speakers|...
    path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
