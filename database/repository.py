"""Data-access functions. Everything returned is a plain dict/dataclass so the
UI never touches live ORM sessions."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

import config
from core.types import SpeakerSegment, TranscriptSegment, Word
from database.db import session_scope
from database.models import (
    AnalysisResult,
    AppSetting,
    ConversationInsight,
    Recording,
    Report,
    Speaker,
    SpeakerSegmentRow,
    TranscriptSegmentRow,
)

_REC_FIELDS = [c for c in Recording.__table__.columns.keys()]


def _rec_dict(r: Recording) -> dict:
    return {k: getattr(r, k) for k in _REC_FIELDS}


# --------------------------------------------------------------------------- recordings

def create_recording(**fields) -> str:
    with session_scope() as s:
        rec = Recording(**fields)
        s.add(rec)
        s.flush()
        return rec.id


def update_recording(recording_id: str, **fields) -> None:
    with session_scope() as s:
        rec = s.get(Recording, recording_id)
        if rec is None:
            raise KeyError(recording_id)
        for k, v in fields.items():
            setattr(rec, k, v)


def get_recording(recording_id: str) -> dict | None:
    with session_scope() as s:
        rec = s.get(Recording, recording_id)
        return _rec_dict(rec) if rec else None


def list_recordings(status: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(Recording).order_by(Recording.created_at.desc())
        if status:
            q = q.where(Recording.status == status)
        return [_rec_dict(r) for r in s.scalars(q)]


def delete_recording(recording_id: str) -> list[str]:
    """Delete DB rows (cascade) and return file paths the caller should remove."""
    with session_scope() as s:
        rec = s.get(Recording, recording_id)
        if rec is None:
            return []
        paths = [p for p in (rec.stored_path, rec.processed_path) if p]
        paths += [r.path for r in s.scalars(select(Report).where(Report.recording_id == recording_id))]
        s.delete(rec)
        return paths


def delete_all_recordings() -> list[str]:
    paths: list[str] = []
    for rec in list_recordings():
        paths += delete_recording(rec["id"])
    return paths


def recordings_older_than(days: int) -> list[str]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    with session_scope() as s:
        return list(s.scalars(select(Recording.id).where(Recording.created_at < cutoff)))


# --------------------------------------------------------------------------- analysis

def save_analysis(recording_id: str, diarization: list[SpeakerSegment], transcript: list[TranscriptSegment],
                  metrics: dict, corrected_ids: set[int] | None = None) -> None:
    """Replace all analysis artefacts for a recording (idempotent).

    Speaker display names chosen by the user are preserved across re-analysis.
    """
    corrected_ids = corrected_ids or set()
    with session_scope() as s:
        rec = s.get(Recording, recording_id)
        if rec is None:
            raise KeyError(recording_id)
        old_names = {sp.label: sp.display_name for sp in s.scalars(select(Speaker).where(Speaker.recording_id == recording_id))}
        for model in (Speaker, SpeakerSegmentRow, TranscriptSegmentRow, ConversationInsight, AnalysisResult):
            s.execute(delete(model).where(model.recording_id == recording_id))

        speaking = metrics.get("speaking", {}).get("speakers", {})
        labels = list(speaking)
        for extra in sorted({t.speaker for t in transcript if t.speaker and t.speaker not in speaking}):
            labels.append(extra)
        for i, label in enumerate(labels):
            st = speaking.get(label, {})
            idx = int(label.split()[-1]) - 1 if label.split()[-1].isdigit() else i
            s.add(Speaker(recording_id=recording_id, label=label, display_name=old_names.get(label, label),
                          color_index=idx, total_time=st.get("total_time", 0.0),
                          share=st.get("share_of_speaking_time", 0.0), num_turns=st.get("num_turns", 0)))

        s.add_all(SpeakerSegmentRow(recording_id=recording_id, speaker_label=d.speaker, start=d.start, end=d.end)
                  for d in diarization)
        for i, t in enumerate(transcript):
            s.add(TranscriptSegmentRow(
                recording_id=recording_id, speaker_label=t.speaker, start=t.start, end=t.end, text=t.text,
                speaker_confidence=t.speaker_confidence, uncertain=t.uncertain, manually_corrected=i in corrected_ids,
                avg_logprob=t.avg_logprob, no_speech_prob=t.no_speech_prob, sentiment=t.sentiment,
                sentiment_score=t.sentiment_score, words=[asdict(w) for w in t.words],
                overlapping_speakers=list(t.overlapping_speakers), language=t.language,
                language_probability=t.language_probability, original_text=t.original_text,
            ))

        for kind, items in (metrics.get("insights") or {}).items():
            for it in items:
                s.add(ConversationInsight(recording_id=recording_id, kind=kind, text=it["text"],
                                          speaker_label=it.get("speaker"), start=it.get("start"),
                                          confidence=it.get("confidence"),
                                          extra={k: v for k, v in it.items() if k not in {"text", "speaker", "start", "confidence"}}))
        for t in metrics.get("topics") or []:
            s.add(ConversationInsight(recording_id=recording_id, kind="topic", text=t["topic"], start=t.get("first_mention"),
                                      extra={"mentions": t.get("mentions"), "speakers": t.get("speakers")}))
        s.add(AnalysisResult(recording_id=recording_id, metrics=metrics))
        rec.num_speakers = metrics.get("speaking", {}).get("num_speakers", len(labels))


def load_analysis(recording_id: str) -> dict | None:
    with session_scope() as s:
        rec = s.get(Recording, recording_id)
        if rec is None:
            return None
        speakers = [
            {"id": sp.id, "label": sp.label, "display_name": sp.display_name, "color_index": sp.color_index,
             "color": config.speaker_color(sp.color_index), "total_time": sp.total_time, "share": sp.share,
             "num_turns": sp.num_turns}
            for sp in s.scalars(select(Speaker).where(Speaker.recording_id == recording_id).order_by(Speaker.color_index))
        ]
        diar = [SpeakerSegment(r.start, r.end, r.speaker_label)
                for r in s.scalars(select(SpeakerSegmentRow).where(SpeakerSegmentRow.recording_id == recording_id)
                                   .order_by(SpeakerSegmentRow.start))]
        rows = list(s.scalars(select(TranscriptSegmentRow).where(TranscriptSegmentRow.recording_id == recording_id)
                              .order_by(TranscriptSegmentRow.start)))
        transcript, row_ids, corrected = [], [], []
        for r in rows:
            transcript.append(TranscriptSegment(
                start=r.start, end=r.end, text=r.text, words=[Word(**w) for w in (r.words or [])],
                avg_logprob=r.avg_logprob, no_speech_prob=r.no_speech_prob, speaker=r.speaker_label,
                speaker_confidence=r.speaker_confidence, uncertain=r.uncertain,
                overlapping_speakers=list(r.overlapping_speakers or []), sentiment=r.sentiment,
                sentiment_score=r.sentiment_score, language=r.language,
                language_probability=r.language_probability, original_text=r.original_text))
            row_ids.append(r.id)
            corrected.append(r.manually_corrected)
        analysis = s.scalar(select(AnalysisResult).where(AnalysisResult.recording_id == recording_id))
        return {
            "recording": _rec_dict(rec),
            "speakers": speakers,
            "diarization": diar,
            "transcript": transcript,
            "transcript_row_ids": row_ids,
            "transcript_corrected": corrected,
            "metrics": analysis.metrics if analysis else {},
        }


def speaker_names(recording_id: str) -> dict[str, str]:
    with session_scope() as s:
        return {sp.label: sp.display_name for sp in s.scalars(select(Speaker).where(Speaker.recording_id == recording_id))}


def rename_speaker(recording_id: str, label: str, display_name: str) -> None:
    display_name = (display_name or "").strip()[:100] or label
    with session_scope() as s:
        sp = s.scalar(select(Speaker).where(Speaker.recording_id == recording_id, Speaker.label == label))
        if sp is None:
            raise KeyError(label)
        sp.display_name = display_name


# --------------------------------------------------------------------------- search

def search_transcripts(query: str, recording_ids: list[str] | None = None, speaker_name: str | None = None,
                       limit: int = 300) -> list[dict]:
    """Full-text search across analysed recordings.

    Speaker filtering uses each recording's own display names — speakers are
    never assumed to be the same person across recordings.
    """
    from core.transcript_search import matches, parse_query

    terms = parse_query(query)
    with session_scope() as s:
        q = (select(TranscriptSegmentRow, Recording.title, Recording.created_at)
             .join(Recording, Recording.id == TranscriptSegmentRow.recording_id)
             .where(Recording.status == "completed"))
        if recording_ids:
            q = q.where(TranscriptSegmentRow.recording_id.in_(recording_ids))
        if terms:  # SQL pre-filter on the longest term, exact (Unicode case-folded) match in Python
            q = q.where(TranscriptSegmentRow.text.ilike(f"%{max(terms, key=len)}%"))
        q = q.order_by(Recording.created_at.desc(), TranscriptSegmentRow.start)
        names = {}
        out = []
        for row, title, created in s.execute(q):
            if terms and not matches(row.text, terms):
                continue
            if row.recording_id not in names:
                names[row.recording_id] = {sp.label: sp.display_name for sp in
                                           s.scalars(select(Speaker).where(Speaker.recording_id == row.recording_id))}
            disp = names[row.recording_id].get(row.speaker_label, row.speaker_label or "Unknown")
            if speaker_name and speaker_name.casefold() not in disp.casefold():
                continue
            out.append({"recording_id": row.recording_id, "title": title, "created_at": created,
                        "speaker_label": row.speaker_label, "speaker_name": disp, "start": row.start,
                        "end": row.end, "text": row.text, "uncertain": row.uncertain})
            if len(out) >= limit:
                break
        return out


# --------------------------------------------------------------------------- privacy

def delete_reports(recording_id: str) -> list[str]:
    """Remove report rows; return their file paths for deletion."""
    with session_scope() as s:
        rows = list(s.scalars(select(Report).where(Report.recording_id == recording_id)))
        paths = [r.path for r in rows]
        for r in rows:
            s.delete(r)
        return paths


def delete_analysis_data(recording_id: str) -> None:
    """Delete transcripts, speakers, segments, insights and analytics but keep the recording entry/audio."""
    with session_scope() as s:
        for model in (Speaker, SpeakerSegmentRow, TranscriptSegmentRow, ConversationInsight, AnalysisResult):
            s.execute(delete(model).where(model.recording_id == recording_id))
        rec = s.get(Recording, recording_id)
        if rec:
            rec.status = "pending"
            rec.num_speakers = 0


# --------------------------------------------------------------------------- dashboard

def dashboard_stats() -> dict:
    with session_scope() as s:
        done = Recording.status == "completed"
        n = s.scalar(select(func.count()).select_from(Recording).where(done)) or 0
        total_dur = s.scalar(select(func.coalesce(func.sum(Recording.duration), 0.0)).where(done)) or 0.0
        total_spk = s.scalar(select(func.coalesce(func.sum(Recording.num_speakers), 0)).where(done)) or 0
        total_proc = s.scalar(select(func.coalesce(func.sum(Recording.processing_seconds), 0.0)).where(done)) or 0.0
        failed = s.scalar(select(func.count()).select_from(Recording).where(Recording.status == "failed")) or 0
        return {
            "recordings": n,
            "total_duration": float(total_dur),
            "avg_duration": float(total_dur) / n if n else 0.0,
            "total_speakers": int(total_spk),
            "total_processing": float(total_proc),
            "failed": failed,
        }


def all_metrics() -> list[dict]:
    """(recording, metrics) pairs for completed recordings — used for trends and comparison."""
    with session_scope() as s:
        rows = s.execute(select(Recording, AnalysisResult).join(AnalysisResult, AnalysisResult.recording_id == Recording.id)
                         .where(Recording.status == "completed").order_by(Recording.created_at)).all()
        return [{"recording": _rec_dict(r), "metrics": a.metrics} for r, a in rows]


# --------------------------------------------------------------------------- reports

def add_report(recording_id: str, kind: str, path: str) -> None:
    with session_scope() as s:
        s.add(Report(recording_id=recording_id, kind=kind, path=path))


def list_reports(recording_id: str | None = None) -> list[dict]:
    with session_scope() as s:
        q = select(Report).order_by(Report.created_at.desc())
        if recording_id:
            q = q.where(Report.recording_id == recording_id)
        return [{"id": r.id, "recording_id": r.recording_id, "kind": r.kind, "path": r.path, "created_at": r.created_at}
                for r in s.scalars(q)]


# --------------------------------------------------------------------------- settings

def get_setting(key: str, default=None):
    with session_scope() as s:
        row = s.get(AppSetting, key)
        return row.value.get("v", default) if row else default


def set_setting(key: str, value) -> None:
    with session_scope() as s:
        row = s.get(AppSetting, key)
        if row is None:
            s.add(AppSetting(key=key, value={"v": value}))
        else:
            row.value = {"v": value}


def load_app_settings() -> config.AppSettings:
    settings = config.AppSettings()
    with session_scope() as s:
        for row in s.scalars(select(AppSetting)):
            if row.key in config.AppSettings.field_names():
                current = getattr(settings, row.key)
                value = row.value.get("v")
                try:
                    setattr(settings, row.key, type(current)(value) if current is not None and value is not None else value)
                except (TypeError, ValueError):
                    pass
    return settings


def save_app_settings(settings: config.AppSettings) -> None:
    for name in config.AppSettings.field_names():
        set_setting(name, getattr(settings, name))
