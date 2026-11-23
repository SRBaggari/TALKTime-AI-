"""Transcript search, XLSX/TXT/HTML exports, comparison, privacy/deletion and scale tests."""
import io
import time
from pathlib import Path

import pytest

import config
from core import comparison
from core.transcript_search import highlight, parse_query, parse_timestamp, search_segments, segment_at
from database import repository
from services import export_service, storage_service
from services.analysis_service import compute_metrics
from tests.conftest import seg, tseg

SEGS = [tseg(0, 4, "Let's review the marketing budget.", "Speaker 1"),
        tseg(4, 9, "The budget report is ready, I'll send it.", "Speaker 2"),
        tseg(10, 14, "మీటింగ్ రేపు ఉంది", "Speaker 1"),
        tseg(65, 70, "Great <b>work</b> everyone & thanks!", "Speaker 2")]


# ---------------------------------------------------------------- search core

@pytest.mark.parametrize("text,expected", [("1:05", 65.0), ("01:02:03", 3723.0), ("83s", 83.0), ("0:07.5", 7.5),
                                           ("budget", None), ("1:75", None)])
def test_parse_timestamp(text, expected):
    assert parse_timestamp(text) == expected


def test_parse_query_phrases():
    assert parse_query('"budget report" ready') == ["budget report", "ready"]


def test_keyword_search_all_terms():
    r = search_segments(SEGS, "budget ready")
    assert r["mode"] == "keyword" and r["indices"] == [1]
    assert search_segments(SEGS, "BUDGET")["indices"] == [0, 1]


def test_phrase_search():
    assert search_segments(SEGS, '"budget report"')["indices"] == [1]
    assert search_segments(SEGS, '"report budget"')["indices"] == []


def test_search_by_speaker_and_time_window():
    assert search_segments(SEGS, "", speakers=["Speaker 2"])["indices"] == [1, 3]
    assert search_segments(SEGS, "", start=9.5, end=20)["indices"] == [2]


def test_search_by_timestamp_finds_line_being_spoken():
    r = search_segments(SEGS, "1:07")
    assert r["mode"] == "timestamp" and r["indices"] == [3]
    assert segment_at(SEGS, 9.3) == 1  # in the gap: nearest line (0.3 s away vs 0.7 s)
    assert segment_at(SEGS, 9.6) == 2
    assert segment_at(SEGS, 40) is None


def test_search_non_latin_script():
    assert search_segments(SEGS, "రేపు")["indices"] == [2]


def test_highlight_escapes_html():
    out = highlight(SEGS[3].text, ["work"])
    assert "<mark>work</mark>" in out and "&lt;b&gt;" in out and "&amp;" in out
    assert "<b>" not in out


# ---------------------------------------------------------------- persisted data

def _store(title="Weekly sync", segs=SEGS):
    rid = repository.create_recording(title=title, original_filename="sync.wav", status="completed", duration=70.0,
                                      language="en")
    diar = [seg(s.start, s.end, s.speaker) for s in segs]
    m = compute_metrics(diar, segs, 70.0, config.AppSettings(), "en")
    m["quality"] = {"score": 82, "label": "Good", "format": {"sample_rate": 44100, "channels": 1, "codec": "mp3"},
                    "speech_level_dbfs": -22.0, "noise_floor_dbfs": -60.0, "snr_db": 38.0, "peak_dbfs": -1.0,
                    "clipping_ratio": 0.0, "clipping_events": 0, "bandwidth_hz": 7000, "issues": []}
    m["processing"] = {"total_seconds": 12.5, "stages": {"transcription": 8.0}, "model_load_seconds": {"transcription": 1.5},
                       "audio_duration": 70.0, "realtime_factor": 0.18, "resources": {"available": True,
                       "peak_rss_mb": 900.0, "mean_cpu_percent": 40.0}}
    repository.save_analysis(rid, diar, segs, m)
    repository.rename_speaker(rid, "Speaker 1", "Priya")
    return rid


def test_repository_search_across_recordings_by_speaker_name(clean_db):
    a, b = _store("Call A"), _store("Call B")
    hits = repository.search_transcripts("budget")
    assert {h["recording_id"] for h in hits} == {a, b} and len(hits) == 4
    hits = repository.search_transcripts("budget", speaker_name="priya")
    assert all(h["speaker_name"] == "Priya" for h in hits) and len(hits) == 2
    assert len(repository.search_transcripts("రేపు", recording_ids=[a])) == 1
    assert repository.search_transcripts("nonexistentword") == []


def test_xlsx_export_contains_real_values(clean_db):
    from openpyxl import load_workbook

    data = repository.load_analysis(_store())
    wb = load_workbook(io.BytesIO(export_service.xlsx_bytes(data)))
    assert {"Summary", "Speakers", "Participation", "Transcript", "Questions", "Insights & topics",
            "Audio quality", "Performance", "Transcript - Priya", "Transcript - Speaker 2"} <= set(wb.sheetnames)
    summary = {r[0]: r[1] for r in wb["Summary"].iter_rows(min_row=2, values_only=True)}
    assert summary["Title"] == "Weekly sync" and summary["Audio quality score"] == 82
    assert summary["Duration (s)"] == 70.0
    tr_rows = list(wb["Transcript"].iter_rows(min_row=2, values_only=True))
    assert len(tr_rows) == 4 and "మీటింగ్ రేపు ఉంది" in [r[6] for r in tr_rows]
    priya = list(wb["Transcript - Priya"].iter_rows(min_row=2, values_only=True))
    assert len(priya) == 2
    perf = {r[0]: r[1] for r in wb["Performance"].iter_rows(min_row=2, values_only=True)}
    assert perf["Peak memory (MB)"] == 900.0


def test_speaker_wise_text_and_interactive_html(clean_db):
    data = repository.load_analysis(_store())
    txt = export_service.speaker_transcript_text(data)
    assert "=== Priya (2 lines) ===" in txt and "[0:00] Priya: Let's review the marketing budget." in txt
    html = export_service.interactive_html(data)
    assert html.startswith("<!doctype html>") and "plotly" in html.lower()
    assert 'src="http' not in html  # plotly.js embedded → works offline
    assert "Share of speaking time" in html


# ---------------------------------------------------------------- comparison

def test_comparison_rows_and_trends(clean_db):
    _store("A")
    _store("B")
    rows = comparison.compare(repository.all_metrics())
    assert len(rows) == 2
    r = rows[0]
    assert r["speakers"] == 2 and r["quality_score"] == 82
    assert r["ranked_shares_pct"] == sorted(r["ranked_shares_pct"], reverse=True)
    assert sum(r["ranked_shares_pct"]) == pytest.approx(100)
    assert r["interruptions_per_min"] >= 0 and r["silence_pct"] >= 0
    t = comparison.trend(rows, "balance")
    assert len(t["points"]) == 2 and t["change"] is None  # needs ≥4 recordings
    t4 = comparison.trend(rows * 2, "duration_min")
    assert t4["change"] == pytest.approx(0)


# ---------------------------------------------------------------- privacy / deletion

def test_granular_deletion(clean_db):
    rid = _store()
    up = config.UPLOAD_DIR / f"{rid}.mp3"
    up.write_bytes(b"ID3")
    rep = config.REPORTS_DIR / f"{rid}.pdf"
    rep.write_bytes(b"%PDF")
    repository.update_recording(rid, stored_path=str(up))
    repository.add_report(rid, "pdf", str(rep))

    assert storage_service.delete_reports(rid) == 1 and not rep.exists()
    assert repository.list_reports(rid) == []
    assert storage_service.delete_original_audio(rid) and not up.exists()
    assert repository.get_recording(rid)["stored_path"] is None

    storage_service.delete_transcripts_and_analysis(rid)
    data = repository.load_analysis(rid)
    assert data["transcript"] == [] and data["metrics"] == {} and data["speakers"] == []
    assert data["recording"]["status"] == "pending"
    assert repository.search_transcripts("budget") == []


def test_token_masking_and_removal(tmp_path, monkeypatch):
    assert config.mask_token("hf_abcdefghijklmnop") == "hf_…mnop"
    assert config.mask_token(None) == "not set"
    env = tmp_path / ".env"
    env.write_text("# secrets\nHF_TOKEN=hf_abcdefghijklmnop\nTALKTIME_DATA_DIR=D:/x\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
    assert config.remove_hf_token_from_env_file(env) is True
    assert env.read_text(encoding="utf-8") == "# secrets\nTALKTIME_DATA_DIR=D:/x\n"
    assert config.get_hf_token() is None
    assert config.remove_hf_token_from_env_file(env) is False


# ---------------------------------------------------------------- scale

def test_large_recording_analytics_scale():
    """2-hour, 4-speaker synthetic conversation (~3,600 turns) must analyse quickly."""
    import random

    rnd = random.Random(7)
    diar, tr, t = [], [], 0.0
    while t < 7200:
        spk = f"Speaker {rnd.randint(1, 4)}"
        d = rnd.uniform(0.5, 3.5)
        diar.append(seg(t, t + d, spk))
        tr.append(tseg(t, t + d, "We need to review the project budget and timeline today?", spk))
        t += d + rnd.uniform(-0.3, 1.5)
    t0 = time.perf_counter()
    m = compute_metrics(diar, tr, 7200.0, config.AppSettings(), "en")
    elapsed = time.perf_counter() - t0
    assert m["speaking"]["num_speakers"] == 4
    assert sum(v["share_of_speaking_time"] for v in m["speaking"]["speakers"].values()) == pytest.approx(1)
    assert elapsed < 60, f"analytics took {elapsed:.1f}s"


def test_resource_monitor_samples_memory():
    import numpy as np

    from services.resource_monitor import ResourceMonitor, system_snapshot

    with ResourceMonitor(interval=0.05) as mon:
        big = np.ones((40_000_000,), dtype=np.float32)  # ~160 MB
        time.sleep(0.4)
        del big
    s = mon.summary()
    assert s["available"] and s["samples"] >= 3 and s["peak_rss_mb"] > 150
    snap = system_snapshot()
    assert snap["memory_total_gb"] > 1 and 0 <= snap["memory_percent"] <= 100
