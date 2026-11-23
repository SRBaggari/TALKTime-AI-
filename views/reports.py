"""PDF report generation and CSV/JSON exports."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from core.report_generator import generate_pdf
from database import repository
from services import export_service
from ui.components import load, page_header, recording_picker, section

page_header("Workspace", "Reports & exports", "Generate a PDF report or export transcript and analytics data. Files are created locally.")
rid = recording_picker()
if not rid:
    st.stop()
data = load(rid)
rec = data["recording"]
slug = re.sub(r"[^A-Za-z0-9]+", "_", rec["title"]).strip("_")[:40] or "recording"

left, right = st.columns([3, 2], gap="large")
with left:
    section("PDF report")
    st.caption("Includes recording details, summary, speaker statistics and charts, timeline, overlaps, silence, "
               "participation, sentiment, key points, questions, limitations and (optionally) the full transcript.")
    include_tr = st.checkbox("Include full speaker-wise transcript", value=True)
    if st.button("Generate PDF report", type="primary", icon=":material/picture_as_pdf:"):
        out = config.REPORTS_DIR / f"{rid}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        with st.spinner("Building report…"):
            try:
                generate_pdf(data, out, include_transcript=include_tr)
                repository.add_report(rid, "pdf", str(out))
                st.session_state.last_report = str(out)
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
    last = st.session_state.get("last_report")
    if last and Path(last).is_file() and Path(last).name.startswith(rid):
        st.success("Report ready.")
        st.download_button("Download PDF", Path(last).read_bytes(), file_name=f"TalkTime_{slug}.pdf",
                           mime="application/pdf", icon=":material/download:", type="primary")

with right:
    section("Data exports")
    exports = [
        ("Transcript (CSV)", export_service.transcript_dataframe, "transcript"),
        ("Speaker statistics (CSV)", export_service.speakers_dataframe, "speakers"),
        ("Speaker segments (CSV)", export_service.segments_dataframe, "segments"),
        ("Overlap & silence events (CSV)", export_service.events_dataframe, "events"),
    ]
    for title, fn, name in exports:
        st.download_button(title, export_service.to_csv_bytes(fn(data)), file_name=f"TalkTime_{slug}_{name}.csv",
                           mime="text/csv", width="stretch", icon=":material/table_view:", key=f"dl_{name}")
    st.download_button("Full analytics (JSON)", export_service.metrics_json_bytes(data), file_name=f"TalkTime_{slug}_analytics.json",
                       mime="application/json", width="stretch", icon=":material/data_object:")
    section("Workbooks and transcripts")

    @st.cache_data(show_spinner=False, max_entries=4)
    def _xlsx(rid_: str, stamp: str) -> bytes:
        return export_service.xlsx_bytes(load(rid_))

    @st.cache_data(show_spinner=False, max_entries=4)
    def _html(rid_: str, stamp: str) -> str:
        return export_service.interactive_html(load(rid_))

    # Cache key changes whenever names or speaker assignments are corrected.
    stamp = str(hash((str(rec.get("updated_at")), tuple(s["display_name"] for s in data["speakers"]),
                      tuple(t.speaker for t in data["transcript"]), len(data["transcript"]))))
    st.download_button("Excel workbook (XLSX)", _xlsx(rid, stamp), file_name=f"TalkTime_{slug}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
                       icon=":material/grid_on:", help="Summary, speakers, participation, transcript, segments, events, "
                       "questions, insights, audio quality, performance, and one transcript sheet per speaker.")
    st.download_button("Speaker-wise transcript (TXT)", export_service.speaker_transcript_text(data).encode("utf-8-sig"),
                       file_name=f"TalkTime_{slug}_transcript.txt", mime="text/plain", width="stretch",
                       icon=":material/article:")
    st.download_button("Interactive charts (HTML)", _html(rid, stamp).encode("utf-8"),
                       file_name=f"TalkTime_{slug}_charts.html", mime="text/html", width="stretch",
                       icon=":material/bar_chart:", help="Self-contained file with zoomable charts; opens offline in any browser.")

section("Previously generated reports")
reports = [r for r in repository.list_reports(rid) if Path(r["path"]).is_file()]
if not reports:
    st.caption("No reports generated for this recording yet.")
for r in reports[:10]:
    c1, c2 = st.columns([4, 1])
    c1.markdown(f"**{r['kind'].upper()}** · {r['created_at']:%d %b %Y %H:%M} · {Path(r['path']).stat().st_size / 1024:.0f} KB")
    c2.download_button("Download", Path(r["path"]).read_bytes(), file_name=f"TalkTime_{slug}_{r['created_at']:%Y%m%d_%H%M}.pdf",
                       mime="application/pdf", key=f"old_{r['id']}", width="stretch")
