"""TalkTime AI — Streamlit entry point.

Run with:  streamlit run app.py   (or run_windows.bat)
"""
from __future__ import annotations

import logging
import time

import config  # noqa: F401  (sets privacy env vars before ML imports)
import streamlit as st

from database.db import init_db
from services import model_manager, storage_service
from services.analysis_service import JobManager
from ui.components import inject_css, settings
from utils.logging_config import setup_logging

st.set_page_config(page_title=config.APP_NAME, page_icon=":material/graphic_eq:", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def bootstrap() -> bool:
    """One-time process setup: directories, logging, database, retention."""
    config.ensure_directories()
    setup_logging()
    init_db()
    from database import repository

    s = repository.load_app_settings()
    removed = storage_service.apply_retention(s.retention_days)
    if removed:
        logging.getLogger(__name__).info("Retention policy removed %d recording(s)", removed)
    return True


@st.cache_data(ttl=60, show_spinner=False)
def readiness(whisper_model: str) -> tuple[bool, int]:
    checks = model_manager.startup_checks(settings_snapshot(whisper_model))
    return model_manager.offline_ready(checks), sum(1 for c in checks if c.required and not c.ok)


def settings_snapshot(whisper_model: str) -> config.AppSettings:
    s = config.AppSettings()
    s.whisper_model = whisper_model
    return s


bootstrap()
inject_css()
s = settings()

pages = {
    "Overview": [
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/space_dashboard:", default=True),
        st.Page("views/upload_audio.py", title="Upload Recording", icon=":material/upload_file:"),
    ],
    "Analysis": [
        st.Page("views/speaker_analysis.py", title="Speaker Analysis", icon=":material/record_voice_over:"),
        st.Page("views/conversation_timeline.py", title="Conversation Timeline", icon=":material/view_timeline:"),
        st.Page("views/conversation_intelligence.py", title="Conversation Intelligence", icon=":material/insights:"),
        st.Page("views/call_comparison.py", title="Call Comparison", icon=":material/compare_arrows:"),
    ],
    "Workspace": [
        st.Page("views/reports.py", title="Reports", icon=":material/description:"),
        st.Page("views/settings.py", title="Settings", icon=":material/settings:"),
    ],
}

st.logo(str(config.ASSETS_DIR / "logo" / "logo.svg"), size="large",
        icon_image=str(config.ASSETS_DIR / "logo" / "icon.svg"))

nav = st.navigation(pages)

with st.sidebar:
    ready, missing = readiness(s.whisper_model)
    if ready:
        st.markdown('<span class="tt-badge tt-badge-ok">✓ Offline ready</span>', unsafe_allow_html=True)
        st.caption("Models cached locally. No data leaves this computer.")
    else:
        st.markdown(f'<span class="tt-badge tt-badge-warn">! Setup incomplete ({missing})</span>', unsafe_allow_html=True)
        st.caption("Some components are missing. See Settings → System status.")
    active = JobManager.active()
    if active:
        job = JobManager.status(active)
        st.markdown('<span class="tt-badge tt-badge-info">• Analysis running</span>', unsafe_allow_html=True)
        st.progress(job["progress"], text=f"{job['progress'] * 100:.0f}% · {int(time.time() - job['started'])}s")
    st.caption(f"v{config.APP_VERSION}")

nav.run()
