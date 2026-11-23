"""Engine/session management for the local SQLite database."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import config
from database.models import Base

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def _enable_sqlite_fk(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.close()


def init_db(url: str | None = None) -> Engine:
    """Create (or re-create for a different URL) the engine and tables."""
    global _engine, _SessionFactory
    url = url or config.DATABASE_URL
    if _engine is not None and str(_engine.url) == url:
        return _engine
    if url.startswith("sqlite:///") and ":memory:" not in url:
        config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    if url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_fk)
    Base.metadata.create_all(engine)
    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine


def get_engine() -> Engine:
    return _engine or init_db()


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionFactory is None:
        init_db()
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> tuple[bool, str]:
    try:
        from sqlalchemy import text

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "SQLite database reachable"
    except Exception as exc:
        return False, f"Database error: {exc.__class__.__name__}"
