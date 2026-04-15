"""SQLAlchemy engine and helpers."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached singleton SQLAlchemy engine.

    Reusing one engine keeps the connection pool alive between calls, which
    avoids the overhead of opening a new pool on every page render.
    """
    settings = get_settings()
    return create_engine(
        settings.sqlalchemy_url,
        pool_pre_ping=True,
        future=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,  # Recycle connections every 30 min
    )
