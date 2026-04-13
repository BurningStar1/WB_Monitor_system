"""SQLAlchemy engine and helpers."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import get_settings


def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.sqlalchemy_url, pool_pre_ping=True, future=True)
