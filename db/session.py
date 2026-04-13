from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from .engine import get_engine

SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False, future=True)


def get_session():
    return SessionLocal()
