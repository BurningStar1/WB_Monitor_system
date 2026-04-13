"""Project configuration module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    wb_api_token: str
    wb_api_host: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    log_level: str
    raw_source: str

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def _read_wb_token() -> str:
    token = os.getenv("WB_API_TOKEN", "").strip()
    if token:
        return token

    token_path = ROOT_DIR / "wb_api_key.txt"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()

    return ""


def get_settings() -> Settings:
    return Settings(
        wb_api_token=_read_wb_token(),
        wb_api_host=os.getenv("WB_API_HOST", "https://statistics-api.wildberries.ru").rstrip("/"),
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "wb_analytics"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        raw_source=os.getenv("RAW_SOURCE", "wildberries"),
    )
