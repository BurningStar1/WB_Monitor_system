from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from db import get_engine


SQL_FILES = [
    "01_schema_raw.sql",
    "02_schema_stg.sql",
    "03_schema_dict.sql",
    "04_schema_mart.sql",
    "05_views_and_marts.sql",
    "06_indexes.sql",
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sql_dir = root / "sql"
    engine = get_engine()

    with engine.begin() as conn:
        for filename in SQL_FILES:
            sql_text = (sql_dir / filename).read_text(encoding="utf-8")
            conn.execute(text(sql_text))

    print("Database initialized")


if __name__ == "__main__":
    main()
