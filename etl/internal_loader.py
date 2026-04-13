from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from db import get_engine
from utils import get_logger, read_xlsx


class InternalDictLoader:
    """Loads seller internal dictionaries from XLSX into dict schema."""

    def __init__(self, log_level: str = "INFO"):
        self.engine = get_engine()
        self.logger = get_logger(self.__class__.__name__, log_level)

    def load_cost_reference(self, path: str | Path) -> None:
        df = read_xlsx(path)
        required = {"nm_id", "supplier_article", "unit_cost", "valid_from", "valid_to"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in cost reference: {missing}")

        rows = df.to_dict(orient="records")
        stmt = text(
            """
            INSERT INTO dict.cost_reference (
                nm_id, supplier_article, unit_cost, valid_from, valid_to, updated_at
            ) VALUES (
                :nm_id, :supplier_article, :unit_cost, :valid_from, :valid_to, now()
            )
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, rows)

        self.logger.info("Loaded cost reference rows=%s", len(rows))

    def load_extra_expenses(self, path: str | Path) -> None:
        df = read_xlsx(path)
        required = {"expense_date", "expense_category", "amount", "comment"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in extra expenses: {missing}")

        rows = df.to_dict(orient="records")
        stmt = text(
            """
            INSERT INTO dict.extra_expenses (
                expense_date, expense_category, amount, comment, updated_at
            ) VALUES (
                :expense_date, :expense_category, :amount, :comment, now()
            )
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, rows)

        self.logger.info("Loaded extra expenses rows=%s", len(rows))

    def load_tax_reference(self, path: str | Path) -> None:
        df = read_xlsx(path)
        required = {"tax_name", "tax_rate_percent", "valid_from", "valid_to"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in tax reference: {missing}")

        rows = df.to_dict(orient="records")
        stmt = text(
            """
            INSERT INTO dict.tax_reference (
                tax_name, tax_rate_percent, valid_from, valid_to, updated_at
            ) VALUES (
                :tax_name, :tax_rate_percent, :valid_from, :valid_to, now()
            )
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt, rows)

        self.logger.info("Loaded tax reference rows=%s", len(rows))
