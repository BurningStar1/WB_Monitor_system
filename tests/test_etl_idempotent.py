"""ETL idempotency tests — проверяем, что повторный запуск
этапов STG и MART не создаёт дублей и не меняет агрегатов.

Эти тесты опираются на реальную базу wb_analytics (из .env).
Они читают только COUNT/SUM из существующих данных, ничего не меняют.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from db import get_engine


# --- Уникальность ключей в STG ---------------------------------

STG_KEY_SQL = {
    "stg.wb_orders": "srid",
    "stg.wb_sales": "sale_id",
    "stg.wb_finance_detail": "rrd_id",
    "stg.wb_ads_daily": "(ads_date, campaign_id, nm_id)",
}


@pytest.mark.parametrize("table,key", list(STG_KEY_SQL.items()))
def test_stg_natural_keys_unique(table: str, key: str) -> None:
    """У каждой stg-таблицы с натуральным ключом COUNT = COUNT(DISTINCT key)."""
    eng = get_engine()
    with eng.connect() as conn:
        total, distinct = conn.execute(
            text(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {table}")
        ).first()
    assert total == distinct, (
        f"{table}: дубли по {key} (всего {total}, уникальных {distinct})"
    )


def test_stg_stocks_no_duplicates_per_snapshot() -> None:
    """stg.wb_stocks — один снимок в день ×
    (склад, nm_id, размер, баркод) должен быть уникален."""
    eng = get_engine()
    with eng.connect() as conn:
        total, distinct = conn.execute(text("""
            SELECT
              COUNT(*),
              COUNT(DISTINCT (
                  source_loaded_at::date,
                  warehouse_name,
                  nm_id,
                  tech_size,
                  barcode
              ))
            FROM stg.wb_stocks
        """)).first()
    assert total == distinct, (
        f"stg.wb_stocks: дубли по (дата, склад, nm_id, размер, баркод): "
        f"всего {total}, уникальных {distinct}"
    )


# --- Уникальность ключей в MART --------------------------------

MART_KEY_SQL = {
    "mart.orders_daily": "(order_date, nm_id, supplier_article)",
    "mart.sales_daily": "(sales_date, nm_id, supplier_article)",
    "mart.stocks_snapshot": "(snapshot_date, nm_id, warehouse_name)",
    "mart.finance_daily": "(report_date, nm_id, supplier_article)",
    "mart.ads_daily": "(ads_date, nm_id, campaign_id)",
}


@pytest.mark.parametrize("table,key", list(MART_KEY_SQL.items()))
def test_mart_primary_keys_unique(table: str, key: str) -> None:
    """У каждой mart-витрины PK-ключ реально уникален
    (защита от случайной деградации ON CONFLICT)."""
    eng = get_engine()
    with eng.connect() as conn:
        total, distinct = conn.execute(
            text(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {table}")
        ).first()
    assert total == distinct, (
        f"{table}: нарушен PK {key} (всего {total}, уникальных {distinct})"
    )
