from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from db import get_engine


def fetch_dataframe(sql_query: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql_query), conn, params=params or {})


def default_date_range() -> tuple[date, date]:
    return date.today() - timedelta(days=90), date.today()


# ── Dashboard (detail for pandas aggregation) ──────────────
DASHBOARD_DETAIL_QUERY = """
SELECT
    sales_date, nm_id, supplier_article, subject, brand,
    orders_count, sales_count, returns_count,
    gross_revenue, net_revenue, commission_amount,
    cost_amount, extra_expenses_amount, tax_amount,
    profit_amount, operating_profit_amount,
    avg_price_before_spp, avg_price_after_spp
FROM mart.sales_daily
WHERE sales_date BETWEEN :d_from AND :d_to
ORDER BY sales_date;
"""

# ── KPI ──────────────────────────────────────────────────────
KPI_QUERY = """
SELECT
    COALESCE(SUM(net_revenue), 0)              AS net_revenue,
    COALESCE(SUM(profit_amount), 0)            AS profit_amount,
    COALESCE(SUM(operating_profit_amount), 0)  AS operating_profit_amount,
    COALESCE(SUM(orders_count), 0)             AS orders_count,
    COALESCE(SUM(sales_count), 0)              AS sales_count,
    COALESCE(SUM(returns_count), 0)            AS returns_count,
    COALESCE(SUM(cost_amount), 0)              AS cost_amount,
    COALESCE(SUM(commission_amount), 0)        AS commission_amount,
    COALESCE(SUM(gross_revenue), 0)            AS gross_revenue
FROM mart.sales_daily
WHERE sales_date BETWEEN :d_from AND :d_to;
"""

KPI_TREND_QUERY = """
SELECT
    sales_date,
    SUM(net_revenue)              AS net_revenue,
    SUM(profit_amount)            AS profit_amount,
    SUM(orders_count)             AS orders_count,
    SUM(sales_count)              AS sales_count
FROM mart.sales_daily
WHERE sales_date BETWEEN :d_from AND :d_to
GROUP BY sales_date
ORDER BY sales_date;
"""

# ── Weekly ───────────────────────────────────────────────────
WEEKLY_QUERY = """
SELECT *
FROM mart.v_sales_weekly
WHERE week_start >= :d_from AND week_end <= :d_to
ORDER BY year_week DESC;
"""

# ── Articles ─────────────────────────────────────────────────
ARTICLE_QUERY = """
SELECT
    nm_id, supplier_article, subject, brand,
    SUM(sales_count)    AS sales_count,
    SUM(returns_count)  AS returns_count,
    SUM(net_revenue)    AS net_revenue,
    SUM(profit_amount)  AS profit_amount,
    SUM(cost_amount)    AS cost_amount
FROM mart.sales_daily
WHERE sales_date BETWEEN :d_from AND :d_to
GROUP BY nm_id, supplier_article, subject, brand
ORDER BY net_revenue DESC;
"""

# ── Stocks ───────────────────────────────────────────────────
STOCKS_QUERY = """
SELECT *
FROM mart.v_stocks_current
ORDER BY quantity_full DESC;
"""

STOCKS_BY_WH_QUERY = """
SELECT *
FROM mart.v_stocks_by_warehouse
ORDER BY warehouse_name, quantity_full DESC;
"""

# ── ABC ──────────────────────────────────────────────────────
ABC_QUERY = """
SELECT *
FROM fn_abc_classify(:d_from, :d_to)
ORDER BY total_revenue DESC;
"""

# ── Profit ───────────────────────────────────────────────────
PROFIT_QUERY = """
SELECT *
FROM mart.v_profit_report
WHERE sales_date BETWEEN :d_from AND :d_to
ORDER BY sales_date DESC, net_revenue DESC;
"""

# ── Statutory ────────────────────────────────────────────────
STATUTORY_QUERY = """
SELECT *
FROM mart.v_statutory_period_report
ORDER BY period_month DESC;
"""

# ── Orders amount (daily, for sparkline cards) ───────────────
ORDERS_DAILY_AMOUNT_QUERY = """
SELECT
    order_date, nm_id, supplier_article, subject, brand,
    orders_amount, orders_amount_disc, orders_count
FROM mart.orders_daily
WHERE order_date BETWEEN :d_from AND :d_to
ORDER BY order_date;
"""

# ── Finance (financial report breakdown) ─────────────────────
FINANCE_DAILY_QUERY = """
SELECT
    report_date, nm_id, supplier_article, subject, brand,
    sales_count, returns_count,
    sales_amount, returns_amount,
    retail_amount, ppvz_for_pay,
    commission_amount, logistics_amount, storage_amount,
    penalty_amount, acceptance_amount, acquiring_amount,
    deduction_amount, additional_payment_amount
FROM mart.finance_daily
WHERE report_date BETWEEN :d_from AND :d_to
ORDER BY report_date;
"""

# ── Extra expenses (correct total from dict, not inflated mart) ──
EXTRA_EXPENSES_QUERY = """
SELECT COALESCE(SUM(amount), 0) AS total
FROM dict.extra_expenses
WHERE expense_date BETWEEN :d_from AND :d_to;
"""

# ── Ads (daily advertising spend from WB Promotion API) ─────────
ADS_DAILY_QUERY = """
SELECT
    ads_date, nm_id, campaign_id, supplier_article,
    views_count, clicks_count, ctr, cpc,
    orders_from_ads, spend_amount
FROM mart.ads_daily
WHERE ads_date BETWEEN :d_from AND :d_to
ORDER BY ads_date;
"""

# ── Stocks history (sparklines for article report) ───────────
STOCKS_HISTORY_QUERY = """
SELECT snapshot_date, nm_id, SUM(quantity_full) AS stock_qty
FROM mart.stocks_snapshot
WHERE snapshot_date >= CURRENT_DATE - INTERVAL '14 days'
GROUP BY snapshot_date, nm_id
ORDER BY snapshot_date;
"""
