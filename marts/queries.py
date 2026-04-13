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

# ── Forecast: daily aggregated with moving averages ──────────
FORECAST_DAILY_QUERY = """
WITH daily AS (
    SELECT
        o.order_date,
        SUM(o.orders_count) AS orders_count,
        SUM(o.orders_amount) AS orders_amount,
        COALESCE(SUM(s.sales_count), 0) AS sales_count,
        COALESCE(SUM(s.net_revenue), 0) AS net_revenue,
        COALESCE(SUM(s.gross_revenue), 0) AS gross_revenue,
        COALESCE(SUM(s.commission_amount), 0) AS commission_amount,
        COALESCE(SUM(s.cost_amount), 0) AS cost_amount,
        COALESCE(SUM(s.profit_amount), 0) AS profit_amount
    FROM mart.orders_daily o
    LEFT JOIN mart.sales_daily s
        ON o.order_date = s.sales_date
    WHERE o.order_date BETWEEN :d_from AND :d_to
    GROUP BY o.order_date
)
SELECT
    order_date,
    orders_count, orders_amount,
    sales_count, net_revenue, gross_revenue,
    commission_amount, cost_amount, profit_amount,
    AVG(orders_count) OVER w7 AS ma_orders_7d,
    AVG(orders_count) OVER w14 AS ma_orders_14d,
    AVG(orders_amount) OVER w7 AS ma_revenue_7d,
    AVG(profit_amount) OVER w7 AS ma_profit_7d
FROM daily
WINDOW
    w7  AS (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),
    w14 AS (ORDER BY order_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
ORDER BY order_date;
"""

# ── Forecast: by article with moving averages ────────────────
FORECAST_ARTICLE_QUERY = """
WITH art AS (
    SELECT
        o.nm_id, o.supplier_article,
        MAX(o.subject) AS subject, MAX(o.brand) AS brand,
        SUM(o.orders_count) AS orders_count,
        SUM(o.orders_amount) AS orders_amount,
        COUNT(DISTINCT o.order_date) AS days_with_orders,
        COALESCE(SUM(s.sales_count), 0) AS sales_count,
        COALESCE(SUM(s.net_revenue), 0) AS net_revenue,
        COALESCE(SUM(s.gross_revenue), 0) AS gross_revenue,
        COALESCE(SUM(s.commission_amount), 0) AS commission_amount,
        COALESCE(SUM(s.cost_amount), 0) AS cost_amount,
        COALESCE(SUM(s.profit_amount), 0) AS profit_amount,
        COALESCE(AVG(s.avg_price_before_spp), 0) AS avg_price_before_spp,
        COALESCE(AVG(s.avg_price_after_spp), 0) AS avg_price_after_spp,
        COALESCE(AVG(s.avg_spp), 0) AS avg_spp_pct
    FROM mart.orders_daily o
    LEFT JOIN mart.sales_daily s
        ON o.order_date = s.sales_date
        AND o.nm_id = s.nm_id AND o.supplier_article = s.supplier_article
    WHERE o.order_date BETWEEN :d_from AND :d_to
    GROUP BY o.nm_id, o.supplier_article
)
SELECT
    a.*,
    COALESCE(st.qty, 0) AS current_stock,
    CASE WHEN a.orders_count > 0 AND a.days_with_orders > 0
        THEN ROUND(a.sales_count::NUMERIC / a.orders_count * 100, 1)
        ELSE 0
    END AS buyout_pct,
    CASE WHEN a.days_with_orders > 0
        THEN ROUND(a.orders_count::NUMERIC / a.days_with_orders, 1)
        ELSE 0
    END AS avg_orders_per_day,
    CASE WHEN a.orders_count > 0 AND a.days_with_orders > 0
        THEN ROUND(
            COALESCE(st.qty, 0)::NUMERIC
            / (a.orders_count::NUMERIC / a.days_with_orders), 0
        )
        ELSE NULL
    END AS days_of_stock
FROM art a
LEFT JOIN (
    SELECT nm_id, SUM(quantity_full) AS qty
    FROM mart.v_stocks_current
    GROUP BY nm_id
) st ON a.nm_id = st.nm_id
ORDER BY a.orders_amount DESC;
"""

# ── Supply needs (Потребность) ───────────────────────────────
SUPPLY_NEEDS_QUERY = """
WITH sales_avg AS (
    SELECT
        nm_id, supplier_article,
        MAX(subject) AS subject, MAX(brand) AS brand,
        ROUND(AVG(orders_count)::NUMERIC, 2) AS avg_orders_day,
        ROUND(AVG(CASE WHEN sales_count > 0 THEN sales_count END)::NUMERIC, 2) AS avg_sales_day,
        SUM(orders_count) AS total_orders,
        SUM(sales_count) AS total_sales,
        COUNT(DISTINCT order_date) AS period_days,
        CASE WHEN SUM(orders_count) > 0
            THEN ROUND(SUM(sales_count)::NUMERIC / SUM(orders_count) * 100, 1)
            ELSE 0
        END AS buyout_pct
    FROM (
        SELECT o.order_date, o.nm_id, o.supplier_article, o.subject, o.brand,
               o.orders_count,
               COALESCE(s.sales_count, 0) AS sales_count
        FROM mart.orders_daily o
        LEFT JOIN mart.sales_daily s
            ON o.order_date = s.sales_date
            AND o.nm_id = s.nm_id AND o.supplier_article = s.supplier_article
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    ) sub
    GROUP BY nm_id, supplier_article
),
stock AS (
    SELECT nm_id, SUM(quantity_full) AS current_stock
    FROM mart.v_stocks_current
    GROUP BY nm_id
),
cost AS (
    SELECT nm_id, supplier_article,
           AVG(cost_amount / NULLIF(sales_count, 0)) AS unit_cost
    FROM mart.sales_daily
    WHERE sales_date >= CURRENT_DATE - INTERVAL '30 days'
      AND sales_count > 0
    GROUP BY nm_id, supplier_article
)
SELECT
    sa.nm_id, sa.supplier_article, sa.subject, sa.brand,
    sa.avg_orders_day, sa.avg_sales_day, sa.buyout_pct,
    COALESCE(st.current_stock, 0) AS current_stock,
    CASE WHEN sa.avg_orders_day > 0
        THEN ROUND(COALESCE(st.current_stock, 0) / sa.avg_orders_day, 0)
        ELSE NULL
    END AS days_of_stock,
    COALESCE(c.unit_cost, 0) AS unit_cost
FROM sales_avg sa
LEFT JOIN stock st ON sa.nm_id = st.nm_id
LEFT JOIN cost c ON sa.nm_id = c.nm_id AND sa.supplier_article = c.supplier_article
ORDER BY sa.avg_orders_day DESC NULLS LAST;
"""

# ── Promotions: article baseline metrics for promo calculator ─
PROMO_BASELINE_QUERY = """
SELECT
    o.nm_id, o.supplier_article,
    MAX(o.subject) AS subject, MAX(o.brand) AS brand,
    ROUND(AVG(o.orders_count)::NUMERIC, 2) AS avg_orders_day,
    COALESCE(ROUND(AVG(s.avg_price_before_spp)::NUMERIC, 0), 0) AS avg_price_before_spp,
    COALESCE(ROUND(AVG(s.avg_price_after_spp)::NUMERIC, 0), 0) AS avg_price_after_spp,
    COALESCE(ROUND(AVG(s.avg_spp)::NUMERIC, 1), 0) AS avg_spp_pct,
    CASE WHEN SUM(o.orders_count) > 0
        THEN ROUND(COALESCE(SUM(s.sales_count), 0)::NUMERIC / SUM(o.orders_count) * 100, 1)
        ELSE 0
    END AS buyout_pct,
    COALESCE(ROUND(AVG(s.commission_amount / NULLIF(s.sales_count, 0))::NUMERIC, 2), 0) AS commission_per_unit,
    COALESCE(ROUND(AVG(s.cost_amount / NULLIF(s.sales_count, 0))::NUMERIC, 2), 0) AS cost_per_unit,
    COALESCE(ROUND(SUM(s.profit_amount)::NUMERIC / NULLIF(SUM(s.sales_count), 0), 2), 0) AS profit_per_unit,
    COALESCE(SUM(s.profit_amount), 0) AS total_profit_30d,
    COALESCE(st.qty, 0) AS current_stock
FROM mart.orders_daily o
LEFT JOIN mart.sales_daily s
    ON o.order_date = s.sales_date AND o.nm_id = s.nm_id
    AND o.supplier_article = s.supplier_article
LEFT JOIN (
    SELECT nm_id, SUM(quantity_full) AS qty
    FROM mart.v_stocks_current GROUP BY nm_id
) st ON o.nm_id = st.nm_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY o.nm_id, o.supplier_article, st.qty
ORDER BY avg_orders_day DESC;
"""

# ── P&L (ОПИУ): monthly breakdown with full fee structure ────
PNL_MONTHLY_QUERY = """
SELECT
    date_trunc('month', report_date)::date AS month,
    SUM(sales_amount) AS sales_before_spp,
    SUM(returns_amount) AS returns_amount,
    SUM(sales_amount - returns_amount) AS net_sales_before_spp,
    SUM(ppvz_for_pay) AS ppvz_for_pay,
    SUM(commission_amount) AS commission,
    SUM(logistics_amount) AS logistics,
    SUM(storage_amount) AS storage,
    SUM(penalty_amount) AS penalty,
    SUM(acceptance_amount) AS acceptance,
    SUM(acquiring_amount) AS acquiring,
    SUM(deduction_amount) AS deduction,
    SUM(additional_payment_amount) AS additional_payment,
    SUM(commission_amount + logistics_amount + storage_amount
        + penalty_amount + acceptance_amount + acquiring_amount
        + deduction_amount) AS total_fees,
    SUM(sales_count) AS sales_count,
    SUM(returns_count) AS returns_count
FROM mart.finance_daily
GROUP BY 1
ORDER BY 1 DESC;
"""

# ── P&L with sales_daily (for cost/profit when finance is sparse)
PNL_SALES_MONTHLY_QUERY = """
SELECT
    date_trunc('month', sales_date)::date AS month,
    SUM(gross_revenue) AS gross_revenue,
    SUM(net_revenue) AS net_revenue,
    SUM(commission_amount) AS commission,
    SUM(cost_amount) AS cost_amount,
    SUM(tax_amount) AS tax_amount,
    SUM(profit_amount) AS profit_amount,
    SUM(operating_profit_amount) AS operating_profit,
    SUM(orders_count) AS orders_count,
    SUM(sales_count) AS sales_count,
    SUM(returns_count) AS returns_count
FROM mart.sales_daily
GROUP BY 1
ORDER BY 1 DESC;
"""
