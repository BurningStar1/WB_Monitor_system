from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from db import get_engine

# Streamlit cache: available inside the app, no-op in scripts/tests.
try:
    import streamlit as _st
    _cache = _st.cache_data(ttl=600, show_spinner=False)
except Exception:  # pragma: no cover
    def _cache(fn):
        return fn


def _fetch_impl(sql_query: str, params_items: tuple | None) -> pd.DataFrame:
    """Inner implementation keyed on hashable params (tuple of sorted items)."""
    params = dict(params_items) if params_items else {}
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql_query), conn, params=params)


_fetch_impl_cached = _cache(_fetch_impl)


def fetch_dataframe(sql_query: str, params: dict | None = None) -> pd.DataFrame:
    """Fetch a dataframe, cached for 10 min per (sql, params) inside Streamlit."""
    params_items = tuple(sorted((params or {}).items()))
    return _fetch_impl_cached(sql_query, params_items)


def default_date_range() -> tuple[date, date]:
    return date.today() - timedelta(days=90), date.today()


# Master list of active articles for global search (no date filter)
ARTICLES_MASTER_QUERY = """
SELECT nm_id, supplier_article, subject, brand
FROM mart.sales_daily
WHERE sales_date >= CURRENT_DATE - INTERVAL '180 days'
GROUP BY nm_id, supplier_article, subject, brand
ORDER BY SUM(sales_count) DESC NULLS LAST
LIMIT 500;
"""


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
# Includes cost_amount + tax_amount via LATERAL JOINs so totals are stable
# regardless of the selected date range (dict lookups are keyed by nm_id).
#
# Tax / net_profit считаются АДДИТИВНО по строкам (rate × pre_tax_row, без
# GREATEST), чтобы SUM(tax_amount) = rate × SUM(pre_tax) точно совпадало с
# агрегатным налогом Raskка. Для прибыльных периодов это даёт корректный
# результат; для убыточных (редкие кейсы) страницы должны клиппать:
#     tax_clipped = max(sum_pre_tax, 0) * rate
#     net_profit  = sum_pre_tax - tax_clipped
# С этой целью в выборку добавлены pre_tax_profit и tax_rate_pct.
FINANCE_DAILY_QUERY = """
SELECT
    f.report_date, f.nm_id, f.supplier_article, f.subject, f.brand,
    f.sales_count, f.returns_count,
    f.sales_amount, f.returns_amount,
    f.retail_amount, f.ppvz_for_pay,
    f.commission_amount, f.logistics_amount, f.storage_amount,
    f.penalty_amount, f.acceptance_amount, f.acquiring_amount,
    f.deduction_amount, f.additional_payment_amount,
    COALESCE(cr.unit_cost, 0)                            AS unit_cost,
    -- Себестоимость по нетто-количеству (sold - returns) по методологии RASK.
    COALESCE(cr.unit_cost, 0)
        * (f.sales_count - f.returns_count)              AS cost_amount,
    COALESCE(ad.spend_amount, 0)                         AS ads_spend,
    COALESCE(tx.tax_rate_percent, 0)                     AS tax_rate_pct,
    -- Pre-tax per RASK ОПИУ (xlsx-методология пользователя):
    --   ppvz − логистика − хранение − штрафы − приёмка − удержания
    --   + доп. платежи − себестоимость.
    -- `deduction_amount` из WB Finance API включает внутреннюю рекламу,
    -- отзывы и прочие удержания — это «всё в одном». Поэтому ads_spend
    -- из Promotion API НЕ вычитается повторно (иначе двойной счёт).
    -- Эквайринг пользователем из xlsx не учитывается.
    f.ppvz_for_pay
        - f.logistics_amount - f.storage_amount
        - f.penalty_amount - f.acceptance_amount
        - f.deduction_amount
        + f.additional_payment_amount
        - COALESCE(cr.unit_cost, 0)
          * (f.sales_count - f.returns_count)              AS pre_tax_profit,
    f.ppvz_for_pay
        - f.logistics_amount - f.storage_amount
        - f.penalty_amount - f.acceptance_amount
        - f.deduction_amount
        + f.additional_payment_amount
        - COALESCE(cr.unit_cost, 0)
          * (f.sales_count - f.returns_count)              AS gross_profit_amount,
    -- Tax и net считаются АДДИТИВНО (без per-row GREATEST), чтобы
    -- SUM(tax) и SUM(net) по дням совпадали с агрегатным налогом
    -- PNL_MONTHLY/FIN_STATUTORY. Если на каком-то дне pre_tax < 0,
    -- то tax_row < 0 (компенсация налога), а net_row = pre_tax_row
    -- × (1-rate). Для визуализации при необходимости клиппить на
    -- уровне страницы: net_display = max(net, 0).
    COALESCE(tx.tax_rate_percent, 0) / 100.0
        * (f.ppvz_for_pay
           - f.logistics_amount - f.storage_amount
           - f.penalty_amount - f.acceptance_amount
           - f.deduction_amount
           + f.additional_payment_amount
           - COALESCE(cr.unit_cost, 0)
             * (f.sales_count - f.returns_count))           AS tax_amount,
    (1 - COALESCE(tx.tax_rate_percent, 0) / 100.0)
        * (f.ppvz_for_pay
           - f.logistics_amount - f.storage_amount
           - f.penalty_amount - f.acceptance_amount
           - f.deduction_amount
           + f.additional_payment_amount
           - COALESCE(cr.unit_cost, 0)
             * (f.sales_count - f.returns_count))           AS net_profit_amount
FROM mart.finance_daily f
LEFT JOIN LATERAL (
    SELECT c.unit_cost FROM dict.cost_reference c
    WHERE c.nm_id = f.nm_id
      AND f.report_date BETWEEN c.valid_from AND c.valid_to
    ORDER BY c.valid_from DESC LIMIT 1
) cr ON true
LEFT JOIN LATERAL (
    SELECT t.tax_rate_percent FROM dict.tax_reference t
    WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
    ORDER BY t.valid_from DESC LIMIT 1
) tx ON true
LEFT JOIN LATERAL (
    SELECT SUM(spend_amount) AS spend_amount FROM mart.ads_daily a
    WHERE a.nm_id = f.nm_id AND a.ads_date = f.report_date
) ad ON true
WHERE f.report_date BETWEEN :d_from AND :d_to
ORDER BY f.report_date;
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
WITH fin_detail AS (
    SELECT
        f.report_date,
        f.sales_count,
        f.ppvz_for_pay,
        f.sales_amount,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        COALESCE(cr.unit_cost, 0) * f.sales_count AS cost_amount,
        f.ppvz_for_pay
            - f.logistics_amount - f.storage_amount
            - f.penalty_amount - f.acceptance_amount
            - f.acquiring_amount - f.deduction_amount
            + f.additional_payment_amount
            - COALESCE(cr.unit_cost, 0) * f.sales_count AS profit_amount
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    WHERE f.report_date BETWEEN :d_from AND :d_to
),
daily AS (
    SELECT
        o.order_date,
        SUM(o.orders_count)                     AS orders_count,
        SUM(o.orders_amount)                    AS orders_amount,
        COALESCE(SUM(fd.sales_count), 0)        AS sales_count,
        COALESCE(SUM(fd.ppvz_for_pay), 0)       AS net_revenue,
        COALESCE(SUM(fd.sales_amount), 0)       AS gross_revenue,
        COALESCE(SUM(fd.commission_amount), 0)  AS commission_amount,
        COALESCE(SUM(fd.cost_amount), 0)        AS cost_amount,
        COALESCE(SUM(fd.profit_amount), 0)      AS profit_amount
    FROM mart.orders_daily o
    LEFT JOIN fin_detail fd
        ON o.order_date = fd.report_date
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
WITH fin_detail AS (
    SELECT
        f.report_date,
        f.nm_id,
        f.supplier_article,
        f.sales_count,
        f.ppvz_for_pay,
        f.sales_amount,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        COALESCE(cr.unit_cost, 0) * f.sales_count AS cost_amount,
        f.ppvz_for_pay
            - f.logistics_amount - f.storage_amount
            - f.penalty_amount - f.acceptance_amount
            - f.acquiring_amount - f.deduction_amount
            + f.additional_payment_amount
            - COALESCE(cr.unit_cost, 0) * f.sales_count AS profit_amount
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    WHERE f.report_date BETWEEN :d_from AND :d_to
),
art AS (
    SELECT
        o.nm_id, o.supplier_article,
        MAX(o.subject) AS subject, MAX(o.brand) AS brand,
        SUM(o.orders_count) AS orders_count,
        SUM(o.orders_amount) AS orders_amount,
        COUNT(DISTINCT o.order_date) AS days_with_orders,
        COALESCE(SUM(fd.sales_count), 0) AS sales_count,
        COALESCE(SUM(fd.ppvz_for_pay), 0) AS net_revenue,
        COALESCE(SUM(fd.sales_amount), 0) AS gross_revenue,
        COALESCE(SUM(fd.commission_amount), 0) AS commission_amount,
        COALESCE(SUM(fd.cost_amount), 0) AS cost_amount,
        COALESCE(SUM(fd.profit_amount), 0) AS profit_amount,
        COALESCE(AVG(s.avg_price_before_spp), 0) AS avg_price_before_spp,
        COALESCE(AVG(s.avg_price_after_spp), 0) AS avg_price_after_spp,
        COALESCE(AVG(s.avg_spp), 0) AS avg_spp_pct
    FROM mart.orders_daily o
    LEFT JOIN fin_detail fd
        ON o.order_date = fd.report_date
        AND o.nm_id = fd.nm_id
    LEFT JOIN mart.sales_daily s
        ON o.order_date = s.sales_date
        AND o.nm_id = s.nm_id
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
            AND o.nm_id = s.nm_id
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
LEFT JOIN cost c ON sa.nm_id = c.nm_id
ORDER BY sa.avg_orders_day DESC NULLS LAST;
"""

# ── P&L (ОПИУ): monthly breakdown with full fee structure ────
# Tax считается на АГРЕГАТНОМ уровне: rate × GREATEST(sum_pre_tax, 0).
# Иначе на пошаговом per-row GREATEST отрицательные строки обнуляются и
# налог завышается. Именно так Raskка считает налог по месячному ОПИУ.
PNL_MONTHLY_QUERY = """
WITH detail AS (
    SELECT
        f.report_date,
        f.sales_count,
        f.returns_count,
        f.sales_amount,
        f.returns_amount,
        f.retail_amount,
        f.ppvz_for_pay,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        -- Себестоимость по нетто-количеству (RASK ОПИУ).
        COALESCE(cr.unit_cost, 0)
            * (f.sales_count - f.returns_count) AS cost_amount,
        COALESCE(tx.tax_rate_percent, 0)        AS tax_rate_pct
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    LEFT JOIN LATERAL (
        SELECT t.tax_rate_percent FROM dict.tax_reference t
        WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
        ORDER BY t.valid_from DESC LIMIT 1
    ) tx ON true
),
ads_monthly AS (
    SELECT date_trunc('month', ads_date)::date AS month,
           SUM(spend_amount)                   AS ads_spend
    FROM mart.ads_daily
    GROUP BY 1
),
extra_monthly AS (
    SELECT date_trunc('month', expense_date)::date AS month,
           SUM(amount)                             AS extra_amount
    FROM dict.extra_expenses
    GROUP BY 1
),
monthly AS (
    SELECT
        date_trunc('month', report_date)::date              AS month,
        SUM(sales_amount)                                   AS sales_before_spp,
        SUM(returns_amount)                                 AS returns_amount,
        SUM(sales_amount - returns_amount)                  AS net_sales_before_spp,
        SUM(retail_amount)                                  AS retail_amount,
        SUM(ppvz_for_pay)                                   AS ppvz_for_pay,
        SUM(commission_amount)                              AS commission,
        SUM(logistics_amount)                               AS logistics,
        SUM(storage_amount)                                 AS storage,
        SUM(penalty_amount)                                 AS penalty,
        SUM(acceptance_amount)                              AS acceptance,
        SUM(acquiring_amount)                               AS acquiring,
        SUM(deduction_amount)                               AS deduction,
        SUM(additional_payment_amount)                      AS additional_payment,
        SUM(commission_amount + logistics_amount + storage_amount
            + penalty_amount + acceptance_amount + acquiring_amount
            + deduction_amount)                             AS total_fees,
        SUM(cost_amount)                                    AS cost_amount,
        MAX(tax_rate_pct)                                   AS tax_rate_pct,
        SUM(sales_count)                                    AS sales_count,
        SUM(returns_count)                                  AS returns_count
    FROM detail
    GROUP BY 1
),
joined AS (
    SELECT
        m.*,
        COALESCE(a.ads_spend, 0)    AS ads_spend,
        COALESCE(e.extra_amount, 0) AS extra_expenses,
        -- EBITDA по методологии РАСК ОПИУ (из эталонного xlsx пользователя):
        -- Формула идёт от "Реализации после СПП" (retail) + correction ppvz→retail
        -- = ppvz_for_pay  минус удержания WB:
        --   - логистика, хранение, штрафы, приёмка
        --   - deduction_amount (внутренняя реклама + отзывы + прочие удержания,
        --     всё из finance_daily — единый источник)
        --   + доп. платежи
        --   - себестоимость
        --   - extra_expenses (прочие операционные расходы из справочника)
        -- Эквайринг НЕ вычитается (xlsx-методология).
        -- ads_spend из WB Promotion API используется только ИНФОРМАЦИОННО
        -- (одна из составляющих deduction_amount) и НЕ вычитается повторно.
        (m.ppvz_for_pay
            - m.logistics - m.storage
            - m.penalty - m.acceptance
            - m.deduction
            + m.additional_payment
            - m.cost_amount
            - COALESCE(e.extra_amount, 0)
        )                            AS pre_tax_profit
    FROM monthly m
    LEFT JOIN ads_monthly a   ON a.month = m.month
    LEFT JOIN extra_monthly e ON e.month = m.month
)
SELECT
    month,
    sales_before_spp, returns_amount, net_sales_before_spp,
    retail_amount, ppvz_for_pay,
    commission, logistics, storage, penalty, acceptance, acquiring,
    deduction, additional_payment, total_fees, cost_amount,
    ads_spend, extra_expenses,
    tax_rate_pct,
    (ppvz_for_pay - cost_amount)                             AS gross_profit,
    pre_tax_profit,
    GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0       AS tax_amount,
    pre_tax_profit
      - GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0   AS net_profit,
    pre_tax_profit
      - GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0   AS profit,
    sales_count, returns_count
FROM joined
ORDER BY month DESC;
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

# ═══════════════════════════════════════════════════════════════
# Finance-based queries (mart.finance_daily — full cost breakdown)
# ═══════════════════════════════════════════════════════════════

# ── FIN Weekly: ISO-week financial summary ──────────────────────
# Налог считается аддитивно на уровне дня (row-level), а затем
# суммируется по неделе. Это важно для ISO-недель, пересекающих
# границу смены ставки (напр., неделя 2026-W01 содержит Dec 29-31
# 2025 со ставкой 0% и Jan 1-4 со ставкой 6%). Аддитивный подход
# даёт корректный налог на таких переходных неделях, иначе
# MAX(rate) на уровне недели завышал бы налог.
FIN_WEEKLY_QUERY = """
WITH detail AS (
    SELECT
        f.report_date,
        f.nm_id,
        f.sales_count,
        f.returns_count,
        f.sales_amount,
        f.returns_amount,
        f.retail_amount,
        f.ppvz_for_pay,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        COALESCE(cr.unit_cost, 0)
            * (f.sales_count - f.returns_count) AS cost_amount,
        COALESCE(tx.tax_rate_percent, 0)        AS tax_rate_pct,
        -- Pre-tax row (xlsx-методология): вычитаем deduction_amount,
        -- а ads_spend из Promotion API — нет (он уже включён в deduction).
        f.ppvz_for_pay
            - f.logistics_amount - f.storage_amount
            - f.penalty_amount - f.acceptance_amount
            - f.deduction_amount
            + f.additional_payment_amount
            - COALESCE(cr.unit_cost, 0)
              * (f.sales_count - f.returns_count) AS pre_tax_row,
        COALESCE(tx.tax_rate_percent, 0) / 100.0
            * (f.ppvz_for_pay
               - f.logistics_amount - f.storage_amount
               - f.penalty_amount - f.acceptance_amount
               - f.deduction_amount
               + f.additional_payment_amount
               - COALESCE(cr.unit_cost, 0)
                 * (f.sales_count - f.returns_count)) AS tax_row
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    LEFT JOIN LATERAL (
        SELECT t.tax_rate_percent FROM dict.tax_reference t
        WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
        ORDER BY t.valid_from DESC LIMIT 1
    ) tx ON true
    WHERE f.report_date BETWEEN :d_from AND :d_to
),
ads_weekly AS (
    SELECT TO_CHAR(ads_date, 'IYYY-IW') AS year_week,
           SUM(spend_amount)            AS ads_spend
    FROM mart.ads_daily
    WHERE ads_date BETWEEN :d_from AND :d_to
    GROUP BY 1
),
extra_weekly AS (
    SELECT TO_CHAR(expense_date, 'IYYY-IW') AS year_week,
           SUM(amount)                      AS extra_amount
    FROM dict.extra_expenses
    WHERE expense_date BETWEEN :d_from AND :d_to
    GROUP BY 1
),
weekly AS (
    SELECT
        TO_CHAR(report_date, 'IYYY-IW')                   AS year_week,
        MIN(report_date)                                   AS week_start,
        MAX(report_date)                                   AS week_end,
        SUM(sales_count)                                   AS sales_count,
        SUM(returns_count)                                 AS returns_count,
        SUM(sales_amount)                                  AS sales_amount,
        SUM(returns_amount)                                AS returns_amount,
        SUM(sales_amount - returns_amount)                 AS realization_pre_spp,
        SUM(retail_amount)                                 AS retail_amount,
        SUM(ppvz_for_pay)                                  AS ppvz_for_pay,
        SUM(commission_amount)                             AS commission,
        SUM(logistics_amount)                              AS logistics,
        SUM(storage_amount)                                AS storage,
        SUM(penalty_amount)                                AS penalty,
        SUM(acceptance_amount)                             AS acceptance,
        SUM(acquiring_amount)                              AS acquiring,
        SUM(deduction_amount)                              AS deduction,
        SUM(additional_payment_amount)                     AS additional_payment,
        SUM(commission_amount + logistics_amount + storage_amount
            + penalty_amount + acceptance_amount + acquiring_amount
            + deduction_amount - additional_payment_amount) AS total_wb_fees,
        SUM(cost_amount)                                   AS cost_amount,
        SUM(pre_tax_row)                                   AS pre_tax_pre_extras,
        SUM(tax_row)                                       AS tax_pre_extras
    FROM detail
    GROUP BY TO_CHAR(report_date, 'IYYY-IW')
)
SELECT
    w.year_week,
    w.week_start,
    w.week_end,
    w.sales_count, w.returns_count,
    w.sales_amount, w.returns_amount,
    w.realization_pre_spp, w.retail_amount,
    w.ppvz_for_pay, w.commission, w.logistics, w.storage, w.penalty,
    w.acceptance, w.acquiring, w.deduction, w.additional_payment,
    w.total_wb_fees, w.cost_amount,
    COALESCE(a.ads_spend, 0)    AS ads_spend,
    COALESCE(e.extra_amount, 0) AS extra_expenses,
    -- ads_spend НЕ вычитаем повторно — он уже включён в deduction_amount
    -- внутри pre_tax_row (xlsx-методология).
    (w.pre_tax_pre_extras
        - COALESCE(e.extra_amount, 0))                       AS pre_tax_profit,
    -- tax_amount per RASK: tax_rate × max(operating_profit, 0).
    GREATEST(
        w.pre_tax_pre_extras
            - COALESCE(e.extra_amount, 0), 0
    ) * (
        SELECT COALESCE(MAX(t.tax_rate_percent), 0) FROM dict.tax_reference t
        WHERE w.week_end BETWEEN t.valid_from AND t.valid_to
    ) / 100.0                                                AS tax_amount,
    (w.pre_tax_pre_extras
        - COALESCE(e.extra_amount, 0))
        - GREATEST(
            w.pre_tax_pre_extras
                - COALESCE(e.extra_amount, 0), 0
        ) * (
            SELECT COALESCE(MAX(t.tax_rate_percent), 0) FROM dict.tax_reference t
            WHERE w.week_end BETWEEN t.valid_from AND t.valid_to
        ) / 100.0                                            AS profit
FROM weekly w
LEFT JOIN ads_weekly a   ON a.year_week = w.year_week
LEFT JOIN extra_weekly e ON e.year_week = w.year_week
ORDER BY w.year_week DESC;
"""

# ── FIN Profit: daily profit report by article ──────────────────
# Возвращает pre_tax_profit и tax_rate_pct — tax считается на уровне
# агрегации (месяц/артикул/период) в Python, чтобы не инфлировать его
# per-row GREATEST'ами. Поле tax_amount здесь = rate × pre_tax_row
# (может быть отрицательным) и НЕ должно суммироваться без клиппинга.
FIN_PROFIT_QUERY = """
SELECT
    f.report_date,
    f.nm_id,
    f.supplier_article,
    f.subject,
    f.brand,
    f.sales_count,
    f.returns_count,
    f.sales_amount,
    f.returns_amount,
    f.retail_amount,
    f.ppvz_for_pay,
    f.commission_amount,
    f.logistics_amount,
    f.storage_amount,
    f.penalty_amount,
    f.acceptance_amount,
    f.acquiring_amount,
    f.deduction_amount,
    f.additional_payment_amount,
    f.commission_amount + f.logistics_amount + f.storage_amount
        + f.penalty_amount + f.acceptance_amount + f.acquiring_amount
        + f.deduction_amount - f.additional_payment_amount   AS total_wb_fees,
    -- Себестоимость по нетто-количеству (RASK ОПИУ).
    COALESCE(cr.unit_cost, 0)
        * (f.sales_count - f.returns_count)                  AS cost_amount,
    COALESCE(ad.spend_amount, 0)                             AS ads_spend,
    COALESCE(tx.tax_rate_percent, 0)                         AS tax_rate_pct,
    -- Pre-tax по дню × артикулу (xlsx-методология):
    --   deduction_amount вычитаем (полные удержания WB);
    --   ads_spend из Promotion API НЕ вычитаем повторно (входит в deduction).
    f.ppvz_for_pay
        - f.logistics_amount - f.storage_amount
        - f.penalty_amount - f.acceptance_amount
        - f.deduction_amount
        + f.additional_payment_amount
        - COALESCE(cr.unit_cost, 0)
          * (f.sales_count - f.returns_count)                 AS pre_tax_profit,
    COALESCE(tx.tax_rate_percent, 0) / 100.0
        * (f.ppvz_for_pay
           - f.logistics_amount - f.storage_amount
           - f.penalty_amount - f.acceptance_amount
           - f.deduction_amount
           + f.additional_payment_amount
           - COALESCE(cr.unit_cost, 0)
             * (f.sales_count - f.returns_count)
          )                                                   AS tax_amount,
    (1 - COALESCE(tx.tax_rate_percent, 0) / 100.0)
        * (f.ppvz_for_pay
           - f.logistics_amount - f.storage_amount
           - f.penalty_amount - f.acceptance_amount
           - f.deduction_amount
           + f.additional_payment_amount
           - COALESCE(cr.unit_cost, 0)
             * (f.sales_count - f.returns_count)
          )                                                   AS profit
FROM mart.finance_daily f
LEFT JOIN LATERAL (
    SELECT c.unit_cost FROM dict.cost_reference c
    WHERE c.nm_id = f.nm_id
      AND f.report_date BETWEEN c.valid_from AND c.valid_to
    ORDER BY c.valid_from DESC LIMIT 1
) cr ON true
LEFT JOIN LATERAL (
    SELECT t.tax_rate_percent FROM dict.tax_reference t
    WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
    ORDER BY t.valid_from DESC LIMIT 1
) tx ON true
LEFT JOIN LATERAL (
    SELECT SUM(spend_amount) AS spend_amount FROM mart.ads_daily a
    WHERE a.nm_id = f.nm_id AND a.ads_date = f.report_date
) ad ON true
WHERE f.report_date BETWEEN :d_from AND :d_to
ORDER BY f.report_date DESC, f.ppvz_for_pay DESC;
"""

# ── FIN Statutory: monthly financial summary ────────────────────
# Формула выровнена с ОПИУ RASKа:
#   • Себестоимость считаем по НЕТТО-количеству (sold − returns), иначе
#     завышаем расход на возвращённых единицах товара (RASK так же).
#   • Добавляем расходы на внутреннюю рекламу (mart.ads_daily) и
#     прочие операционные расходы / отзывы (dict.extra_expenses).
#   • Налог = ставка × GREATEST(операционная_прибыль, 0); рассчитывается
#     на агрегатном уровне (по месяцу), чтобы убыточные строки не
#     обнулялись раньше времени и не завышали налог.
FIN_STATUTORY_QUERY = """
WITH detail AS (
    SELECT
        f.report_date,
        f.sales_count,
        f.returns_count,
        f.sales_amount,
        f.returns_amount,
        f.ppvz_for_pay,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        -- Себестоимость по нетто-количеству (sold − returns); знак сохраняем,
        -- чтобы дни с одними возвратами уменьшали итоговую себестоимость
        -- (возврат товара возвращает себестоимость на склад).
        COALESCE(cr.unit_cost, 0)
            * (f.sales_count - f.returns_count) AS cost_amount,
        COALESCE(tx.tax_rate_percent, 0)        AS tax_rate_pct
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    LEFT JOIN LATERAL (
        SELECT t.tax_rate_percent FROM dict.tax_reference t
        WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
        ORDER BY t.valid_from DESC LIMIT 1
    ) tx ON true
),
ads_monthly AS (
    SELECT date_trunc('month', ads_date)::date AS month,
           SUM(spend_amount)                   AS ads_spend
    FROM mart.ads_daily
    GROUP BY 1
),
extra_monthly AS (
    SELECT date_trunc('month', expense_date)::date AS month,
           SUM(amount)                             AS extra_amount
    FROM dict.extra_expenses
    GROUP BY 1
),
monthly AS (
    SELECT
        date_trunc('month', report_date)::date              AS month,
        SUM(sales_count)                                    AS sales_count,
        SUM(returns_count)                                  AS returns_count,
        SUM(sales_amount)                                   AS sales_amount,
        SUM(returns_amount)                                 AS returns_amount,
        SUM(ppvz_for_pay)                                   AS ppvz_for_pay,
        SUM(commission_amount)                              AS commission,
        SUM(logistics_amount)                               AS logistics,
        SUM(storage_amount)                                 AS storage,
        SUM(penalty_amount)                                 AS penalty,
        SUM(acceptance_amount)                              AS acceptance,
        SUM(acquiring_amount)                               AS acquiring,
        SUM(deduction_amount)                               AS deduction,
        SUM(additional_payment_amount)                      AS additional_payment,
        SUM(commission_amount + logistics_amount + storage_amount
            + penalty_amount + acceptance_amount + acquiring_amount
            + deduction_amount - additional_payment_amount) AS total_wb_fees,
        SUM(cost_amount)                                    AS cost_amount,
        MAX(tax_rate_pct)                                   AS tax_rate_pct
    FROM detail
    GROUP BY 1
),
joined AS (
    SELECT
        m.*,
        COALESCE(a.ads_spend, 0)    AS ads_spend,
        COALESCE(e.extra_amount, 0) AS extra_expenses,
        -- EBITDA / Валовая маржа по методологии RASK ОПИУ (из xlsx):
        --   К перечислению (ppvz_for_pay)
        -- − Логистика, − Хранение, − Штрафы, − Платная приемка
        -- − deduction_amount (внутренняя реклама + отзывы + прочие
        --   удержания — один бакет по Finance API)
        -- + Доп. платежи
        -- − Себестоимость
        -- − Прочие операционные расходы (dict.extra_expenses)
        -- Эквайринг НЕ вычитается (xlsx-методология не учитывает).
        -- ads_spend из Promotion API хранится отдельно для дашборда,
        -- но НЕ вычитается повторно (включён в deduction_amount).
        (m.ppvz_for_pay
            - m.logistics - m.storage
            - m.penalty - m.acceptance
            - m.deduction
            + m.additional_payment
            - m.cost_amount
            - COALESCE(e.extra_amount, 0)
        )                            AS pre_tax_profit
    FROM monthly m
    LEFT JOIN ads_monthly a   ON a.month = m.month
    LEFT JOIN extra_monthly e ON e.month = m.month
)
SELECT
    month,
    sales_count, returns_count, sales_amount, returns_amount,
    ppvz_for_pay, commission, logistics, storage, penalty,
    acceptance, acquiring, deduction, additional_payment,
    total_wb_fees, cost_amount,
    ads_spend, extra_expenses,
    pre_tax_profit,
    GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0      AS tax_amount,
    pre_tax_profit
      - GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0  AS profit,
    pre_tax_profit
      - GREATEST(pre_tax_profit, 0) * tax_rate_pct / 100.0  AS operating_profit
FROM joined
ORDER BY month DESC;
"""

# ── FIN Article: per-article financial summary ──────────────────
# Налог считается аддитивно на уровне строки (день×артикул) и затем
# суммируется по артикулу. Это важно для мультигодовых диапазонов,
# где артикул имел продажи и до 2026 (ставка 0%), и с 2026-01-01
# (ставка 6%). MAX(rate) на уровне артикула применял бы 6% ко всему
# pre_tax_profit, включая период с rate=0, что завышает налог.
FIN_ARTICLE_QUERY = """
WITH detail AS (
    SELECT
        f.nm_id,
        f.supplier_article,
        f.subject,
        f.brand,
        f.report_date,
        f.sales_count,
        f.returns_count,
        f.sales_amount,
        f.returns_amount,
        f.ppvz_for_pay,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        COALESCE(cr.unit_cost, 0)
            * (f.sales_count - f.returns_count) AS cost_amount,
        COALESCE(tx.tax_rate_percent, 0)        AS tax_rate_pct,
        -- Pre-tax по строке (xlsx-методология):
        --   ppvz − логистика − хранение − штрафы − приёмка
        --   − deduction_amount (внутрь уже входит реклама и прочие удержания)
        --   + доп. платежи − себестоимость.
        -- extra_expenses прибавляются/вычитаются на уровне артикула
        -- в финальном SELECT (чтобы не "дробились" по дням).
        f.ppvz_for_pay
            - f.logistics_amount - f.storage_amount
            - f.penalty_amount - f.acceptance_amount
            - f.deduction_amount
            + f.additional_payment_amount
            - COALESCE(cr.unit_cost, 0)
              * (f.sales_count - f.returns_count) AS pre_tax_row,
        COALESCE(tx.tax_rate_percent, 0) / 100.0
            * (f.ppvz_for_pay
               - f.logistics_amount - f.storage_amount
               - f.penalty_amount - f.acceptance_amount
               - f.deduction_amount
               + f.additional_payment_amount
               - COALESCE(cr.unit_cost, 0)
                 * (f.sales_count - f.returns_count)) AS tax_row
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    LEFT JOIN LATERAL (
        SELECT t.tax_rate_percent FROM dict.tax_reference t
        WHERE f.report_date BETWEEN t.valid_from AND t.valid_to
        ORDER BY t.valid_from DESC LIMIT 1
    ) tx ON true
    WHERE f.report_date BETWEEN :d_from AND :d_to
),
ads_per_article AS (
    SELECT nm_id, SUM(spend_amount) AS ads_spend
    FROM mart.ads_daily
    WHERE ads_date BETWEEN :d_from AND :d_to
    GROUP BY nm_id
),
extra_per_article AS (
    -- Подтягиваем extra_expenses, привязанные к конкретному nm_id
    -- (если nm_id не указан — расход распределяется в общем итоге).
    SELECT nm_id, SUM(amount) AS extra_amount
    FROM dict.extra_expenses
    WHERE expense_date BETWEEN :d_from AND :d_to
      AND nm_id IS NOT NULL
    GROUP BY nm_id
),
-- Выбираем «последнюю» версию supplier_article для nm_id в периоде —
-- если артикул переименовался, агрегируем по nm_id, чтобы реклама и
-- extras (заданные на nm_id) не дублировались в разных группах.
latest_article AS (
    SELECT DISTINCT ON (nm_id)
        nm_id, supplier_article, subject, brand
    FROM mart.finance_daily
    WHERE report_date BETWEEN :d_from AND :d_to
    ORDER BY nm_id, report_date DESC
)
SELECT
    d.nm_id,
    la.supplier_article                                   AS supplier_article,
    MAX(la.subject)                                       AS subject,
    MAX(la.brand)                                         AS brand,
    SUM(d.sales_count)                                    AS sales_count,
    SUM(d.returns_count)                                  AS returns_count,
    SUM(d.sales_amount)                                   AS sales_amount,
    SUM(d.returns_amount)                                 AS returns_amount,
    SUM(d.ppvz_for_pay)                                   AS ppvz_for_pay,
    SUM(d.commission_amount)                              AS commission,
    SUM(d.logistics_amount)                               AS logistics,
    SUM(d.storage_amount)                                 AS storage,
    SUM(d.penalty_amount)                                 AS penalty,
    SUM(d.commission_amount + d.logistics_amount + d.storage_amount
        + d.penalty_amount + d.acceptance_amount + d.acquiring_amount
        + d.deduction_amount - d.additional_payment_amount) AS total_wb_fees,
    SUM(d.cost_amount)                                    AS cost_amount,
    COALESCE(MAX(a.ads_spend), 0)                         AS ads_spend,
    COALESCE(MAX(e.extra_amount), 0)                      AS extra_expenses,
    MAX(d.tax_rate_pct)                                   AS tax_rate_pct,
    -- pre_tax по артикулу: pre_tax_row уже учёл deduction_amount
    -- (=внутренняя реклама+отзывы+прочие). ads_spend из Promotion API
    -- НЕ вычитаем повторно. Убавляем только extra_expenses (справочник
    -- "Прочие операционные расходы" — оффлайн-траты типа курьер, образцы).
    (SUM(d.pre_tax_row)
        - COALESCE(MAX(e.extra_amount), 0))                AS pre_tax_profit,
    (SUM(d.tax_row)
        - COALESCE(MAX(e.extra_amount), 0)
          * MAX(d.tax_rate_pct) / 100.0)                   AS tax_amount,
    SUM(d.pre_tax_row - d.tax_row)
        - COALESCE(MAX(e.extra_amount), 0)
          * (1 - MAX(d.tax_rate_pct) / 100.0)              AS profit
FROM detail d
LEFT JOIN latest_article   la ON la.nm_id = d.nm_id
LEFT JOIN ads_per_article   a ON a.nm_id = d.nm_id
LEFT JOIN extra_per_article e ON e.nm_id = d.nm_id
GROUP BY d.nm_id, la.supplier_article
ORDER BY ppvz_for_pay DESC;
"""

# ── FIN Promo Baseline: 30-day unit economics from finance_daily ─
FIN_PROMO_BASELINE_QUERY = """
WITH detail AS (
    SELECT
        f.nm_id,
        f.supplier_article,
        f.subject,
        f.brand,
        f.sales_count,
        f.returns_count,
        f.ppvz_for_pay,
        f.commission_amount,
        f.logistics_amount,
        f.storage_amount,
        f.penalty_amount,
        f.acceptance_amount,
        f.acquiring_amount,
        f.deduction_amount,
        f.additional_payment_amount,
        -- Net qty (sold − returns), без учёта рекламы (она идёт на уровне
        -- агрегата артикула в fin_agg ниже).
        COALESCE(cr.unit_cost, 0)
            * (f.sales_count - f.returns_count)   AS cost_amount,
        COALESCE(cr.unit_cost, 0)                  AS unit_cost,
        f.ppvz_for_pay
            - f.logistics_amount - f.storage_amount
            - f.penalty_amount - f.acceptance_amount
            + f.additional_payment_amount
            - COALESCE(cr.unit_cost, 0)
              * (f.sales_count - f.returns_count) AS profit_before_tax
    FROM mart.finance_daily f
    LEFT JOIN LATERAL (
        SELECT c.unit_cost FROM dict.cost_reference c
        WHERE c.nm_id = f.nm_id
          AND f.report_date BETWEEN c.valid_from AND c.valid_to
        ORDER BY c.valid_from DESC LIMIT 1
    ) cr ON true
    WHERE f.report_date >= CURRENT_DATE - INTERVAL '30 days'
),
fin_agg AS (
    SELECT
        nm_id,
        supplier_article,
        MAX(subject)                                                    AS subject,
        MAX(brand)                                                      AS brand,
        SUM(sales_count)                                                AS sales_count,
        SUM(returns_count)                                              AS returns_count,
        SUM(ppvz_for_pay)                                               AS ppvz_for_pay,
        SUM(cost_amount)                                                AS cost_amount,
        SUM(profit_before_tax)                                          AS total_profit_30d,
        ROUND(SUM(commission_amount) / NULLIF(SUM(sales_count), 0), 2)  AS commission_per_unit,
        ROUND(SUM(logistics_amount) / NULLIF(SUM(sales_count), 0), 2)   AS logistics_per_unit,
        ROUND(SUM(cost_amount) / NULLIF(SUM(sales_count), 0), 2)        AS cost_per_unit,
        ROUND(SUM(profit_before_tax) / NULLIF(SUM(sales_count), 0), 2)  AS profit_per_unit,
        ROUND(SUM(ppvz_for_pay) / NULLIF(SUM(sales_count), 0), 2)       AS payout_per_unit,
        ROUND(
            SUM(commission_amount + logistics_amount + storage_amount
                + penalty_amount + acceptance_amount + acquiring_amount
                + deduction_amount - additional_payment_amount)
            / NULLIF(SUM(sales_count), 0), 2
        )                                                               AS wb_fees_per_unit
    FROM detail
    GROUP BY nm_id, supplier_article
),
ord_agg AS (
    SELECT
        o.nm_id, o.supplier_article,
        ROUND(AVG(o.orders_count)::NUMERIC, 2)          AS avg_orders_day,
        COALESCE(ROUND(AVG(s.avg_price_before_spp)::NUMERIC, 0), 0) AS avg_price_before_spp,
        COALESCE(ROUND(AVG(s.avg_price_after_spp)::NUMERIC, 0), 0)  AS avg_price_after_spp,
        COALESCE(ROUND(AVG(s.avg_spp)::NUMERIC, 1), 0)              AS avg_spp_pct,
        CASE WHEN SUM(o.orders_count) > 0
            THEN ROUND(COALESCE(SUM(s.sales_count), 0)::NUMERIC
                       / SUM(o.orders_count) * 100, 1)
            ELSE 0
        END AS buyout_pct
    FROM mart.orders_daily o
    LEFT JOIN mart.sales_daily s
        ON o.order_date = s.sales_date AND o.nm_id = s.nm_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY o.nm_id, o.supplier_article
)
SELECT
    fa.nm_id,
    fa.supplier_article,
    fa.subject,
    fa.brand,
    COALESCE(oa.avg_orders_day, 0)          AS avg_orders_day,
    fa.sales_count,
    COALESCE(oa.avg_price_before_spp, 0)    AS avg_price_before_spp,
    COALESCE(oa.avg_price_after_spp, 0)     AS avg_price_after_spp,
    COALESCE(oa.avg_spp_pct, 0)             AS avg_spp_pct,
    COALESCE(oa.buyout_pct, 0)              AS buyout_pct,
    fa.commission_per_unit,
    fa.logistics_per_unit,
    fa.cost_per_unit,
    fa.profit_per_unit,
    fa.payout_per_unit,
    fa.wb_fees_per_unit,
    fa.total_profit_30d,
    COALESCE(st.qty, 0)                     AS current_stock
FROM fin_agg fa
LEFT JOIN ord_agg oa
    ON fa.nm_id = oa.nm_id
LEFT JOIN (
    SELECT nm_id, SUM(quantity_full) AS qty
    FROM mart.v_stocks_current GROUP BY nm_id
) st ON fa.nm_id = st.nm_id
ORDER BY fa.total_profit_30d DESC;
"""
