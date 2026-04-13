-- --- Представления для отчётности --- --

-- --- Недельная агрегация продаж (для еженедельного дашборда) --- --
CREATE OR REPLACE VIEW mart.v_sales_weekly AS
SELECT
    to_char(sales_date, 'IYYY-IW') AS year_week,
    MIN(sales_date) AS week_start,
    MAX(sales_date) AS week_end,
    SUM(orders_count) AS orders_count,
    SUM(sales_count) AS sales_count,
    SUM(returns_count) AS returns_count,
    SUM(gross_revenue) AS gross_revenue,
    SUM(net_revenue) AS net_revenue,
    SUM(commission_amount) AS commission_amount,
    SUM(cost_amount) AS cost_amount,
    SUM(extra_expenses_amount) AS extra_expenses_amount,
    SUM(tax_amount) AS tax_amount,
    SUM(profit_amount) AS profit_amount,
    SUM(operating_profit_amount) AS operating_profit_amount
FROM mart.sales_daily
GROUP BY 1;

-- --- Сводка по артикулам (для отчёта по артикулам) --- --
CREATE OR REPLACE VIEW mart.v_sales_by_article AS
SELECT
    nm_id,
    supplier_article,
    MAX(subject) AS subject,
    MAX(brand) AS brand,
    SUM(orders_count) AS orders_count,
    SUM(sales_count) AS sales_count,
    SUM(returns_count) AS returns_count,
    SUM(gross_revenue) AS gross_revenue,
    SUM(net_revenue) AS net_revenue,
    SUM(commission_amount) AS commission_amount,
    SUM(cost_amount) AS cost_amount,
    SUM(profit_amount) AS profit_amount,
    SUM(operating_profit_amount) AS operating_profit_amount,
    CASE WHEN SUM(orders_count) > 0
        THEN ROUND(SUM(sales_count)::NUMERIC / SUM(orders_count) * 100, 1)
        ELSE 0
    END AS buyout_pct
FROM mart.sales_daily
GROUP BY nm_id, supplier_article;

-- --- Текущие остатки (последний срез по складам) --- --
CREATE OR REPLACE VIEW mart.v_stocks_current AS
SELECT
    ss.nm_id,
    ss.supplier_article,
    ss.warehouse_name,
    ss.subject,
    ss.brand,
    ss.quantity,
    ss.in_way_to_client,
    ss.in_way_from_client,
    ss.quantity_full,
    ss.price,
    ss.discount,
    ss.updated_at
FROM mart.stocks_snapshot ss
INNER JOIN (
    SELECT nm_id, warehouse_name, MAX(snapshot_date) AS max_date
    FROM mart.stocks_snapshot
    GROUP BY nm_id, warehouse_name
) latest ON ss.nm_id = latest.nm_id
    AND ss.warehouse_name = latest.warehouse_name
    AND ss.snapshot_date = latest.max_date;

-- --- Отчёт по прибыли (для финансового отчёта) --- --
CREATE OR REPLACE VIEW mart.v_profit_report AS
SELECT
    sales_date,
    nm_id,
    supplier_article,
    subject,
    brand,
    net_revenue,
    commission_amount,
    cost_amount,
    extra_expenses_amount,
    tax_amount,
    profit_amount,
    operating_profit_amount,
    CASE WHEN net_revenue > 0
        THEN ROUND(profit_amount / net_revenue * 100, 1)
        ELSE 0
    END AS margin_pct,
    CASE WHEN (cost_amount + commission_amount) > 0
        THEN ROUND(profit_amount / (cost_amount + commission_amount) * 100, 1)
        ELSE 0
    END AS roi_pct
FROM mart.sales_daily;

-- --- Месячный отчёт (для регламентной отчётности) --- --
CREATE OR REPLACE VIEW mart.v_statutory_period_report AS
SELECT
    date_trunc('month', sales_date)::date AS period_month,
    SUM(orders_count) AS orders_count,
    SUM(sales_count) AS sales_count,
    SUM(returns_count) AS returns_count,
    SUM(net_revenue) AS net_revenue,
    SUM(commission_amount) AS commission_amount,
    SUM(cost_amount) AS cost_amount,
    SUM(extra_expenses_amount) AS extra_expenses_amount,
    SUM(tax_amount) AS tax_amount,
    SUM(profit_amount) AS profit_amount,
    SUM(operating_profit_amount) AS operating_profit_amount
FROM mart.sales_daily
GROUP BY 1;

-- --- ABC-анализ (по выручке) --- --
CREATE OR REPLACE VIEW mart.v_abc_analysis AS
WITH base AS (
    SELECT
        nm_id,
        supplier_article,
        MAX(subject) AS subject,
        MAX(brand) AS brand,
        SUM(net_revenue) AS net_revenue,
        SUM(profit_amount) AS profit_amount,
        SUM(orders_count) AS orders_count
    FROM mart.sales_daily
    GROUP BY nm_id, supplier_article
), ranked AS (
    SELECT
        *,
        SUM(net_revenue) OVER (ORDER BY net_revenue DESC ROWS UNBOUNDED PRECEDING)
            / NULLIF(SUM(net_revenue) OVER (), 0) AS revenue_share_cum
    FROM base
)
SELECT
    nm_id,
    supplier_article,
    subject,
    brand,
    net_revenue,
    profit_amount,
    orders_count,
    revenue_share_cum,
    CASE
        WHEN revenue_share_cum <= 0.80 THEN 'A'
        WHEN revenue_share_cum <= 0.95 THEN 'B'
        ELSE 'C'
    END AS abc_class
FROM ranked;

-- --- Динамика заказов по дням (для дашборда — мини-графики) --- --
CREATE OR REPLACE VIEW mart.v_orders_daily_trend AS
SELECT
    order_date,
    SUM(orders_count) AS orders_count,
    SUM(orders_amount) AS orders_amount,
    SUM(cancelled_count) AS cancelled_count
FROM mart.orders_daily
GROUP BY order_date
ORDER BY order_date;

-- --- Рука на Пульсе: артикулы × дни (для тепловой карты) --- --
CREATE OR REPLACE VIEW mart.v_rnp_daily AS
SELECT
    o.order_date AS report_date,
    o.nm_id,
    o.supplier_article,
    o.subject,
    o.brand,
    o.orders_count,
    o.orders_amount,
    COALESCE(s.sales_count, 0) AS sales_count,
    COALESCE(s.returns_count, 0) AS returns_count,
    COALESCE(s.net_revenue, 0) AS net_revenue,
    COALESCE(s.profit_amount, 0) AS profit_amount,
    COALESCE(s.operating_profit_amount, 0) AS operating_profit_amount
FROM mart.orders_daily o
LEFT JOIN mart.sales_daily s
    ON o.order_date = s.sales_date
    AND o.nm_id = s.nm_id
    AND o.supplier_article = s.supplier_article;

-- --- Прогноз заказов: база для скользящей средней --- --
CREATE OR REPLACE VIEW mart.v_forecast_orders AS
SELECT
    o.order_date,
    o.nm_id,
    o.supplier_article,
    o.orders_count,
    AVG(o.orders_count) OVER (
        PARTITION BY o.nm_id, o.supplier_article
        ORDER BY o.order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS forecast_7d,
    AVG(o.orders_count) OVER (
        PARTITION BY o.nm_id, o.supplier_article
        ORDER BY o.order_date
        ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
    ) AS forecast_14d,
    sc.quantity_full AS current_stock,
    CASE
        WHEN AVG(o.orders_count) OVER (
            PARTITION BY o.nm_id, o.supplier_article
            ORDER BY o.order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) > 0
        THEN ROUND(
            COALESCE(sc.quantity_full, 0)::NUMERIC
            / AVG(o.orders_count) OVER (
                PARTITION BY o.nm_id, o.supplier_article
                ORDER BY o.order_date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 1
        )
        ELSE NULL
    END AS days_of_stock
FROM mart.orders_daily o
LEFT JOIN mart.v_stocks_current sc
    ON o.nm_id = sc.nm_id
    AND o.supplier_article = sc.supplier_article;

-- --- Остатки по складам с долями (для отчёта по складам) --- --
CREATE OR REPLACE VIEW mart.v_stocks_by_warehouse AS
SELECT
    warehouse_name,
    nm_id,
    supplier_article,
    subject,
    brand,
    quantity,
    quantity_full,
    CASE WHEN SUM(quantity_full) OVER () > 0
        THEN ROUND(quantity_full::NUMERIC / SUM(quantity_full) OVER () * 100, 2)
        ELSE 0
    END AS share_pct
FROM mart.v_stocks_current;

-- --- Сводка по артикулам с остатками (для отчёта по артикулам) --- --
CREATE OR REPLACE VIEW mart.v_article_report AS
SELECT
    sa.nm_id,
    sa.supplier_article,
    sa.subject,
    sa.brand,
    sa.orders_count,
    sa.sales_count,
    sa.returns_count,
    sa.buyout_pct,
    sa.gross_revenue,
    sa.net_revenue,
    sa.commission_amount,
    sa.profit_amount,
    sa.operating_profit_amount,
    COALESCE(st.total_quantity, 0) AS stock_quantity,
    COALESCE(st.total_quantity_full, 0) AS stock_quantity_full,
    CASE WHEN sa.orders_count > 0
        THEN ROUND(sa.orders_count::NUMERIC
            / NULLIF((SELECT MAX(sales_date) - MIN(sales_date) + 1 FROM mart.sales_daily), 0), 1)
        ELSE 0
    END AS orders_per_day
FROM mart.v_sales_by_article sa
LEFT JOIN (
    SELECT nm_id, supplier_article,
        SUM(quantity) AS total_quantity,
        SUM(quantity_full) AS total_quantity_full
    FROM mart.v_stocks_current
    GROUP BY nm_id, supplier_article
) st ON sa.nm_id = st.nm_id AND sa.supplier_article = st.supplier_article;
