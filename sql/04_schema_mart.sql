CREATE SCHEMA IF NOT EXISTS mart;

-- --- Витрина продаж и финансовых показателей --- --
CREATE TABLE IF NOT EXISTS mart.sales_daily (
    sales_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT NOT NULL,
    subject TEXT,
    brand TEXT,
    orders_count BIGINT NOT NULL DEFAULT 0,
    sales_count BIGINT NOT NULL DEFAULT 0,
    returns_count BIGINT NOT NULL DEFAULT 0,
    gross_revenue NUMERIC(14, 2) NOT NULL DEFAULT 0,
    net_revenue NUMERIC(14, 2) NOT NULL DEFAULT 0,
    commission_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    cost_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    extra_expenses_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    tax_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    profit_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    operating_profit_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    avg_spp NUMERIC(8, 2) DEFAULT 0,
    avg_price_before_spp NUMERIC(14, 2) DEFAULT 0,
    avg_price_after_spp NUMERIC(14, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sales_date, nm_id, supplier_article)
);

-- --- Витрина заказов --- --
CREATE TABLE IF NOT EXISTS mart.orders_daily (
    order_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT NOT NULL,
    subject TEXT,
    brand TEXT,
    orders_count BIGINT NOT NULL DEFAULT 0,
    orders_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    orders_amount_disc NUMERIC(14, 2) NOT NULL DEFAULT 0,
    cancelled_count BIGINT NOT NULL DEFAULT 0,
    avg_price NUMERIC(14, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (order_date, nm_id, supplier_article)
);

-- --- Витрина остатков --- --
CREATE TABLE IF NOT EXISTS mart.stocks_snapshot (
    snapshot_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT,
    warehouse_name TEXT NOT NULL,
    subject TEXT,
    brand TEXT,
    quantity INT NOT NULL DEFAULT 0,
    in_way_to_client INT NOT NULL DEFAULT 0,
    in_way_from_client INT NOT NULL DEFAULT 0,
    quantity_full INT NOT NULL DEFAULT 0,
    price NUMERIC(14, 2) DEFAULT 0,
    discount NUMERIC(6, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (snapshot_date, nm_id, warehouse_name)
);

-- --- Витрина финансовых отчётов --- --
CREATE TABLE IF NOT EXISTS mart.finance_daily (
    report_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT NOT NULL,
    subject TEXT,
    brand TEXT,
    sales_count BIGINT NOT NULL DEFAULT 0,
    returns_count BIGINT NOT NULL DEFAULT 0,
    sales_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    returns_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    retail_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    ppvz_for_pay NUMERIC(14, 2) NOT NULL DEFAULT 0,
    commission_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    logistics_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    storage_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    penalty_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    acceptance_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    acquiring_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    deduction_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    additional_payment_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (report_date, nm_id, supplier_article)
);

-- --- Витрина рекламы --- --
CREATE TABLE IF NOT EXISTS mart.ads_daily (
    ads_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL DEFAULT 0,
    supplier_article TEXT,
    views_count BIGINT NOT NULL DEFAULT 0,
    clicks_count BIGINT NOT NULL DEFAULT 0,
    ctr NUMERIC(8, 4) DEFAULT 0,
    cpc NUMERIC(14, 2) DEFAULT 0,
    orders_from_ads BIGINT NOT NULL DEFAULT 0,
    spend_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ads_date, nm_id, campaign_id)
);
