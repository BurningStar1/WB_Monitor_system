-- ============================================================
-- Схема _diagram — ТОЛЬКО ДЛЯ ВИЗУАЛИЗАЦИИ ER-диаграммы
-- Не используется в приложении. Содержит все таблицы БД
-- с FK-связями для построения единой модели в DBeaver.
-- ============================================================

DROP SCHEMA IF EXISTS _diagram CASCADE;
CREATE SCHEMA _diagram;

-- ======================== APP ======================== --

CREATE TABLE _diagram.users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.wb_accounts (
    id BIGSERIAL PRIMARY KEY,
    account_name TEXT NOT NULL,
    api_token TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.user_accounts (
    user_id BIGINT NOT NULL REFERENCES _diagram.users(id),
    account_id BIGINT NOT NULL REFERENCES _diagram.wb_accounts(id),
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, account_id)
);

-- ======================== RAW ======================== --

CREATE TABLE _diagram.wb_api_payloads (
    id BIGSERIAL PRIMARY KEY,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_params JSONB NOT NULL,
    raw_payload JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ======================== STG ======================== --

CREATE TABLE _diagram.wb_orders (
    srid TEXT PRIMARY KEY,
    date TIMESTAMPTZ,
    last_change_date TIMESTAMPTZ,
    supplier_article TEXT,
    tech_size TEXT,
    barcode TEXT,
    total_price NUMERIC(14, 2),
    discount_percent NUMERIC(6, 2),
    warehouse_name TEXT,
    oblast TEXT,
    income_id BIGINT,
    odid BIGINT,
    nm_id BIGINT,
    subject TEXT,
    category TEXT,
    brand TEXT,
    is_cancel BOOLEAN DEFAULT FALSE,
    cancel_dt TIMESTAMPTZ,
    source_payload_id BIGINT REFERENCES _diagram.wb_api_payloads(id),
    source_loaded_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.wb_sales (
    sale_id TEXT PRIMARY KEY,
    date TIMESTAMPTZ,
    last_change_date TIMESTAMPTZ,
    supplier_article TEXT,
    tech_size TEXT,
    barcode TEXT,
    total_price NUMERIC(14, 2),
    discount_percent NUMERIC(6, 2),
    is_supply BOOLEAN DEFAULT FALSE,
    is_realization BOOLEAN DEFAULT FALSE,
    promo_code_discount NUMERIC(14, 2),
    warehouse_name TEXT,
    country_name TEXT,
    oblast_okrug_name TEXT,
    region_name TEXT,
    income_id BIGINT,
    odid BIGINT,
    spp NUMERIC(8, 2),
    for_pay NUMERIC(14, 2),
    finished_price NUMERIC(14, 2),
    price_with_disc NUMERIC(14, 2),
    nm_id BIGINT,
    subject TEXT,
    category TEXT,
    brand TEXT,
    is_storno BOOLEAN DEFAULT FALSE,
    source_payload_id BIGINT REFERENCES _diagram.wb_api_payloads(id),
    source_loaded_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.wb_stocks (
    id BIGSERIAL PRIMARY KEY,
    warehouse_name TEXT,
    supplier_article TEXT,
    nm_id BIGINT,
    barcode TEXT,
    quantity INT,
    in_way_to_client INT,
    in_way_from_client INT,
    quantity_full INT,
    category TEXT,
    subject TEXT,
    brand TEXT,
    tech_size TEXT,
    price NUMERIC(14, 2),
    discount NUMERIC(6, 2),
    is_supply BOOLEAN DEFAULT FALSE,
    is_realization BOOLEAN DEFAULT FALSE,
    source_payload_id BIGINT REFERENCES _diagram.wb_api_payloads(id),
    source_loaded_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ======================== DICT ======================== --

CREATE TABLE _diagram.cost_reference (
    id BIGSERIAL PRIMARY KEY,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT,
    unit_cost NUMERIC(14, 2) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT DATE '2999-12-31',
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.extra_expenses (
    id BIGSERIAL PRIMARY KEY,
    expense_date DATE NOT NULL,
    expense_category TEXT NOT NULL,
    amount NUMERIC(14, 2) NOT NULL,
    comment TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE _diagram.tax_reference (
    id BIGSERIAL PRIMARY KEY,
    tax_name TEXT NOT NULL,
    tax_rate_percent NUMERIC(8, 4) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT DATE '2999-12-31',
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ======================== MART ======================== --

CREATE TABLE _diagram.orders_daily (
    order_date DATE NOT NULL,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT NOT NULL,
    subject TEXT,
    brand TEXT,
    orders_count BIGINT NOT NULL DEFAULT 0,
    orders_amount NUMERIC(14, 2) NOT NULL DEFAULT 0,
    cancelled_count BIGINT NOT NULL DEFAULT 0,
    avg_price NUMERIC(14, 2) DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (order_date, nm_id, supplier_article)
);

CREATE TABLE _diagram.sales_daily (
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

CREATE TABLE _diagram.stocks_snapshot (
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

CREATE TABLE _diagram.ads_daily (
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

-- ======================== КОММЕНТАРИИ К ТАБЛИЦАМ ======================== --
-- (DBeaver отображает комментарии в диаграмме)

COMMENT ON SCHEMA _diagram IS 'Визуализация полной модели БД (не используется в приложении)';

COMMENT ON TABLE _diagram.users IS '[APP] Пользователи (сотрудники)';
COMMENT ON TABLE _diagram.wb_accounts IS '[APP] Аккаунты WB (кабинеты продавца)';
COMMENT ON TABLE _diagram.user_accounts IS '[APP] Связь пользователь ↔ аккаунт (M:N)';

COMMENT ON TABLE _diagram.wb_api_payloads IS '[RAW] Сырые JSON-ответы WB API';

COMMENT ON TABLE _diagram.wb_orders IS '[STG] Заказы (нормализовано)';
COMMENT ON TABLE _diagram.wb_sales IS '[STG] Продажи (нормализовано)';
COMMENT ON TABLE _diagram.wb_stocks IS '[STG] Остатки (нормализовано)';

COMMENT ON TABLE _diagram.cost_reference IS '[DICT] Справочник себестоимости';
COMMENT ON TABLE _diagram.extra_expenses IS '[DICT] Дополнительные расходы';
COMMENT ON TABLE _diagram.tax_reference IS '[DICT] Справочник налогов';

COMMENT ON TABLE _diagram.orders_daily IS '[MART] Витрина заказов (день × артикул)';
COMMENT ON TABLE _diagram.sales_daily IS '[MART] Витрина продаж и финансов (день × артикул)';
COMMENT ON TABLE _diagram.stocks_snapshot IS '[MART] Витрина остатков (день × артикул × склад)';
COMMENT ON TABLE _diagram.ads_daily IS '[MART] Витрина рекламы (день × артикул × кампания)';
