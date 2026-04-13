CREATE SCHEMA IF NOT EXISTS dict;

CREATE TABLE IF NOT EXISTS dict.cost_reference (
    id BIGSERIAL PRIMARY KEY,
    nm_id BIGINT NOT NULL,
    supplier_article TEXT,
    unit_cost NUMERIC(14, 2) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT DATE '2999-12-31',
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dict.extra_expenses (
    id BIGSERIAL PRIMARY KEY,
    expense_date DATE NOT NULL,
    expense_category TEXT NOT NULL,
    amount NUMERIC(14, 2) NOT NULL,
    comment TEXT,
    nm_id BIGINT,              -- NULL = Нераспределённое (расход не привязан к артикулу)
    supplier_article TEXT,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dict.tax_reference (
    id BIGSERIAL PRIMARY KEY,
    tax_name TEXT NOT NULL,
    tax_rate_percent NUMERIC(8, 4) NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE NOT NULL DEFAULT DATE '2999-12-31',
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
