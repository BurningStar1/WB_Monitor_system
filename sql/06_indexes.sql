-- --- Индексы RAW --- --
CREATE INDEX IF NOT EXISTS ix_raw_payloads_endpoint_loaded_at
    ON raw.wb_api_payloads (endpoint, loaded_at);

-- --- Индексы STG --- --
CREATE INDEX IF NOT EXISTS ix_stg_orders_date_nm_id
    ON stg.wb_orders (date, nm_id);

CREATE INDEX IF NOT EXISTS ix_stg_sales_date_nm_id
    ON stg.wb_sales (date, nm_id);

CREATE INDEX IF NOT EXISTS ix_stg_stocks_nm_id_updated_at
    ON stg.wb_stocks (nm_id, updated_at);

-- --- Индексы DICT --- --
CREATE INDEX IF NOT EXISTS ix_dict_cost_reference_nm_id_period
    ON dict.cost_reference (nm_id, valid_from, valid_to);

-- --- Индексы MART --- --
CREATE INDEX IF NOT EXISTS ix_mart_sales_daily_date_nm_id
    ON mart.sales_daily (sales_date, nm_id);

CREATE INDEX IF NOT EXISTS ix_mart_orders_daily_date_nm_id
    ON mart.orders_daily (order_date, nm_id);

CREATE INDEX IF NOT EXISTS ix_mart_stocks_snapshot_date_nm_id
    ON mart.stocks_snapshot (snapshot_date, nm_id);

CREATE INDEX IF NOT EXISTS ix_mart_ads_daily_date_nm_id
    ON mart.ads_daily (ads_date, nm_id);

-- --- Индексы Finance --- --
CREATE INDEX IF NOT EXISTS ix_stg_finance_detail_sale_dt_nm_id
    ON stg.wb_finance_detail (sale_dt, nm_id);

CREATE INDEX IF NOT EXISTS ix_mart_finance_daily_date_nm_id
    ON mart.finance_daily (report_date, nm_id);
