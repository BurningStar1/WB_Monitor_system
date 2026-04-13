-- ============================================================
-- 07_triggers_functions_procedures.sql
-- Триггеры, функции и процедуры БД wb_analytics
-- ============================================================

-- Схема журналирования
CREATE SCHEMA IF NOT EXISTS log;

CREATE TABLE IF NOT EXISTS log.event_log (
    id          bigserial PRIMARY KEY,
    event_type  text NOT NULL,
    table_name  text NOT NULL,
    record_key  text,
    old_value   jsonb,
    new_value   jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_event_log_type_created
    ON log.event_log (event_type, created_at);

-- ============================================================
-- TRIGGER 1 (simple): auto-set updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION fn_trg_set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $do$
DECLARE
    tbl text;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'stg.wb_orders', 'stg.wb_sales', 'stg.wb_stocks',
        'mart.sales_daily', 'mart.orders_daily', 'mart.stocks_snapshot'
    ] LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_set_updated_at_%s ON %s;
             CREATE TRIGGER trg_set_updated_at_%s
                 BEFORE INSERT OR UPDATE ON %s
                 FOR EACH ROW EXECUTE FUNCTION fn_trg_set_updated_at();',
            split_part(tbl, '.', 2), tbl,
            split_part(tbl, '.', 2), tbl
        );
    END LOOP;
END $do$;

-- ============================================================
-- TRIGGER 2 (simple): auto-set loaded_at on raw
-- ============================================================
CREATE OR REPLACE FUNCTION fn_trg_set_loaded_at()
RETURNS trigger AS $$
BEGIN
    IF NEW.loaded_at IS NULL THEN
        NEW.loaded_at := now();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_loaded_at ON raw.wb_api_payloads;
CREATE TRIGGER trg_set_loaded_at
    BEFORE INSERT ON raw.wb_api_payloads
    FOR EACH ROW EXECUTE FUNCTION fn_trg_set_loaded_at();

-- ============================================================
-- TRIGGER 3 (complex): validate sales data
-- ============================================================
CREATE OR REPLACE FUNCTION fn_trg_validate_sales_data()
RETURNS trigger AS $$
DECLARE
    v_is_return boolean;
BEGIN
    IF NEW.sale_id IS NULL OR NEW.sale_id = '' THEN
        RAISE EXCEPTION 'sale_id cannot be null or empty';
    END IF;

    IF NEW.nm_id IS NULL THEN
        RAISE WARNING 'nm_id is NULL for sale_id=%', NEW.sale_id;
    END IF;

    v_is_return := COALESCE(NEW.is_storno, false) OR LEFT(NEW.sale_id, 1) = 'R';

    IF NOT v_is_return THEN
        IF NEW.total_price IS NOT NULL AND NEW.total_price < 0 THEN
            RAISE EXCEPTION 'total_price negative for non-return sale: id=%, val=%',
                NEW.sale_id, NEW.total_price;
        END IF;
        IF NEW.finished_price IS NOT NULL AND NEW.finished_price < 0 THEN
            RAISE EXCEPTION 'finished_price negative for non-return sale: id=%, val=%',
                NEW.sale_id, NEW.finished_price;
        END IF;
        IF NEW.for_pay IS NOT NULL AND NEW.for_pay < 0 THEN
            RAISE EXCEPTION 'for_pay negative for non-return sale: id=%, val=%',
                NEW.sale_id, NEW.for_pay;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_validate_sales_data ON stg.wb_sales;
CREATE TRIGGER trg_validate_sales_data
    BEFORE INSERT OR UPDATE ON stg.wb_sales
    FOR EACH ROW EXECUTE FUNCTION fn_trg_validate_sales_data();

-- ============================================================
-- TRIGGER 4 (complex): log order cancellation
-- ============================================================
CREATE OR REPLACE FUNCTION fn_trg_log_order_cancellation()
RETURNS trigger AS $$
BEGIN
    IF OLD.is_cancel = false AND NEW.is_cancel = true THEN
        INSERT INTO log.event_log (event_type, table_name, record_key, old_value, new_value)
        VALUES (
            'order_cancelled',
            'stg.wb_orders',
            NEW.srid,
            jsonb_build_object(
                'is_cancel', OLD.is_cancel,
                'cancel_dt', OLD.cancel_dt,
                'total_price', OLD.total_price
            ),
            jsonb_build_object(
                'is_cancel', NEW.is_cancel,
                'cancel_dt', NEW.cancel_dt,
                'total_price', NEW.total_price,
                'supplier_article', NEW.supplier_article,
                'nm_id', NEW.nm_id
            )
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_log_order_cancellation ON stg.wb_orders;
CREATE TRIGGER trg_log_order_cancellation
    AFTER UPDATE ON stg.wb_orders
    FOR EACH ROW EXECUTE FUNCTION fn_trg_log_order_cancellation();

-- ============================================================
-- FUNCTION 1 (simple): calculate margin percentage
-- ============================================================
CREATE OR REPLACE FUNCTION fn_calc_margin(
    p_revenue numeric,
    p_cost    numeric
) RETURNS numeric AS $$
BEGIN
    IF p_revenue IS NULL OR p_revenue = 0 THEN
        RETURN 0;
    END IF;
    RETURN ROUND((p_revenue - COALESCE(p_cost, 0)) / p_revenue * 100, 2);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- FUNCTION 2 (simple): format ISO week label
-- ============================================================
CREATE OR REPLACE FUNCTION fn_format_week_label(p_date date)
RETURNS text AS $$
BEGIN
    RETURN TO_CHAR(p_date, 'IYYY') || '-W' || LPAD(TO_CHAR(p_date, 'IW'), 2, '0');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- FUNCTION 3 (complex): ABC classification by revenue
-- ============================================================
CREATE OR REPLACE FUNCTION fn_abc_classify(
    p_date_from date DEFAULT '2000-01-01',
    p_date_to   date DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    nm_id            bigint,
    supplier_article text,
    subject          text,
    brand            text,
    total_revenue    numeric,
    revenue_share    numeric,
    cumulative_share numeric,
    abc_category     char(1)
) AS $$
BEGIN
    RETURN QUERY
    WITH revenue AS (
        SELECT
            s.nm_id,
            MAX(s.supplier_article) AS supplier_article,
            MAX(s.subject) AS subject,
            MAX(s.brand) AS brand,
            COALESCE(SUM(s.net_revenue), 0) AS total_revenue
        FROM mart.sales_daily s
        WHERE s.sales_date BETWEEN p_date_from AND p_date_to
          AND s.nm_id IS NOT NULL
        GROUP BY s.nm_id
    ),
    ranked AS (
        SELECT
            r.*,
            ROUND(r.total_revenue / NULLIF(SUM(r.total_revenue) OVER (), 0) * 100, 2) AS revenue_share,
            ROUND(SUM(r.total_revenue) OVER (ORDER BY r.total_revenue DESC)
                / NULLIF(SUM(r.total_revenue) OVER (), 0) * 100, 2) AS cumulative_share
        FROM revenue r
        WHERE r.total_revenue > 0
    )
    SELECT
        ranked.nm_id, ranked.supplier_article, ranked.subject, ranked.brand,
        ranked.total_revenue, ranked.revenue_share, ranked.cumulative_share,
        CASE
            WHEN ranked.cumulative_share <= 80 THEN 'A'
            WHEN ranked.cumulative_share <= 95 THEN 'B'
            ELSE 'C'
        END::char(1)
    FROM ranked
    ORDER BY ranked.total_revenue DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================
-- FUNCTION 4 (complex): estimate stock days remaining
-- ============================================================
CREATE OR REPLACE FUNCTION fn_calc_stock_days(
    p_nm_id     bigint,
    p_warehouse text DEFAULT NULL
)
RETURNS TABLE (
    nm_id           bigint,
    warehouse_name  text,
    current_stock   int,
    avg_daily_sales numeric,
    estimated_days  numeric
) AS $$
BEGIN
    RETURN QUERY
    WITH latest_stock AS (
        SELECT ss.nm_id, ss.warehouse_name, ss.quantity_full AS current_stock
        FROM mart.stocks_snapshot ss
        WHERE ss.nm_id = p_nm_id
          AND (p_warehouse IS NULL OR ss.warehouse_name = p_warehouse)
          AND ss.snapshot_date = (
              SELECT MAX(ss2.snapshot_date)
              FROM mart.stocks_snapshot ss2
              WHERE ss2.nm_id = ss.nm_id AND ss2.warehouse_name = ss.warehouse_name
          )
    ),
    daily_sales AS (
        SELECT sd.nm_id, COALESCE(AVG(sd.sales_count), 0) AS avg_daily_sales
        FROM mart.sales_daily sd
        WHERE sd.nm_id = p_nm_id
          AND sd.sales_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY sd.nm_id
    )
    SELECT ls.nm_id, ls.warehouse_name, ls.current_stock,
           COALESCE(ds.avg_daily_sales, 0),
           CASE WHEN COALESCE(ds.avg_daily_sales, 0) = 0 THEN 999
                ELSE ROUND(ls.current_stock / ds.avg_daily_sales, 1)
           END
    FROM latest_stock ls
    LEFT JOIN daily_sales ds ON ds.nm_id = ls.nm_id;
END;
$$ LANGUAGE plpgsql STABLE;

-- ============================================================
-- PROCEDURE 1 (simple): truncate STG tables
-- ============================================================
CREATE OR REPLACE PROCEDURE sp_truncate_stg()
LANGUAGE plpgsql AS $$
BEGIN
    TRUNCATE TABLE stg.wb_orders CASCADE;
    TRUNCATE TABLE stg.wb_sales CASCADE;
    TRUNCATE TABLE stg.wb_stocks CASCADE;

    INSERT INTO log.event_log (event_type, table_name, record_key)
    VALUES ('stg_truncated', 'stg.*', 'all tables');

    RAISE NOTICE 'STG tables truncated successfully';
END;
$$;

-- ============================================================
-- PROCEDURE 2 (simple): refresh mart tables
-- ============================================================
CREATE OR REPLACE PROCEDURE sp_refresh_marts()
LANGUAGE plpgsql AS $$
DECLARE
    v_start timestamptz;
    v_count bigint;
BEGIN
    v_start := clock_timestamp();

    INSERT INTO mart.orders_daily (
        order_date, nm_id, supplier_article, subject, brand,
        orders_count, orders_amount, cancelled_count, avg_price
    )
    SELECT
        o.date::date, o.nm_id, o.supplier_article,
        MAX(o.subject), MAX(o.brand),
        COUNT(*), COALESCE(SUM(o.total_price), 0),
        COUNT(*) FILTER (WHERE o.is_cancel),
        COALESCE(AVG(o.total_price), 0)
    FROM stg.wb_orders o
    WHERE o.nm_id IS NOT NULL
    GROUP BY o.date::date, o.nm_id, o.supplier_article
    ON CONFLICT (order_date, nm_id, supplier_article) DO UPDATE SET
        orders_count = EXCLUDED.orders_count,
        orders_amount = EXCLUDED.orders_amount,
        cancelled_count = EXCLUDED.cancelled_count,
        avg_price = EXCLUDED.avg_price,
        subject = EXCLUDED.subject,
        brand = EXCLUDED.brand;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RAISE NOTICE 'orders_daily: % rows', v_count;

    INSERT INTO mart.stocks_snapshot (
        snapshot_date, nm_id, supplier_article, warehouse_name,
        subject, brand, quantity, in_way_to_client, in_way_from_client,
        quantity_full, price, discount
    )
    SELECT
        s.source_loaded_at::date, s.nm_id, s.supplier_article, s.warehouse_name,
        MAX(s.subject), MAX(s.brand),
        SUM(s.quantity), SUM(s.in_way_to_client), SUM(s.in_way_from_client),
        SUM(s.quantity_full), AVG(s.price), AVG(s.discount)
    FROM stg.wb_stocks s
    WHERE s.nm_id IS NOT NULL
    GROUP BY s.source_loaded_at::date, s.nm_id, s.supplier_article, s.warehouse_name
    ON CONFLICT (snapshot_date, nm_id, warehouse_name) DO UPDATE SET
        quantity = EXCLUDED.quantity,
        in_way_to_client = EXCLUDED.in_way_to_client,
        in_way_from_client = EXCLUDED.in_way_from_client,
        quantity_full = EXCLUDED.quantity_full,
        price = EXCLUDED.price,
        discount = EXCLUDED.discount,
        supplier_article = EXCLUDED.supplier_article,
        subject = EXCLUDED.subject,
        brand = EXCLUDED.brand;

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RAISE NOTICE 'stocks_snapshot: % rows', v_count;

    INSERT INTO log.event_log (event_type, table_name, record_key)
    VALUES ('marts_refreshed', 'mart.*',
            'duration: ' || EXTRACT(EPOCH FROM clock_timestamp() - v_start)::text || 's');

    RAISE NOTICE 'All marts refreshed in % seconds',
        ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_start)::numeric, 2);
END;
$$;

-- ============================================================
-- PROCEDURE 3 (complex): full ETL cycle with error handling
-- ============================================================
CREATE OR REPLACE PROCEDURE sp_run_full_etl(p_days_back int DEFAULT 7)
LANGUAGE plpgsql AS $$
DECLARE
    v_start      timestamptz;
    v_step_start timestamptz;
    v_count      bigint;
    v_errors     text[] := '{}';
BEGIN
    v_start := clock_timestamp();

    BEGIN
        v_step_start := clock_timestamp();
        INSERT INTO stg.wb_orders (
            srid, date, last_change_date, supplier_article, tech_size, barcode,
            total_price, discount_percent, warehouse_name, oblast, income_id,
            odid, nm_id, subject, category, brand, is_cancel, cancel_dt, source_loaded_at
        )
        SELECT DISTINCT ON ((r.item ->> 'srid'))
            r.item ->> 'srid',
            NULLIF(r.item ->> 'date', '')::timestamptz,
            NULLIF(r.item ->> 'lastChangeDate', '')::timestamptz,
            r.item ->> 'supplierArticle', r.item ->> 'techSize', r.item ->> 'barcode',
            NULLIF(r.item ->> 'totalPrice', '')::numeric(14,2),
            NULLIF(r.item ->> 'discountPercent', '')::numeric(6,2),
            r.item ->> 'warehouseName', r.item ->> 'oblast',
            NULLIF(r.item ->> 'incomeID', '')::bigint,
            NULLIF(r.item ->> 'odid', '')::bigint,
            NULLIF(r.item ->> 'nmId', '')::bigint,
            r.item ->> 'subject', r.item ->> 'category', r.item ->> 'brand',
            COALESCE((r.item ->> 'isCancel')::boolean, false),
            NULLIF(r.item ->> 'cancel_dt', '')::timestamptz,
            p.loaded_at
        FROM raw.wb_api_payloads p
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload ELSE '[]'::jsonb END
        ) AS r(item)
        WHERE p.endpoint = 'orders'
          AND p.loaded_at >= CURRENT_DATE - (p_days_back || ' days')::interval
        ON CONFLICT (srid) DO UPDATE SET
            last_change_date = EXCLUDED.last_change_date,
            total_price = EXCLUDED.total_price,
            discount_percent = EXCLUDED.discount_percent,
            is_cancel = EXCLUDED.is_cancel,
            cancel_dt = EXCLUDED.cancel_dt,
            source_loaded_at = EXCLUDED.source_loaded_at;
        GET DIAGNOSTICS v_count = ROW_COUNT;
        RAISE NOTICE 'STG orders: % rows (% sec)', v_count,
            ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_step_start)::numeric, 2);
    EXCEPTION WHEN OTHERS THEN
        v_errors := array_append(v_errors, 'orders: ' || SQLERRM);
        RAISE WARNING 'Error loading STG orders: %', SQLERRM;
    END;

    BEGIN
        v_step_start := clock_timestamp();
        INSERT INTO stg.wb_sales (
            sale_id, date, last_change_date, supplier_article, tech_size, barcode,
            total_price, discount_percent, is_supply, is_realization, promo_code_discount,
            warehouse_name, country_name, oblast_okrug_name, region_name,
            income_id, odid, spp, for_pay, finished_price, price_with_disc,
            nm_id, subject, category, brand, is_storno, source_loaded_at
        )
        SELECT DISTINCT ON ((r.item ->> 'saleID'))
            r.item ->> 'saleID',
            NULLIF(r.item ->> 'date', '')::timestamptz,
            NULLIF(r.item ->> 'lastChangeDate', '')::timestamptz,
            r.item ->> 'supplierArticle', r.item ->> 'techSize', r.item ->> 'barcode',
            NULLIF(r.item ->> 'totalPrice', '')::numeric(14,2),
            NULLIF(r.item ->> 'discountPercent', '')::numeric(6,2),
            COALESCE((r.item ->> 'isSupply')::boolean, false),
            COALESCE((r.item ->> 'isRealization')::boolean, false),
            NULLIF(r.item ->> 'promoCodeDiscount', '')::numeric(14,2),
            r.item ->> 'warehouseName', r.item ->> 'countryName',
            r.item ->> 'oblastOkrugName', r.item ->> 'regionName',
            NULLIF(r.item ->> 'incomeID', '')::bigint,
            NULLIF(r.item ->> 'odid', '')::bigint,
            NULLIF(r.item ->> 'spp', '')::numeric(8,2),
            NULLIF(r.item ->> 'forPay', '')::numeric(14,2),
            NULLIF(r.item ->> 'finishedPrice', '')::numeric(14,2),
            NULLIF(r.item ->> 'priceWithDisc', '')::numeric(14,2),
            NULLIF(r.item ->> 'nmId', '')::bigint,
            r.item ->> 'subject', r.item ->> 'category', r.item ->> 'brand',
            COALESCE((r.item ->> 'isStorno')::boolean, false),
            p.loaded_at
        FROM raw.wb_api_payloads p
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload ELSE '[]'::jsonb END
        ) AS r(item)
        WHERE p.endpoint = 'sales'
          AND p.loaded_at >= CURRENT_DATE - (p_days_back || ' days')::interval
        ON CONFLICT (sale_id) DO UPDATE SET
            last_change_date = EXCLUDED.last_change_date,
            for_pay = EXCLUDED.for_pay,
            finished_price = EXCLUDED.finished_price,
            price_with_disc = EXCLUDED.price_with_disc,
            is_storno = EXCLUDED.is_storno,
            source_loaded_at = EXCLUDED.source_loaded_at;
        GET DIAGNOSTICS v_count = ROW_COUNT;
        RAISE NOTICE 'STG sales: % rows (% sec)', v_count,
            ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_step_start)::numeric, 2);
    EXCEPTION WHEN OTHERS THEN
        v_errors := array_append(v_errors, 'sales: ' || SQLERRM);
        RAISE WARNING 'Error loading STG sales: %', SQLERRM;
    END;

    BEGIN
        v_step_start := clock_timestamp();
        INSERT INTO stg.wb_stocks (
            warehouse_name, supplier_article, nm_id, barcode, quantity,
            in_way_to_client, in_way_from_client, quantity_full,
            category, subject, brand, tech_size, price,
            discount, is_supply, is_realization, source_loaded_at
        )
        SELECT
            r.item ->> 'warehouseName', r.item ->> 'supplierArticle',
            NULLIF(r.item ->> 'nmId', '')::bigint, r.item ->> 'barcode',
            NULLIF(r.item ->> 'quantity', '')::int,
            NULLIF(r.item ->> 'inWayToClient', '')::int,
            NULLIF(r.item ->> 'inWayFromClient', '')::int,
            NULLIF(r.item ->> 'quantityFull', '')::int,
            r.item ->> 'category', r.item ->> 'subject', r.item ->> 'brand',
            r.item ->> 'techSize',
            NULLIF(r.item ->> 'Price', '')::numeric(14,2),
            NULLIF(r.item ->> 'Discount', '')::numeric(6,2),
            COALESCE((r.item ->> 'isSupply')::boolean, false),
            COALESCE((r.item ->> 'isRealization')::boolean, false),
            p.loaded_at
        FROM raw.wb_api_payloads p
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload ELSE '[]'::jsonb END
        ) AS r(item)
        WHERE p.endpoint = 'stocks'
          AND p.loaded_at >= CURRENT_DATE - (p_days_back || ' days')::interval;
        GET DIAGNOSTICS v_count = ROW_COUNT;
        RAISE NOTICE 'STG stocks: % rows (% sec)', v_count,
            ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_step_start)::numeric, 2);
    EXCEPTION WHEN OTHERS THEN
        v_errors := array_append(v_errors, 'stocks: ' || SQLERRM);
        RAISE WARNING 'Error loading STG stocks: %', SQLERRM;
    END;

    CALL sp_refresh_marts();

    INSERT INTO log.event_log (event_type, table_name, record_key, new_value)
    VALUES (
        CASE WHEN array_length(v_errors, 1) > 0 THEN 'etl_completed_with_errors'
             ELSE 'etl_completed' END,
        'all',
        'duration: ' || ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_start)::numeric, 2)::text || 's',
        CASE WHEN array_length(v_errors, 1) > 0 THEN to_jsonb(v_errors) ELSE NULL END
    );

    IF array_length(v_errors, 1) > 0 THEN
        RAISE NOTICE 'ETL completed with % error(s) in % seconds',
            array_length(v_errors, 1),
            ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_start)::numeric, 2);
    ELSE
        RAISE NOTICE 'ETL completed successfully in % seconds',
            ROUND(EXTRACT(EPOCH FROM clock_timestamp() - v_start)::numeric, 2);
    END IF;
END;
$$;

-- ============================================================
-- PROCEDURE 4 (complex): generate period report with cursor
-- ============================================================
CREATE OR REPLACE PROCEDURE sp_generate_period_report(
    p_date_from  date,
    p_date_to    date,
    INOUT result refcursor DEFAULT 'period_report'
)
LANGUAGE plpgsql AS $$
DECLARE
    v_total_revenue  numeric;
    v_total_cost     numeric;
    v_total_orders   bigint;
    v_total_sales    bigint;
    v_total_returns  bigint;
BEGIN
    SELECT
        COALESCE(SUM(net_revenue), 0),
        COALESCE(SUM(cost_amount), 0),
        COALESCE(SUM(orders_count), 0),
        COALESCE(SUM(sales_count), 0),
        COALESCE(SUM(returns_count), 0)
    INTO v_total_revenue, v_total_cost, v_total_orders, v_total_sales, v_total_returns
    FROM mart.sales_daily
    WHERE sales_date BETWEEN p_date_from AND p_date_to;

    RAISE NOTICE 'Period: % to %', p_date_from, p_date_to;
    RAISE NOTICE 'Revenue: %, Cost: %, Margin: %',
        v_total_revenue, v_total_cost,
        fn_calc_margin(v_total_revenue, v_total_cost);
    RAISE NOTICE 'Orders: %, Sales: %, Returns: %',
        v_total_orders, v_total_sales, v_total_returns;

    OPEN result FOR
        SELECT
            sd.nm_id,
            sd.supplier_article,
            MAX(sd.subject) AS subject,
            MAX(sd.brand) AS brand,
            SUM(sd.sales_count) AS sales_count,
            SUM(sd.returns_count) AS returns_count,
            SUM(sd.net_revenue) AS net_revenue,
            SUM(sd.cost_amount) AS cost_amount,
            SUM(sd.profit_amount) AS profit_amount,
            fn_calc_margin(SUM(sd.net_revenue), SUM(sd.cost_amount)) AS margin_pct,
            fn_format_week_label(MIN(sd.sales_date)) AS first_week,
            fn_format_week_label(MAX(sd.sales_date)) AS last_week
        FROM mart.sales_daily sd
        WHERE sd.sales_date BETWEEN p_date_from AND p_date_to
          AND sd.nm_id IS NOT NULL
        GROUP BY sd.nm_id, sd.supplier_article
        ORDER BY SUM(sd.net_revenue) DESC;

    INSERT INTO log.event_log (event_type, table_name, record_key, new_value)
    VALUES ('period_report_generated', 'mart.sales_daily',
            p_date_from::text || ' to ' || p_date_to::text,
            jsonb_build_object(
                'revenue', v_total_revenue,
                'cost', v_total_cost,
                'orders', v_total_orders,
                'sales', v_total_sales,
                'returns', v_total_returns
            ));
END;
$$;
