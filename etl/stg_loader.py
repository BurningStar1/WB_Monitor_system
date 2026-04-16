from __future__ import annotations

from sqlalchemy import text

from db import get_engine
from utils import get_logger


class StgLoader:
    def __init__(self, log_level: str = "INFO"):
        self.engine = get_engine()
        self.logger = get_logger(self.__class__.__name__, log_level)

    def run(self) -> None:
        """Loads normalized records from raw JSON into STG tables."""
        stmt_orders = text(
            """
            INSERT INTO stg.wb_orders (
                srid, date, last_change_date, supplier_article, tech_size, barcode,
                total_price, discount_percent, price_with_disc, finished_price,
                warehouse_name, oblast, income_id,
                odid, nm_id, subject, category, brand, is_cancel, cancel_dt,
                source_loaded_at, updated_at
            )
            SELECT DISTINCT ON ((r.item ->> 'srid'))
                r.item ->> 'srid' AS srid,
                NULLIF(r.item ->> 'date', '')::timestamptz,
                NULLIF(r.item ->> 'lastChangeDate', '')::timestamptz,
                r.item ->> 'supplierArticle',
                r.item ->> 'techSize',
                r.item ->> 'barcode',
                NULLIF(r.item ->> 'totalPrice', '')::numeric(14,2),
                NULLIF(r.item ->> 'discountPercent', '')::numeric(6,2),
                NULLIF(r.item ->> 'priceWithDisc', '')::numeric(14,2),
                NULLIF(r.item ->> 'finishedPrice', '')::numeric(14,2),
                r.item ->> 'warehouseName',
                r.item ->> 'oblast',
                NULLIF(r.item ->> 'incomeID', '')::bigint,
                NULLIF(r.item ->> 'odid', '')::bigint,
                NULLIF(r.item ->> 'nmId', '')::bigint,
                r.item ->> 'subject',
                r.item ->> 'category',
                r.item ->> 'brand',
                COALESCE((r.item ->> 'isCancel')::boolean, false),
                NULLIF(r.item ->> 'cancel_dt', '')::timestamptz,
                p.loaded_at,
                now()
            FROM raw.wb_api_payloads p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload
                    ELSE '[]'::jsonb
                END
            ) AS r(item)
            WHERE p.endpoint = 'orders'
            ORDER BY (r.item ->> 'srid'), p.loaded_at DESC
            ON CONFLICT (srid) DO UPDATE SET
                last_change_date = EXCLUDED.last_change_date,
                total_price = EXCLUDED.total_price,
                discount_percent = EXCLUDED.discount_percent,
                price_with_disc = EXCLUDED.price_with_disc,
                finished_price = EXCLUDED.finished_price,
                is_cancel = EXCLUDED.is_cancel,
                cancel_dt = EXCLUDED.cancel_dt,
                source_loaded_at = EXCLUDED.source_loaded_at,
                updated_at = now();
            """
        )

        stmt_sales = text(
            """
            INSERT INTO stg.wb_sales (
                sale_id, date, last_change_date, supplier_article, tech_size, barcode,
                total_price, discount_percent, is_supply, is_realization, promo_code_discount,
                warehouse_name, country_name, oblast_okrug_name, region_name,
                income_id, odid, spp, for_pay, finished_price, price_with_disc,
                nm_id, subject, category, brand, is_storno,
                source_loaded_at, updated_at
            )
            SELECT DISTINCT ON ((r.item ->> 'saleID'))
                r.item ->> 'saleID' AS sale_id,
                NULLIF(r.item ->> 'date', '')::timestamptz,
                NULLIF(r.item ->> 'lastChangeDate', '')::timestamptz,
                r.item ->> 'supplierArticle',
                r.item ->> 'techSize',
                r.item ->> 'barcode',
                NULLIF(r.item ->> 'totalPrice', '')::numeric(14,2),
                NULLIF(r.item ->> 'discountPercent', '')::numeric(6,2),
                COALESCE((r.item ->> 'isSupply')::boolean, false),
                COALESCE((r.item ->> 'isRealization')::boolean, false),
                NULLIF(r.item ->> 'promoCodeDiscount', '')::numeric(14,2),
                r.item ->> 'warehouseName',
                r.item ->> 'countryName',
                r.item ->> 'oblastOkrugName',
                r.item ->> 'regionName',
                NULLIF(r.item ->> 'incomeID', '')::bigint,
                NULLIF(r.item ->> 'odid', '')::bigint,
                NULLIF(r.item ->> 'spp', '')::numeric(8,2),
                NULLIF(r.item ->> 'forPay', '')::numeric(14,2),
                NULLIF(r.item ->> 'finishedPrice', '')::numeric(14,2),
                NULLIF(r.item ->> 'priceWithDisc', '')::numeric(14,2),
                NULLIF(r.item ->> 'nmId', '')::bigint,
                r.item ->> 'subject',
                r.item ->> 'category',
                r.item ->> 'brand',
                COALESCE((r.item ->> 'isStorno')::boolean, false),
                p.loaded_at,
                now()
            FROM raw.wb_api_payloads p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload
                    ELSE '[]'::jsonb
                END
            ) AS r(item)
            WHERE p.endpoint = 'sales'
            ON CONFLICT (sale_id) DO UPDATE SET
                last_change_date = EXCLUDED.last_change_date,
                for_pay = EXCLUDED.for_pay,
                finished_price = EXCLUDED.finished_price,
                price_with_disc = EXCLUDED.price_with_disc,
                is_storno = EXCLUDED.is_storno,
                source_loaded_at = EXCLUDED.source_loaded_at,
                updated_at = now();
            """
        )

        # Дедуплицируем в рамках одного снимка (одна дата × склад × артикул ×
        # размер × баркод): выбираем самый свежий payload для данной комбинации.
        stmt_stocks = text(
            """
            INSERT INTO stg.wb_stocks (
                warehouse_name, supplier_article, nm_id, barcode, quantity,
                in_way_to_client, in_way_from_client, quantity_full,
                category, subject, brand, tech_size, price,
                discount, is_supply, is_realization,
                source_loaded_at, updated_at
            )
            SELECT DISTINCT ON (
                p.loaded_at::date,
                r.item ->> 'warehouseName',
                NULLIF(r.item ->> 'nmId', '')::bigint,
                r.item ->> 'techSize',
                r.item ->> 'barcode'
            )
                r.item ->> 'warehouseName',
                r.item ->> 'supplierArticle',
                NULLIF(r.item ->> 'nmId', '')::bigint,
                r.item ->> 'barcode',
                NULLIF(r.item ->> 'quantity', '')::int,
                NULLIF(r.item ->> 'inWayToClient', '')::int,
                NULLIF(r.item ->> 'inWayFromClient', '')::int,
                NULLIF(r.item ->> 'quantityFull', '')::int,
                r.item ->> 'category',
                r.item ->> 'subject',
                r.item ->> 'brand',
                r.item ->> 'techSize',
                NULLIF(r.item ->> 'Price', '')::numeric(14,2),
                NULLIF(r.item ->> 'Discount', '')::numeric(6,2),
                COALESCE((r.item ->> 'isSupply')::boolean, false),
                COALESCE((r.item ->> 'isRealization')::boolean, false),
                p.loaded_at,
                now()
            FROM raw.wb_api_payloads p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload
                    ELSE '[]'::jsonb
                END
            ) AS r(item)
            WHERE p.endpoint = 'stocks'
            ORDER BY
                p.loaded_at::date,
                r.item ->> 'warehouseName',
                NULLIF(r.item ->> 'nmId', '')::bigint,
                r.item ->> 'techSize',
                r.item ->> 'barcode',
                p.loaded_at DESC;
            """
        )

        stmt_finance = text(
            """
            INSERT INTO stg.wb_finance_detail (
                rrd_id, realizationreport_id, nm_id, sa_name, barcode,
                subject_name, brand_name, doc_type_name, supplier_oper_name,
                order_dt, sale_dt, rr_dt, quantity,
                retail_amount, retail_price_withdisc_rub,
                ppvz_sales_commission, ppvz_for_pay,
                delivery_rub, storage_fee, penalty, additional_payment,
                acceptance, acquiring_fee, ppvz_reward, ppvz_vw, ppvz_vw_nds,
                deduction, rebill_logistic_cost,
                office_name, srid, gi_id,
                source_loaded_at, updated_at
            )
            SELECT DISTINCT ON ((r.item ->> 'rrd_id')::bigint)
                (r.item ->> 'rrd_id')::bigint,
                NULLIF(r.item ->> 'realizationreport_id', '')::bigint,
                NULLIF(r.item ->> 'nm_id', '')::bigint,
                r.item ->> 'sa_name',
                r.item ->> 'barcode',
                r.item ->> 'subject_name',
                r.item ->> 'brand_name',
                r.item ->> 'doc_type_name',
                r.item ->> 'supplier_oper_name',
                NULLIF(r.item ->> 'order_dt', '')::timestamptz,
                NULLIF(r.item ->> 'sale_dt', '')::timestamptz,
                NULLIF(r.item ->> 'rr_dt', '')::timestamptz,
                COALESCE(NULLIF(r.item ->> 'quantity', '')::int, 0),
                COALESCE(NULLIF(r.item ->> 'retail_amount', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'retail_price_withdisc_rub', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'ppvz_sales_commission', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'ppvz_for_pay', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'delivery_rub', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'storage_fee', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'penalty', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'additional_payment', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'acceptance', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'acquiring_fee', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'ppvz_reward', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'ppvz_vw', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'ppvz_vw_nds', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'deduction', '')::numeric, 0),
                COALESCE(NULLIF(r.item ->> 'rebill_logistic_cost', '')::numeric, 0),
                r.item ->> 'office_name',
                r.item ->> 'srid',
                NULLIF(r.item ->> 'gi_id', '')::bigint,
                p.loaded_at,
                now()
            FROM raw.wb_api_payloads p
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE
                    WHEN jsonb_typeof(p.raw_payload) = 'array' THEN p.raw_payload
                    ELSE '[]'::jsonb
                END
            ) AS r(item)
            WHERE p.endpoint = 'finance'
              AND (r.item ->> 'rrd_id') IS NOT NULL
            ON CONFLICT (rrd_id) DO UPDATE SET
                ppvz_for_pay = EXCLUDED.ppvz_for_pay,
                delivery_rub = EXCLUDED.delivery_rub,
                storage_fee = EXCLUDED.storage_fee,
                penalty = EXCLUDED.penalty,
                additional_payment = EXCLUDED.additional_payment,
                acceptance = EXCLUDED.acceptance,
                deduction = EXCLUDED.deduction,
                rebill_logistic_cost = EXCLUDED.rebill_logistic_cost,
                source_loaded_at = EXCLUDED.source_loaded_at,
                updated_at = now();
            """
        )

        stmt_ads = text(
            """
            INSERT INTO stg.wb_ads_daily (
                ads_date, campaign_id, nm_id,
                views, clicks, ctr, cpc, spend,
                atbs, orders_count, shks, sum_price,
                source_loaded_at, updated_at
            )
            SELECT
                ads_date, campaign_id, nm_id,
                SUM(views)::int,
                SUM(clicks)::int,
                CASE WHEN SUM(views) > 0
                     THEN SUM(clicks)::numeric / SUM(views) * 100
                     ELSE 0 END,
                CASE WHEN SUM(clicks) > 0
                     THEN SUM(spend) / SUM(clicks)
                     ELSE 0 END,
                SUM(spend),
                SUM(atbs)::int,
                SUM(orders_count)::int,
                SUM(shks)::int,
                SUM(sum_price),
                MAX(source_loaded_at),
                now()
            FROM (
                SELECT DISTINCT ON (
                    (d.item ->> 'date')::date,
                    (c.item ->> 'advertId')::bigint,
                    COALESCE((n.item ->> 'nmId')::bigint, 0),
                    (a.item ->> 'appType')::int
                )
                    (d.item ->> 'date')::date AS ads_date,
                    (c.item ->> 'advertId')::bigint AS campaign_id,
                    COALESCE((n.item ->> 'nmId')::bigint, 0) AS nm_id,
                    COALESCE((n.item ->> 'views')::int, 0) AS views,
                    COALESCE((n.item ->> 'clicks')::int, 0) AS clicks,
                    COALESCE((n.item ->> 'sum')::numeric, 0) AS spend,
                    COALESCE((n.item ->> 'atbs')::int, 0) AS atbs,
                    COALESCE((n.item ->> 'orders')::int, 0) AS orders_count,
                    COALESCE((n.item ->> 'shks')::int, 0) AS shks,
                    COALESCE((n.item ->> 'sum_price')::numeric, 0) AS sum_price,
                    p.loaded_at AS source_loaded_at
                FROM raw.wb_api_payloads p
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE WHEN jsonb_typeof(p.raw_payload) = 'array'
                         THEN p.raw_payload ELSE '[]'::jsonb END
                ) AS c(item)
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE WHEN c.item -> 'days' IS NOT NULL
                              AND jsonb_typeof(c.item -> 'days') = 'array'
                         THEN c.item -> 'days' ELSE '[]'::jsonb END
                ) AS d(item)
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE WHEN d.item -> 'apps' IS NOT NULL
                              AND jsonb_typeof(d.item -> 'apps') = 'array'
                         THEN d.item -> 'apps' ELSE '[]'::jsonb END
                ) AS a(item)
                CROSS JOIN LATERAL jsonb_array_elements(
                    CASE WHEN a.item -> 'nms' IS NOT NULL
                              AND jsonb_typeof(a.item -> 'nms') = 'array'
                         THEN a.item -> 'nms' ELSE '[]'::jsonb END
                ) AS n(item)
                WHERE p.endpoint = 'ads'
                  AND (c.item ->> 'advertId') IS NOT NULL
                ORDER BY (d.item ->> 'date')::date,
                         (c.item ->> 'advertId')::bigint,
                         COALESCE((n.item ->> 'nmId')::bigint, 0),
                         (a.item ->> 'appType')::int,
                         p.loaded_at DESC
            ) sub
            GROUP BY ads_date, campaign_id, nm_id
            ON CONFLICT (ads_date, campaign_id, nm_id) DO UPDATE SET
                views = EXCLUDED.views,
                clicks = EXCLUDED.clicks,
                ctr = EXCLUDED.ctr,
                cpc = EXCLUDED.cpc,
                spend = EXCLUDED.spend,
                atbs = EXCLUDED.atbs,
                orders_count = EXCLUDED.orders_count,
                shks = EXCLUDED.shks,
                sum_price = EXCLUDED.sum_price,
                source_loaded_at = EXCLUDED.source_loaded_at,
                updated_at = now();
            """
        )

        # stg.wb_stocks не имеет естественного ключа дедупликации
        # (снимок остатков без уникального идентификатора), поэтому
        # перед INSERT чистим таблицу. История остатков сохраняется
        # в mart.stocks_snapshot (там есть ON CONFLICT по дате × nm_id × склад).
        stmt_stocks_truncate = text("TRUNCATE TABLE stg.wb_stocks RESTART IDENTITY;")

        with self.engine.begin() as conn:
            conn.execute(stmt_orders)
            conn.execute(stmt_sales)
            conn.execute(stmt_stocks_truncate)
            conn.execute(stmt_stocks)
            conn.execute(stmt_finance)
            try:
                conn.execute(stmt_ads)
            except Exception as e:
                self.logger.warning("Ads STG loading skipped: %s", e)

        self.logger.info("STG loading completed")
