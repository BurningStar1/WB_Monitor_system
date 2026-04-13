from __future__ import annotations

from sqlalchemy import text

from db import get_engine
from utils import get_logger


class MartLoader:
    def __init__(self, log_level: str = "INFO"):
        self.engine = get_engine()
        self.logger = get_logger(self.__class__.__name__, log_level)

    def run(self) -> None:
        self._load_orders_daily()
        self._load_sales_daily()
        self._load_stocks_snapshot()
        self._load_finance_daily()
        self._load_ads_daily()
        self.logger.info("MART loading completed")

    # --- Витрина заказов --- #
    def _load_orders_daily(self) -> None:
        stmt = text(
            """
            INSERT INTO mart.orders_daily (
                order_date, nm_id, supplier_article, subject, brand,
                orders_count, orders_amount, orders_amount_disc,
                cancelled_count, avg_price, updated_at
            )
            SELECT
                o.date::date AS order_date,
                o.nm_id,
                o.supplier_article,
                MAX(o.subject) AS subject,
                MAX(o.brand) AS brand,
                COUNT(*) FILTER (WHERE NOT o.is_cancel) AS orders_count,
                COALESCE(SUM(o.total_price) FILTER (WHERE NOT o.is_cancel), 0) AS orders_amount,
                COALESCE(SUM(o.price_with_disc) FILTER (WHERE NOT o.is_cancel), 0) AS orders_amount_disc,
                COUNT(*) FILTER (WHERE o.is_cancel) AS cancelled_count,
                COALESCE(AVG(o.total_price), 0) AS avg_price,
                now()
            FROM stg.wb_orders o
            WHERE o.nm_id IS NOT NULL
            GROUP BY o.date::date, o.nm_id, o.supplier_article
            ON CONFLICT (order_date, nm_id, supplier_article) DO UPDATE SET
                orders_count = EXCLUDED.orders_count,
                orders_amount = EXCLUDED.orders_amount,
                orders_amount_disc = EXCLUDED.orders_amount_disc,
                cancelled_count = EXCLUDED.cancelled_count,
                avg_price = EXCLUDED.avg_price,
                subject = EXCLUDED.subject,
                brand = EXCLUDED.brand,
                updated_at = now();
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        self.logger.info("orders_daily loaded")

    # --- Витрина продаж и финансовых показателей --- #
    def _load_sales_daily(self) -> None:
        stmt = text(
            """
            INSERT INTO mart.sales_daily (
                sales_date, nm_id, supplier_article, subject, brand,
                orders_count, sales_count, returns_count,
                gross_revenue, net_revenue, commission_amount,
                cost_amount, extra_expenses_amount, tax_amount,
                profit_amount, operating_profit_amount,
                avg_spp, avg_price_before_spp, avg_price_after_spp,
                updated_at
            )
            SELECT
                base.sales_date,
                base.nm_id,
                base.supplier_article,
                base.subject,
                base.brand,
                COALESCE(ord.orders_count, 0) AS orders_count,
                base.sales_count,
                base.returns_count,
                base.gross_revenue,
                base.net_revenue,
                base.commission_amount,
                COALESCE(cr.unit_cost, 0) * base.sales_count AS cost_amount,
                COALESCE(ex.amount, 0) AS extra_expenses_amount,
                base.net_revenue * COALESCE(tx.tax_rate_percent, 0) / 100.0 AS tax_amount,
                base.net_revenue
                    - COALESCE(cr.unit_cost, 0) * base.sales_count
                    - COALESCE(ex.amount, 0) AS profit_amount,
                base.net_revenue
                    - COALESCE(cr.unit_cost, 0) * base.sales_count
                    - COALESCE(ex.amount, 0)
                    - base.net_revenue * COALESCE(tx.tax_rate_percent, 0) / 100.0
                    AS operating_profit_amount,
                base.avg_spp,
                base.avg_price_before_spp,
                base.avg_price_after_spp,
                now()
            FROM (
                SELECT
                    s.date::date AS sales_date,
                    s.nm_id,
                    s.supplier_article,
                    MAX(s.subject) AS subject,
                    MAX(s.brand) AS brand,
                    COUNT(*) FILTER (WHERE s.sale_id NOT LIKE 'R%') AS sales_count,
                    COUNT(*) FILTER (WHERE s.sale_id LIKE 'R%') AS returns_count,
                    COALESCE(SUM(s.finished_price) FILTER (WHERE s.sale_id NOT LIKE 'R%'), 0) AS gross_revenue,
                    COALESCE(SUM(s.for_pay), 0) AS net_revenue,
                    COALESCE(SUM(s.finished_price - s.for_pay) FILTER (WHERE s.sale_id NOT LIKE 'R%'), 0) AS commission_amount,
                    COALESCE(AVG(s.spp) FILTER (WHERE s.sale_id NOT LIKE 'R%'), 0) AS avg_spp,
                    COALESCE(AVG(s.price_with_disc) FILTER (WHERE s.sale_id NOT LIKE 'R%'), 0) AS avg_price_before_spp,
                    COALESCE(AVG(s.finished_price) FILTER (WHERE s.sale_id NOT LIKE 'R%'), 0) AS avg_price_after_spp
                FROM stg.wb_sales s
                WHERE s.nm_id IS NOT NULL
                GROUP BY s.date::date, s.nm_id, s.supplier_article
            ) base
            LEFT JOIN LATERAL (
                SELECT COUNT(*) AS orders_count
                FROM stg.wb_orders o
                WHERE o.nm_id = base.nm_id
                  AND o.date::date = base.sales_date
            ) ord ON true
            LEFT JOIN LATERAL (
                SELECT c.unit_cost
                FROM dict.cost_reference c
                WHERE c.nm_id = base.nm_id
                  AND base.sales_date BETWEEN c.valid_from AND c.valid_to
                ORDER BY c.valid_from DESC
                LIMIT 1
            ) cr ON true
            LEFT JOIN LATERAL (
                SELECT SUM(e.amount) AS amount
                FROM dict.extra_expenses e
                WHERE e.expense_date = base.sales_date
            ) ex ON true
            LEFT JOIN LATERAL (
                SELECT t.tax_rate_percent
                FROM dict.tax_reference t
                WHERE base.sales_date BETWEEN t.valid_from AND t.valid_to
                ORDER BY t.valid_from DESC
                LIMIT 1
            ) tx ON true
            ON CONFLICT (sales_date, nm_id, supplier_article) DO UPDATE SET
                orders_count = EXCLUDED.orders_count,
                sales_count = EXCLUDED.sales_count,
                returns_count = EXCLUDED.returns_count,
                gross_revenue = EXCLUDED.gross_revenue,
                net_revenue = EXCLUDED.net_revenue,
                commission_amount = EXCLUDED.commission_amount,
                cost_amount = EXCLUDED.cost_amount,
                extra_expenses_amount = EXCLUDED.extra_expenses_amount,
                tax_amount = EXCLUDED.tax_amount,
                profit_amount = EXCLUDED.profit_amount,
                operating_profit_amount = EXCLUDED.operating_profit_amount,
                avg_spp = EXCLUDED.avg_spp,
                avg_price_before_spp = EXCLUDED.avg_price_before_spp,
                avg_price_after_spp = EXCLUDED.avg_price_after_spp,
                subject = EXCLUDED.subject,
                brand = EXCLUDED.brand,
                updated_at = now();
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        self.logger.info("sales_daily loaded")

    # --- Витрина остатков --- #
    def _load_stocks_snapshot(self) -> None:
        stmt = text(
            """
            INSERT INTO mart.stocks_snapshot (
                snapshot_date, nm_id, supplier_article, warehouse_name,
                subject, brand,
                quantity, in_way_to_client, in_way_from_client, quantity_full,
                price, discount, updated_at
            )
            SELECT
                s.source_loaded_at::date AS snapshot_date,
                s.nm_id,
                s.supplier_article,
                s.warehouse_name,
                MAX(s.subject) AS subject,
                MAX(s.brand) AS brand,
                SUM(s.quantity) AS quantity,
                SUM(s.in_way_to_client) AS in_way_to_client,
                SUM(s.in_way_from_client) AS in_way_from_client,
                SUM(s.quantity_full) AS quantity_full,
                AVG(s.price) AS price,
                AVG(s.discount) AS discount,
                now()
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
                brand = EXCLUDED.brand,
                updated_at = now();
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        self.logger.info("stocks_snapshot loaded")

    # --- Витрина финансовых отчётов --- #
    def _load_finance_daily(self) -> None:
        stmt = text(
            """
            INSERT INTO mart.finance_daily (
                report_date, nm_id, supplier_article, subject, brand,
                sales_count, returns_count,
                sales_amount, returns_amount,
                retail_amount, ppvz_for_pay,
                commission_amount, logistics_amount, storage_amount,
                penalty_amount, acceptance_amount, acquiring_amount,
                deduction_amount, additional_payment_amount,
                updated_at
            )
            SELECT
                agg.report_date, agg.nm_id, agg.supplier_article,
                agg.subject, agg.brand,
                agg.sales_count, agg.returns_count,
                agg.sales_amount, agg.returns_amount,
                agg.retail_net,
                agg.ppvz_for_pay_net,
                (agg.sales_amount - agg.returns_amount) - agg.ppvz_for_pay_net
                    AS commission_amount,
                agg.logistics_amount, agg.storage_amount,
                agg.penalty_amount, agg.acceptance_amount,
                agg.acquiring_amount, agg.deduction_amount,
                agg.additional_payment_amount,
                now()
            FROM (
                SELECT
                    f.rr_dt::date AS report_date,
                    f.nm_id,
                    f.sa_name AS supplier_article,
                    MAX(f.subject_name) AS subject,
                    MAX(f.brand_name) AS brand,
                    -- Counts: only trade rows (with retail_amount > 0)
                    COALESCE(COUNT(*) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%продажа%'
                          AND f.retail_amount > 0
                    ), 0) AS sales_count,
                    COALESCE(COUNT(*) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%возврат%'
                          AND f.retail_amount > 0
                    ), 0) AS returns_count,
                    -- Revenue: retail_price_withdisc_rub * quantity (= Реализация до СПП)
                    COALESCE(SUM(f.retail_price_withdisc_rub * f.quantity) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%продажа%'
                          AND f.retail_amount > 0
                    ), 0) AS sales_amount,
                    COALESCE(SUM(f.retail_price_withdisc_rub * f.quantity) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%возврат%'
                          AND f.retail_amount > 0
                    ), 0) AS returns_amount,
                    -- Net retail_amount (= Реализация после СПП)
                    COALESCE(SUM(f.retail_amount) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%продажа%'
                    ), 0)
                    - COALESCE(SUM(f.retail_amount) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%возврат%'
                          AND f.retail_amount > 0
                    ), 0) AS retail_net,
                    -- Net ppvz_for_pay (sales - returns + compensation)
                    COALESCE(SUM(f.ppvz_for_pay) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%продажа%'
                    ), 0)
                    - COALESCE(SUM(f.ppvz_for_pay) FILTER (
                        WHERE f.supplier_oper_name ILIKE '%возврат%'
                          AND f.retail_amount > 0
                    ), 0)
                    + COALESCE(SUM(f.ppvz_for_pay) FILTER (
                        WHERE f.retail_amount = 0
                          AND f.delivery_rub = 0
                          AND f.storage_fee = 0
                          AND f.penalty = 0
                          AND f.acceptance = 0
                          AND f.ppvz_for_pay <> 0
                    ), 0) AS ppvz_for_pay_net,
                    -- Services (always positive costs)
                    COALESCE(SUM(f.delivery_rub), 0) AS logistics_amount,
                    COALESCE(SUM(f.storage_fee), 0) AS storage_amount,
                    COALESCE(SUM(f.penalty), 0) AS penalty_amount,
                    COALESCE(SUM(f.acceptance), 0) AS acceptance_amount,
                    COALESCE(SUM(f.acquiring_fee), 0) AS acquiring_amount,
                    COALESCE(SUM(f.deduction), 0) AS deduction_amount,
                    COALESCE(SUM(f.additional_payment), 0)
                        AS additional_payment_amount
                FROM stg.wb_finance_detail f
                WHERE f.nm_id IS NOT NULL
                  AND f.rr_dt IS NOT NULL
                GROUP BY f.rr_dt::date, f.nm_id, f.sa_name
            ) agg
            ON CONFLICT (report_date, nm_id, supplier_article) DO UPDATE SET
                sales_count = EXCLUDED.sales_count,
                returns_count = EXCLUDED.returns_count,
                sales_amount = EXCLUDED.sales_amount,
                returns_amount = EXCLUDED.returns_amount,
                retail_amount = EXCLUDED.retail_amount,
                ppvz_for_pay = EXCLUDED.ppvz_for_pay,
                commission_amount = EXCLUDED.commission_amount,
                logistics_amount = EXCLUDED.logistics_amount,
                storage_amount = EXCLUDED.storage_amount,
                penalty_amount = EXCLUDED.penalty_amount,
                acceptance_amount = EXCLUDED.acceptance_amount,
                acquiring_amount = EXCLUDED.acquiring_amount,
                deduction_amount = EXCLUDED.deduction_amount,
                additional_payment_amount = EXCLUDED.additional_payment_amount,
                subject = EXCLUDED.subject,
                brand = EXCLUDED.brand,
                updated_at = now();
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        self.logger.info("finance_daily loaded")

    # --- Витрина рекламы --- #
    def _load_ads_daily(self) -> None:
        stmt = text(
            """
            INSERT INTO mart.ads_daily (
                ads_date, nm_id, campaign_id, supplier_article,
                views_count, clicks_count, ctr, cpc,
                orders_from_ads, spend_amount, updated_at
            )
            SELECT
                a.ads_date,
                a.nm_id,
                a.campaign_id,
                MAX(s.supplier_article) AS supplier_article,
                SUM(a.views) AS views_count,
                SUM(a.clicks) AS clicks_count,
                CASE WHEN SUM(a.views) > 0
                     THEN SUM(a.clicks)::numeric / SUM(a.views) * 100
                     ELSE 0 END AS ctr,
                CASE WHEN SUM(a.clicks) > 0
                     THEN SUM(a.spend) / SUM(a.clicks)
                     ELSE 0 END AS cpc,
                SUM(a.orders_count) AS orders_from_ads,
                SUM(a.spend) AS spend_amount,
                now()
            FROM stg.wb_ads_daily a
            LEFT JOIN (
                SELECT DISTINCT nm_id, supplier_article
                FROM stg.wb_sales
                WHERE nm_id IS NOT NULL
            ) s ON s.nm_id = a.nm_id
            WHERE a.nm_id IS NOT NULL AND a.nm_id > 0
            GROUP BY a.ads_date, a.nm_id, a.campaign_id
            ON CONFLICT (ads_date, nm_id, campaign_id) DO UPDATE SET
                supplier_article = EXCLUDED.supplier_article,
                views_count = EXCLUDED.views_count,
                clicks_count = EXCLUDED.clicks_count,
                ctr = EXCLUDED.ctr,
                cpc = EXCLUDED.cpc,
                orders_from_ads = EXCLUDED.orders_from_ads,
                spend_amount = EXCLUDED.spend_amount,
                updated_at = now();
            """
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)
        self.logger.info("ads_daily loaded")
