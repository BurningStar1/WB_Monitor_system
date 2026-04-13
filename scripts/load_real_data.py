"""
Загрузка реальных данных WB API → PostgreSQL (все слои).
Запуск: python scripts/load_real_data.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import openpyxl

# --- Config --- #
API_TOKEN = open(r'C:\Users\petru\OneDrive\Desktop\Текстовый документ.txt').read().strip()
BASE_URL = 'https://statistics-api.wildberries.ru/api/v1/supplier'
HEADERS = {'Authorization': API_TOKEN}
DB_DSN = 'host=localhost port=5432 dbname=wb_analytics user=postgres password=12345678'
XLSX_PATH = r'C:\Users\petru\Downloads\Шаблон_Себестоимость_wb.xlsx'
DATE_FROM = '2026-01-01'  # 3+ месяца данных


def get_conn():
    return psycopg2.connect(DB_DSN)


# ================================================================
# 1. RAW: Загрузка сырых JSON из API
# ================================================================
def fetch_api(endpoint, params):
    """Запрос к WB Statistics API с ретраями."""
    for attempt in range(3):
        try:
            r = requests.get(
                f'{BASE_URL}/{endpoint}',
                headers=HEADERS,
                params=params,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f'  Rate limit, жду 60с... (попытка {attempt+1})')
                time.sleep(60)
            else:
                print(f'  Ошибка {r.status_code}: {r.text[:200]}')
                return []
        except Exception as e:
            print(f'  Exception: {e}, попытка {attempt+1}')
            time.sleep(10)
    return []


def load_raw():
    """Загружает 3 эндпоинта в raw.wb_api_payloads."""
    conn = get_conn()
    cur = conn.cursor()

    endpoints = [
        ('orders', {'dateFrom': DATE_FROM, 'flag': 1}),
        ('sales',  {'dateFrom': DATE_FROM, 'flag': 1}),
        ('stocks', {'dateFrom': DATE_FROM}),
    ]

    results = {}
    for ep, params in endpoints:
        print(f'[RAW] Загружаю {ep}...')
        data = fetch_api(ep, params)
        print(f'  Получено: {len(data)} записей')
        results[ep] = data

        cur.execute("""
            INSERT INTO raw.wb_api_payloads (source, endpoint, request_params, raw_payload)
            VALUES (%s, %s, %s, %s)
        """, ('wb_statistics', ep, json.dumps(params), json.dumps(data)))

    conn.commit()
    cur.close()
    conn.close()
    return results


# ================================================================
# 2. STG: Нормализация JSON → реляционные таблицы
# ================================================================
def load_stg_orders(orders):
    """Загружает заказы в stg.wb_orders по шаблону UPSERT."""
    if not orders:
        print('[STG] Нет заказов для загрузки')
        return
    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for o in orders:
        rows.append((
            o.get('srid', ''),
            o.get('date'),
            o.get('lastChangeDate'),
            o.get('supplierArticle', ''),
            o.get('techSize', ''),
            o.get('barcode', ''),
            o.get('totalPrice', 0),
            o.get('discountPercent', 0),
            o.get('warehouseName', ''),
            o.get('oblastOkrugName', ''),
            o.get('incomeID', 0),
            0,  # odid — нет в новом API
            o.get('nmId', 0),
            o.get('subject', ''),
            o.get('category', ''),
            o.get('brand', ''),
            o.get('isCancel', False),
            o.get('cancelDate'),
            datetime.now(),
        ))

    sql = """
        INSERT INTO stg.wb_orders (
            srid, date, last_change_date, supplier_article, tech_size, barcode,
            total_price, discount_percent, warehouse_name, oblast,
            income_id, odid, nm_id, subject, category, brand,
            is_cancel, cancel_dt, source_loaded_at
        ) VALUES %s
        ON CONFLICT (srid) DO UPDATE SET
            last_change_date = EXCLUDED.last_change_date,
            is_cancel = EXCLUDED.is_cancel,
            cancel_dt = EXCLUDED.cancel_dt,
            updated_at = now()
    """
    execute_values(cur, sql, rows)
    conn.commit()
    print(f'[STG] wb_orders: загружено {len(rows)} записей')
    cur.close()
    conn.close()


def load_stg_sales(sales):
    """Загружает продажи/возвраты в stg.wb_sales по шаблону UPSERT."""
    if not sales:
        print('[STG] Нет продаж для загрузки')
        return
    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for s in sales:
        rows.append((
            s.get('saleID', ''),
            s.get('date'),
            s.get('lastChangeDate'),
            s.get('supplierArticle', ''),
            s.get('techSize', ''),
            s.get('barcode', ''),
            s.get('totalPrice', 0),
            s.get('discountPercent', 0),
            s.get('isSupply', False),
            s.get('isRealization', False),
            0,  # promoCodeDiscount — нет в ответе
            s.get('warehouseName', ''),
            s.get('countryName', ''),
            s.get('oblastOkrugName', ''),
            s.get('regionName', ''),
            s.get('incomeID', 0),
            0,  # odid
            s.get('spp', 0),
            s.get('forPay', 0),
            s.get('finishedPrice', 0),
            s.get('priceWithDisc', 0),
            s.get('nmId', 0),
            s.get('subject', ''),
            s.get('category', ''),
            s.get('brand', ''),
            False,  # is_storno
            datetime.now(),
        ))

    sql = """
        INSERT INTO stg.wb_sales (
            sale_id, date, last_change_date, supplier_article, tech_size, barcode,
            total_price, discount_percent, is_supply, is_realization, promo_code_discount,
            warehouse_name, country_name, oblast_okrug_name, region_name,
            income_id, odid, spp, for_pay, finished_price, price_with_disc,
            nm_id, subject, category, brand, is_storno, source_loaded_at
        ) VALUES %s
        ON CONFLICT (sale_id) DO UPDATE SET
            last_change_date = EXCLUDED.last_change_date,
            for_pay = EXCLUDED.for_pay,
            updated_at = now()
    """
    execute_values(cur, sql, rows)
    conn.commit()
    print(f'[STG] wb_sales: загружено {len(rows)} записей')
    cur.close()
    conn.close()


def load_stg_stocks(stocks):
    """Загружает остатки в stg.wb_stocks."""
    if not stocks:
        print('[STG] Нет остатков для загрузки')
        return
    conn = get_conn()
    cur = conn.cursor()

    # Очищаем старые остатки — это снимок на текущий момент
    cur.execute("TRUNCATE stg.wb_stocks RESTART IDENTITY")

    rows = []
    for s in stocks:
        rows.append((
            s.get('warehouseName', ''),
            s.get('supplierArticle', ''),
            s.get('nmId', 0),
            s.get('barcode', ''),
            s.get('quantity', 0),
            s.get('inWayToClient', 0),
            s.get('inWayFromClient', 0),
            s.get('quantityFull', 0),
            s.get('category', ''),
            s.get('subject', ''),
            s.get('brand', ''),
            s.get('techSize', ''),
            s.get('Price', 0),
            s.get('Discount', 0),
            s.get('isSupply', False),
            s.get('isRealization', False),
            datetime.now(),
        ))

    sql = """
        INSERT INTO stg.wb_stocks (
            warehouse_name, supplier_article, nm_id, barcode,
            quantity, in_way_to_client, in_way_from_client, quantity_full,
            category, subject, brand, tech_size, price, discount,
            is_supply, is_realization, source_loaded_at
        ) VALUES %s
    """
    execute_values(cur, sql, rows)
    conn.commit()
    print(f'[STG] wb_stocks: загружено {len(rows)} записей')
    cur.close()
    conn.close()


# ================================================================
# 3. DICT: Загрузка себестоимости из XLSX
# ================================================================
def load_dict_cost():
    """Загружает себестоимость из XLSX в dict.cost_reference."""
    wb = openpyxl.load_workbook(XLSX_PATH)
    ws = wb.active

    conn = get_conn()
    cur = conn.cursor()

    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        nm_id = row[3]       # Артикул WB
        article = row[4]     # Артикул продавца
        cost = row[7]        # Текущая себестоимость
        date_str = row[2]    # Дата

        if nm_id is None or cost is None or cost == '':
            continue

        # Парсинг даты
        if date_str:
            if isinstance(date_str, datetime):
                valid_from = date_str.date()
            else:
                try:
                    valid_from = datetime.strptime(str(date_str), '%d.%m.%Y').date()
                except:
                    valid_from = datetime(2026, 1, 1).date()
        else:
            valid_from = datetime(2026, 1, 1).date()

        rows.append((
            int(nm_id),
            str(article) if article else '',
            float(cost),
            valid_from,
            datetime(2999, 12, 31).date(),
        ))

    sql = """
        INSERT INTO dict.cost_reference (nm_id, supplier_article, unit_cost, valid_from, valid_to)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    execute_values(cur, sql, rows)
    conn.commit()
    print(f'[DICT] cost_reference: загружено {len(rows)} записей')
    cur.close()
    conn.close()


def load_dict_extras():
    """Загружает типовые расходы и налоги."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO dict.extra_expenses (expense_date, expense_category, amount, comment) VALUES
        ('2026-01-15', 'Логистика',     15400.00, 'Доставка до склада WB, январь'),
        ('2026-02-01', 'Упаковка',       5200.00, 'Пакеты, короба, февраль'),
        ('2026-02-15', 'Логистика',     12800.00, 'Доставка до склада WB, февраль'),
        ('2026-03-01', 'Реклама',       22000.00, 'Промо-кампания март'),
        ('2026-03-10', 'Фото/контент',   7500.00, 'Съёмка новых артикулов'),
        ('2026-03-15', 'Логистика',     14200.00, 'Доставка до склада WB, март'),
        ('2026-04-01', 'Упаковка',       4900.00, 'Пакеты, короба, апрель'),
        ('2026-04-05', 'Реклама',       18500.00, 'Продвижение бренда RadSvet')
    """)

    cur.execute("""
        INSERT INTO dict.tax_reference (tax_name, tax_rate_percent, valid_from, valid_to) VALUES
        ('УСН 6%', 6.0000, '2026-01-01', '2999-12-31')
    """)

    conn.commit()
    print('[DICT] extra_expenses: 8 записей, tax_reference: 1 запись')
    cur.close()
    conn.close()


# ================================================================
# 4. MART: Агрегация данных в витрины
# ================================================================
def load_marts():
    """Агрегирует stg → mart."""
    conn = get_conn()
    cur = conn.cursor()

    # orders_daily
    cur.execute("""
        INSERT INTO mart.orders_daily (order_date, nm_id, supplier_article, subject, brand,
            orders_count, orders_amount, cancelled_count, avg_price)
        SELECT
            o.date::date, o.nm_id, o.supplier_article,
            MAX(o.subject), MAX(o.brand),
            COUNT(*), SUM(o.total_price),
            SUM(CASE WHEN o.is_cancel THEN 1 ELSE 0 END),
            ROUND(AVG(o.total_price), 2)
        FROM stg.wb_orders o
        GROUP BY o.date::date, o.nm_id, o.supplier_article
        ON CONFLICT (order_date, nm_id, supplier_article) DO UPDATE SET
            orders_count = EXCLUDED.orders_count,
            orders_amount = EXCLUDED.orders_amount,
            cancelled_count = EXCLUDED.cancelled_count,
            avg_price = EXCLUDED.avg_price,
            updated_at = now()
    """)
    print(f'[MART] orders_daily: {cur.rowcount} строк')

    # sales_daily
    cur.execute("""
        INSERT INTO mart.sales_daily (
            sales_date, nm_id, supplier_article, subject, brand,
            orders_count, sales_count, returns_count,
            gross_revenue, net_revenue, commission_amount,
            cost_amount, extra_expenses_amount, tax_amount,
            profit_amount, operating_profit_amount,
            avg_spp, avg_price_before_spp, avg_price_after_spp
        )
        SELECT
            s.date::date, s.nm_id, s.supplier_article,
            MAX(s.subject), MAX(s.brand),
            COUNT(*),
            SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END),
            SUM(CASE WHEN s.sale_id LIKE 'R%' THEN 1 ELSE 0 END),
            SUM(s.total_price),
            SUM(s.for_pay),
            ROUND(SUM(s.for_pay) * 0.15, 2),
            COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END),
            0,
            ROUND(SUM(s.for_pay) * 0.06, 2),
            SUM(s.for_pay)
                - ROUND(SUM(s.for_pay) * 0.15, 2)
                - COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END),
            SUM(s.for_pay)
                - ROUND(SUM(s.for_pay) * 0.15, 2)
                - COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END)
                - ROUND(SUM(s.for_pay) * 0.06, 2),
            ROUND(AVG(s.spp), 2),
            ROUND(AVG(s.total_price), 2),
            ROUND(AVG(s.for_pay), 2)
        FROM stg.wb_sales s
        LEFT JOIN dict.cost_reference cr
            ON s.nm_id = cr.nm_id
            AND s.date::date BETWEEN cr.valid_from AND cr.valid_to
        GROUP BY s.date::date, s.nm_id, s.supplier_article
        ON CONFLICT (sales_date, nm_id, supplier_article) DO UPDATE SET
            sales_count = EXCLUDED.sales_count,
            returns_count = EXCLUDED.returns_count,
            gross_revenue = EXCLUDED.gross_revenue,
            net_revenue = EXCLUDED.net_revenue,
            commission_amount = EXCLUDED.commission_amount,
            cost_amount = EXCLUDED.cost_amount,
            profit_amount = EXCLUDED.profit_amount,
            operating_profit_amount = EXCLUDED.operating_profit_amount,
            updated_at = now()
    """)
    print(f'[MART] sales_daily: {cur.rowcount} строк')

    # stocks_snapshot
    cur.execute("""
        INSERT INTO mart.stocks_snapshot (
            snapshot_date, nm_id, supplier_article, warehouse_name,
            subject, brand, quantity, in_way_to_client, in_way_from_client,
            quantity_full, price, discount
        )
        SELECT
            CURRENT_DATE, s.nm_id, s.supplier_article, s.warehouse_name,
            s.subject, s.brand, s.quantity, s.in_way_to_client, s.in_way_from_client,
            s.quantity_full, s.price, s.discount
        FROM stg.wb_stocks s
        ON CONFLICT (snapshot_date, nm_id, warehouse_name) DO UPDATE SET
            quantity = EXCLUDED.quantity,
            quantity_full = EXCLUDED.quantity_full,
            updated_at = now()
    """)
    print(f'[MART] stocks_snapshot: {cur.rowcount} строк')

    # app: тестовый пользователь
    cur.execute("""
        INSERT INTO app.users (username, password_hash, full_name)
        VALUES ('admin', '$2b$12$placeholder_hash_for_demo', 'Препелица П. П.')
        ON CONFLICT (username) DO NOTHING
    """)
    cur.execute("""
        INSERT INTO app.wb_accounts (account_name, api_token, is_active)
        VALUES ('RadSvet (Препелица4200)', 'token_hidden', true)
        ON CONFLICT DO NOTHING
    """)
    cur.execute("""
        INSERT INTO app.user_accounts (user_id, account_id)
        SELECT u.id, a.id FROM app.users u CROSS JOIN app.wb_accounts a
        WHERE u.username = 'admin'
        ON CONFLICT DO NOTHING
    """)
    print('[APP] users + wb_accounts + user_accounts: OK')

    conn.commit()
    cur.close()
    conn.close()


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print('=' * 50)
    print('WB Analytics — Загрузка реальных данных')
    print('=' * 50)

    # 1. RAW
    api_data = load_raw()

    # 2. STG
    load_stg_orders(api_data.get('orders', []))
    load_stg_sales(api_data.get('sales', []))
    load_stg_stocks(api_data.get('stocks', []))

    # 3. DICT
    load_dict_cost()
    load_dict_extras()

    # 4. MART
    load_marts()

    print()
    print('=' * 50)
    print('Загрузка завершена!')
    print('=' * 50)
