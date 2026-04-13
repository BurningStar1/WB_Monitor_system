-- --- SEED DATA: заполнение всех слоёв реалистичными данными --- --
-- 5 товаров, ~30 дней (март 2026), все схемы

-- ============================================================
-- 1. RAW: 3 записи — по одной на каждый эндпоинт API
-- ============================================================
INSERT INTO raw.wb_api_payloads (source, endpoint, request_params, raw_payload)
VALUES
('wb_statistics', 'orders', '{"dateFrom": "2026-03-01T00:00:00"}',
 '[{"srid":"SR00000001","date":"2026-03-01T10:23:00","supplierArticle":"ART-1001","techSize":"42","barcode":"2000000000011","totalPrice":2490.00,"discountPercent":15,"warehouseName":"Коледино","oblast":"Московская","incomeID":101,"odid":5001,"nmId":100001,"subject":"Футболка","category":"Одежда","brand":"SportLine","isCancel":false},{"srid":"SR00000002","date":"2026-03-01T11:05:00","supplierArticle":"ART-1002","techSize":"0","barcode":"2000000000028","totalPrice":1890.00,"discountPercent":10,"warehouseName":"Подольск","oblast":"Московская","incomeID":102,"odid":5002,"nmId":100002,"subject":"Кроссовки","category":"Обувь","brand":"SportLine","isCancel":false}]'::jsonb),

('wb_statistics', 'sales', '{"dateFrom": "2026-03-01T00:00:00", "flag": 1}',
 '[{"saleID":"S0000000001","date":"2026-03-01T14:20:00","supplierArticle":"ART-1001","techSize":"42","barcode":"2000000000011","totalPrice":2490.00,"discountPercent":15,"isSupply":false,"isRealization":true,"promoCodeDiscount":0,"warehouseName":"Коледино","countryName":"Россия","oblastOkrugName":"Центральный","regionName":"Московская область","incomeID":101,"odid":5001,"spp":5.00,"forPay":2116.50,"finishedPrice":2116.50,"priceWithDisc":2116.50,"nmId":100001,"subject":"Футболка","category":"Одежда","brand":"SportLine","isStorno":false}]'::jsonb),

('wb_statistics', 'stocks', '{"dateFrom": "2026-03-01T00:00:00"}',
 '[{"warehouseName":"Коледино","supplierArticle":"ART-1001","nmId":100001,"barcode":"2000000000011","quantity":150,"inWayToClient":12,"inWayFromClient":3,"quantityFull":165,"category":"Одежда","subject":"Футболка","brand":"SportLine","techSize":"42","Price":2490.00,"Discount":15,"isSupply":true,"isRealization":false},{"warehouseName":"Подольск","supplierArticle":"ART-1002","nmId":100002,"barcode":"2000000000028","quantity":80,"inWayToClient":5,"inWayFromClient":1,"quantityFull":86,"category":"Обувь","subject":"Кроссовки","brand":"SportLine","techSize":"0","Price":1890.00,"Discount":10,"isSupply":true,"isRealization":false}]'::jsonb);


-- ============================================================
-- 2. STG: заказы (30 дней × 5 товаров = ~60 записей)
-- ============================================================
INSERT INTO stg.wb_orders (srid, date, last_change_date, supplier_article, tech_size, barcode, total_price, discount_percent, warehouse_name, oblast, income_id, odid, nm_id, subject, category, brand, is_cancel, source_loaded_at)
SELECT
    'SR' || LPAD((row_number() OVER ())::text, 10, '0') AS srid,
    d::timestamptz + (random() * interval '12 hours') AS date,
    d::timestamptz + (random() * interval '12 hours') + interval '1 hour' AS last_change_date,
    a.supplier_article,
    a.tech_size,
    a.barcode,
    a.base_price AS total_price,
    a.discount AS discount_percent,
    (ARRAY['Коледино','Подольск','Казань','Краснодар','Хабаровск'])[1 + (random()*4)::int] AS warehouse_name,
    (ARRAY['Московская','Ленинградская','Свердловская','Краснодарский край','Новосибирская'])[1 + (random()*4)::int] AS oblast,
    (1000 + (random()*999)::int) AS income_id,
    (5000 + row_number() OVER ()) AS odid,
    a.nm_id,
    a.subject,
    a.category,
    a.brand,
    (random() < 0.08) AS is_cancel,
    now() AS source_loaded_at
FROM generate_series('2026-03-01'::date, '2026-03-30'::date, '1 day') d
CROSS JOIN (VALUES
    ('ART-1001', '42',  '2000000000011', 2490.00, 15.0, 100001, 'Футболка',      'Одежда',       'SportLine'),
    ('ART-1002', '0',   '2000000000028', 1890.00, 10.0, 100002, 'Кроссовки',     'Обувь',        'SportLine'),
    ('ART-2001', 'M',   '2000000000035', 3490.00, 20.0, 100003, 'Куртка',        'Верхняя одежда','UrbanWear'),
    ('ART-2002', 'L',   '2000000000042', 990.00,   5.0, 100004, 'Шорты',         'Одежда',       'UrbanWear'),
    ('ART-3001', 'ONE', '2000000000059', 590.00,  12.0, 100005, 'Бейсболка',     'Аксессуары',   'CapStyle')
) AS a(supplier_article, tech_size, barcode, base_price, discount, nm_id, subject, category, brand)
-- по 1-4 заказа в день на товар
WHERE random() < 0.6
ON CONFLICT (srid) DO NOTHING;


-- ============================================================
-- 3. STG: продажи (генерируем из заказов, ~70% конверсия)
-- ============================================================
INSERT INTO stg.wb_sales (sale_id, date, last_change_date, supplier_article, tech_size, barcode, total_price, discount_percent, is_supply, is_realization, promo_code_discount, warehouse_name, country_name, oblast_okrug_name, region_name, income_id, odid, spp, for_pay, finished_price, price_with_disc, nm_id, subject, category, brand, is_storno, source_loaded_at)
SELECT
    CASE WHEN random() < 0.92 THEN 'S' ELSE 'R' END || LPAD((row_number() OVER ())::text, 10, '0') AS sale_id,
    o.date + interval '2 hours' + (random() * interval '6 hours'),
    o.last_change_date + interval '3 hours',
    o.supplier_article,
    o.tech_size,
    o.barcode,
    o.total_price,
    o.discount_percent,
    false,
    true,
    CASE WHEN random() < 0.15 THEN ROUND((random() * 200)::numeric, 2) ELSE 0 END,
    o.warehouse_name,
    'Россия',
    'Центральный',
    o.oblast,
    o.income_id,
    o.odid,
    ROUND((random() * 10)::numeric, 2) AS spp,
    ROUND(o.total_price * (1 - o.discount_percent/100) * 0.85, 2) AS for_pay,
    ROUND(o.total_price * (1 - o.discount_percent/100), 2) AS finished_price,
    ROUND(o.total_price * (1 - o.discount_percent/100), 2) AS price_with_disc,
    o.nm_id,
    o.subject,
    o.category,
    o.brand,
    false,
    now()
FROM stg.wb_orders o
WHERE o.is_cancel = false AND random() < 0.75
ON CONFLICT (sale_id) DO NOTHING;


-- ============================================================
-- 4. STG: остатки (последний срез по 5 складам × 5 товаров)
-- ============================================================
INSERT INTO stg.wb_stocks (warehouse_name, supplier_article, nm_id, barcode, quantity, in_way_to_client, in_way_from_client, quantity_full, category, subject, brand, tech_size, price, discount, is_supply, is_realization, source_loaded_at)
SELECT
    w.name,
    a.supplier_article,
    a.nm_id,
    a.barcode,
    (50 + (random()*200)::int) AS quantity,
    (random()*20)::int AS in_way_to_client,
    (random()*5)::int AS in_way_from_client,
    (50 + (random()*200)::int + (random()*20)::int + (random()*5)::int) AS quantity_full,
    a.category,
    a.subject,
    a.brand,
    a.tech_size,
    a.base_price,
    a.discount,
    true,
    false,
    now()
FROM (VALUES
    ('ART-1001', '42',  '2000000000011', 2490.00, 15.0, 100001, 'Футболка',  'Одежда',        'SportLine'),
    ('ART-1002', '0',   '2000000000028', 1890.00, 10.0, 100002, 'Кроссовки', 'Обувь',         'SportLine'),
    ('ART-2001', 'M',   '2000000000035', 3490.00, 20.0, 100003, 'Куртка',    'Верхняя одежда', 'UrbanWear'),
    ('ART-2002', 'L',   '2000000000042', 990.00,   5.0, 100004, 'Шорты',     'Одежда',        'UrbanWear'),
    ('ART-3001', 'ONE', '2000000000059', 590.00,  12.0, 100005, 'Бейсболка', 'Аксессуары',    'CapStyle')
) AS a(supplier_article, tech_size, barcode, base_price, discount, nm_id, subject, category, brand)
CROSS JOIN (VALUES ('Коледино'),('Подольск'),('Казань'),('Краснодар'),('Хабаровск')) AS w(name);


-- ============================================================
-- 5. DICT: себестоимость
-- ============================================================
INSERT INTO dict.cost_reference (nm_id, supplier_article, unit_cost, valid_from, valid_to) VALUES
(100001, 'ART-1001',  620.00, '2026-01-01', '2999-12-31'),
(100002, 'ART-1002',  780.00, '2026-01-01', '2999-12-31'),
(100003, 'ART-2001', 1450.00, '2026-01-01', '2999-12-31'),
(100004, 'ART-2002',  310.00, '2026-01-01', '2999-12-31'),
(100005, 'ART-3001',  145.00, '2026-01-01', '2999-12-31');


-- ============================================================
-- 6. DICT: дополнительные расходы
-- ============================================================
INSERT INTO dict.extra_expenses (expense_date, expense_category, amount, comment) VALUES
('2026-03-01', 'Логистика',  12500.00, 'Доставка до склада WB, март'),
('2026-03-01', 'Упаковка',    4800.00, 'Пакеты, короба, март'),
('2026-03-10', 'Реклама',    18000.00, 'Промо-кампания топ-карточки ART-1001'),
('2026-03-15', 'Фото/контент', 6500.00, 'Съёмка новых артикулов'),
('2026-03-20', 'Логистика',   9200.00, 'Доставка допартии на склад'),
('2026-03-25', 'Реклама',    15000.00, 'Продвижение бренда UrbanWear');


-- ============================================================
-- 7. DICT: налоговые ставки
-- ============================================================
INSERT INTO dict.tax_reference (tax_name, tax_rate_percent, valid_from, valid_to) VALUES
('УСН 6%',   6.0000, '2026-01-01', '2999-12-31'),
('НДС 20%', 20.0000, '2026-01-01', '2999-12-31');


-- ============================================================
-- 8. MART: orders_daily (агрегация из stg.wb_orders)
-- ============================================================
INSERT INTO mart.orders_daily (order_date, nm_id, supplier_article, subject, brand, orders_count, orders_amount, cancelled_count, avg_price)
SELECT
    o.date::date AS order_date,
    o.nm_id,
    o.supplier_article,
    MAX(o.subject),
    MAX(o.brand),
    COUNT(*) AS orders_count,
    SUM(o.total_price) AS orders_amount,
    SUM(CASE WHEN o.is_cancel THEN 1 ELSE 0 END) AS cancelled_count,
    ROUND(AVG(o.total_price), 2) AS avg_price
FROM stg.wb_orders o
GROUP BY o.date::date, o.nm_id, o.supplier_article
ON CONFLICT (order_date, nm_id, supplier_article) DO UPDATE SET
    orders_count = EXCLUDED.orders_count,
    orders_amount = EXCLUDED.orders_amount,
    cancelled_count = EXCLUDED.cancelled_count,
    avg_price = EXCLUDED.avg_price,
    updated_at = now();


-- ============================================================
-- 9. MART: sales_daily (агрегация из stg.wb_sales + dict)
-- ============================================================
INSERT INTO mart.sales_daily (
    sales_date, nm_id, supplier_article, subject, brand,
    orders_count, sales_count, returns_count,
    gross_revenue, net_revenue, commission_amount,
    cost_amount, extra_expenses_amount, tax_amount,
    profit_amount, operating_profit_amount,
    avg_spp, avg_price_before_spp, avg_price_after_spp
)
SELECT
    s.date::date AS sales_date,
    s.nm_id,
    s.supplier_article,
    MAX(s.subject),
    MAX(s.brand),
    COUNT(*) AS orders_count,
    SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END) AS sales_count,
    SUM(CASE WHEN s.sale_id LIKE 'R%' THEN 1 ELSE 0 END) AS returns_count,
    SUM(s.total_price) AS gross_revenue,
    SUM(s.for_pay) AS net_revenue,
    ROUND(SUM(s.for_pay) * 0.15, 2) AS commission_amount,
    COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END) AS cost_amount,
    0 AS extra_expenses_amount,
    ROUND(SUM(s.for_pay) * 0.06, 2) AS tax_amount,
    SUM(s.for_pay)
        - ROUND(SUM(s.for_pay) * 0.15, 2)
        - COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END)
    AS profit_amount,
    SUM(s.for_pay)
        - ROUND(SUM(s.for_pay) * 0.15, 2)
        - COALESCE(MAX(cr.unit_cost), 0) * SUM(CASE WHEN s.sale_id LIKE 'S%' THEN 1 ELSE 0 END)
        - ROUND(SUM(s.for_pay) * 0.06, 2)
    AS operating_profit_amount,
    ROUND(AVG(s.spp), 2) AS avg_spp,
    ROUND(AVG(s.total_price), 2) AS avg_price_before_spp,
    ROUND(AVG(s.for_pay), 2) AS avg_price_after_spp
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
    updated_at = now();


-- ============================================================
-- 10. MART: stocks_snapshot
-- ============================================================
INSERT INTO mart.stocks_snapshot (snapshot_date, nm_id, supplier_article, warehouse_name, subject, brand, quantity, in_way_to_client, in_way_from_client, quantity_full, price, discount)
SELECT
    CURRENT_DATE AS snapshot_date,
    s.nm_id,
    s.supplier_article,
    s.warehouse_name,
    s.subject,
    s.brand,
    s.quantity,
    s.in_way_to_client,
    s.in_way_from_client,
    s.quantity_full,
    s.price,
    s.discount
FROM stg.wb_stocks s
ON CONFLICT (snapshot_date, nm_id, warehouse_name) DO UPDATE SET
    quantity = EXCLUDED.quantity,
    quantity_full = EXCLUDED.quantity_full,
    updated_at = now();


-- ============================================================
-- 11. APP: тестовые пользователи и аккаунт
-- ============================================================
INSERT INTO app.users (username, password_hash, full_name) VALUES
('admin',    '$2b$12$LJ3m5Zq8vK9X1234567890abcdefghijklmnopqrstuvwxyz', 'Препелица П. П.'),
('analyst1', '$2b$12$AB3c4Dq8vK9X1234567890abcdefghijklmnopqrstuvwxyz', 'Иванов И. И.')
ON CONFLICT (username) DO NOTHING;

INSERT INTO app.wb_accounts (account_name, api_token) VALUES
('SportLine Official', 'wbstat_test_token_xxxxxxxxxxxxx'),
('UrbanWear Store',    'wbstat_test_token_yyyyyyyyyyyyy')
ON CONFLICT DO NOTHING;

INSERT INTO app.user_accounts (user_id, account_id)
SELECT u.id, a.id
FROM app.users u
CROSS JOIN app.wb_accounts a
WHERE u.username = 'admin'
ON CONFLICT DO NOTHING;
