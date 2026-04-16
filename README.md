# WB Analytics — аналитический сервис для селлера Wildberries

Streamlit-приложение и ETL-конвейер, собирающий данные Wildberries
(Seller API: «Статистика», «Финансовые отчёты», «Продвижение») в PostgreSQL
и считающий 19 управленческих отчётов: дашборд, ОПИУ, артикулы,
рентабельность, прогноз, рекламу, ABC/когорты, РнП, алёрты и т. д.

Финансовые формулы выровнены с xlsx-методологией RASK ОПИУ селлера
копейка-в-копейку (25 месяцев, 2024-04 → 2026-04).

## 1. Архитектура

```
Wildberries API         PostgreSQL                  Streamlit
 orders/sales/stocks   raw   -> stg  -> mart   ─>   19 страниц
 finance                JSONB    типы      витрины   отчётов
 ads (promotion)                 ключи     PK/UPSERT
```

- **raw** — `raw.wb_api_payloads`: JSONB-снимки всех API-ответов
  (append-only, дедуп на уровне stg).
- **stg** — нормализованные таблицы `stg.wb_orders`, `wb_sales`,
  `wb_stocks`, `wb_finance_detail`, `wb_ads_daily`. Загружаются UPSERT'ом
  по естественным ключам; `wb_stocks` — через `DISTINCT ON` + `TRUNCATE`.
- **mart** — 5 витрин с ежедневной агрегацией: `orders_daily`, `sales_daily`,
  `stocks_snapshot`, `finance_daily`, `ads_daily`. Все идемпотентны
  (`ON CONFLICT ... DO UPDATE`).
- **marts/queries.py** — SQL для отчётов: `PNL_MONTHLY`, `FIN_WEEKLY`,
  `FIN_PROFIT`, `FIN_ARTICLE`, `FINANCE_DAILY`, `FORECAST_DAILY` и др.
- **app/** — Streamlit: `Home.py` + 19 страниц в `app/pages/`.
  Общие утилиты — `app/styles.py` (Plotly-конфиг, таблицы, фильтры).

## 2. Структура репозитория

| Путь | Назначение |
|------|-----------|
| `api/` | Клиент Wildberries Seller API (`wb_client.py`, `endpoints.py`) |
| `config/` | Настройки из `.env` через Pydantic-like `Settings` |
| `db/` | SQLAlchemy engine + репозитории |
| `etl/` | `RawLoader`, `StgLoader`, `MartLoader` |
| `marts/` | Каталог SQL-запросов для отчётов |
| `app/` | Streamlit-приложение (Home + 19 страниц) |
| `sql/` | DDL-скрипты схем raw/stg/dict/mart + индексы |
| `scripts/` | `init_db.py`, `run_etl.py`, `load_internal_dicts.py` |
| `tests/` | pytest: формулы, DQC, идемпотентность ETL |
| `checks/` | Контроль качества данных (`assert_not_null`, `assert_non_negative`) |
| `utils/` | Логирование |

## 3. Быстрый старт

### 3.1. Требования

- Python 3.12
- PostgreSQL 14+ (проверено на 16)
- Доступ к Wildberries Seller API (JWT-токен)

### 3.2. Установка зависимостей

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Настройка `.env`

```bash
cp .env.example .env
```

В файле `.env` заполнить:

- `WB_API_TOKEN` — JWT от WB (опционально, можно задать через UI).
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` — подключение
  к PostgreSQL.
- `APP_USERS` — пары `логин:пароль` для входа в Streamlit через запятую.

### 3.4. Создание БД и схем

Создать БД:

```bash
psql -U postgres -h localhost -c "CREATE DATABASE wb_analytics;"
```

Накатить схемы, индексы и представления:

```bash
python scripts/init_db.py
```

Скрипт проигрывает `sql/01_schema_raw.sql`, `02_schema_stg.sql`,
`03_schema_dict.sql`, `04_schema_mart.sql`, `05_views_and_marts.sql`,
`06_indexes.sql` в одной транзакции.

### 3.5. (Опционально) Загрузить справочники из XLSX

```bash
python scripts/load_internal_dicts.py \
  --cost data/internal/cost_reference.xlsx \
  --expenses data/internal/extra_expenses.xlsx \
  --tax data/internal/tax_reference.xlsx
```

Справочники покрывают себестоимость по `nm_id × датам`, прочие расходы
(курьер, образцы) и ставки УСН по датам (6 % с 2026-01-01).

### 3.6. Сбор данных (ETL)

Обновить за 30 дней:

```bash
python scripts/run_etl.py --days-back 30
```

Последовательность: `RawLoader` (скачивает JSON в `raw.wb_api_payloads`)
→ `StgLoader` (парсит JSONB в stg-таблицы) → `MartLoader` (строит витрины).
Все шаги идемпотентны — повторный запуск не создаёт дублей.

Параметры:

- `--days-back N` — сколько дней в прошлое
  (по умолчанию 30; API отдаёт до ~6 мес. для заказов/продаж и до 2 лет
  для финансов и рекламы).
- `--skip-ads` — пропустить рекламу (медленный rate limit).

ETL можно запускать и через UI: `Home.py → «Данные и API»`.

### 3.7. Запуск приложения

```bash
streamlit run app/Home.py --server.port 8503
```

Открыть `http://localhost:8503/`.

**Учётки по умолчанию:**
- `admin` / `admin123`
- `analyst` / `wb2024`

### 3.8. Тесты

```bash
pytest -q
```

Сейчас 23 теста:

- `test_sql_formulas.py` — 10 тестов на согласованность всех финансовых
  витрин (SUM(FIN_PROFIT) = SUM(FIN_ARTICLE) = SUM(FINANCE_DAILY)
  = PNL_MONTHLY = KPI-дашборд) за каждый из 25 месяцев.
- `test_etl_idempotent.py` — 10 тестов на уникальность PK/ключей в
  stg и mart (защита от дублей).
- `test_quality_checks.py` — 2 теста DQC-утилит.
- `test_settings.py` — конфиг.

## 4. Страницы приложения

| # | Страница | Суть |
|---|----------|------|
| — | Home | Статус данных, управление API-ключом, запуск ETL, алёрты |
| 01 | KPI-дашборд | Карточки + спарклайны + клик-модалки с детальным графиком |
| 02 | Еженедельный отчёт | Динамика по неделям (ISO) |
| 03 | Отчёт по артикулам | Детализация: заказы, продажи, остатки, прибыль |
| 04 | Остатки на складах | Распределение, оборачиваемость, капитализация |
| 05 | ABC-анализ | Мульти-метрика: выручка × прибыль × заказы |
| 06 | Рентабельность | Waterfall (Реализация → Удержания WB → Прибыль) |
| 07 | Отчёт за период | Помесячная сводка |
| 08 | Прогноз | Прогноз заказов и прибыли (скользящая средняя) |
| 09 | Потребность | Расчёт поставки по скорости продаж |
| 10 | Калькулятор акций | Оценка участия в промо-акциях WB |
| 11 | ОПИУ | Отчёт о прибылях и убытках (xlsx-RASK) |
| 12 | Неделя к неделе | Сравнение двух недель |
| 13 | Конверсия рекламы | CTR, CPC, DRR, ROAS по кампаниям |
| 14 | Справочники | Редактирование cost/extras/tax прямо из UI |
| 15 | РнП | «Рука на Пульсе» — ежедневная матрица артикулов |
| 16 | Алёрты | Низкие остатки, убыточные, падение спроса |
| 17 | Юнит-экономика | CM1 / CM2 / CM3 на один юнит |
| 18 | Когорты | Возраст артикула и retention |
| 19 | Качество данных | Проверки полноты и целостности витрин |

## 5. Финансовая методология (canon)

Все отчёты используют одну формулу прибыли, эталон — xlsx «ОПИУ» селлера:

```
Реализация после СПП   = SUM(retail_amount)                -- из WB Finance
К перечислению (ppvz)  = SUM(ppvz_for_pay)                 -- агрегат без комиссии
Комиссия               = Реализация после СПП − ppvz
Услуги WB              = logistics + storage + penalty
                         + acceptance + deduction - additional
   ⚠ commission уже в ppvz_for_pay, НЕ вычитается повторно
   ⚠ acquiring xlsx НЕ учитывает — исключён
   ⚠ ads_spend — часть deduction_amount, НЕ вычитается повторно
Себестоимость          = SUM(unit_cost × (sales_count − returns_count))
Прочие расходы         = SUM(extras.amount)                 -- курьер, образцы
Pre-tax                = ppvz − Услуги WB − Себестоимость − Прочие
Налог                  = GREATEST(SUM(Pre-tax), 0) × rate   -- УСН, агрегатно
Чистая прибыль         = Pre-tax − Налог
```

Налог считается двумя способами:
- **Агрегатный** (`GREATEST(SUM(pre_tax),0) × rate`) — для PNL/FIN_WEEKLY/KPI.
- **Аддитивный** (`pre_tax × rate` на строку) — для FIN_PROFIT/FIN_ARTICLE,
  чтобы SUM по строкам = агрегат.

## 6. Безопасность

- Токен WB хранится в `.env` или `wb_api_key.txt` (оба в `.gitignore`).
- Пароли пользователей в `.env` (`APP_USERS`). В проде — поднять auth-слой
  (OAuth/LDAP) или перенести на Nginx basic-auth.
- Доступ к PostgreSQL — по параметрам из `.env`, шифрование — на стороне
  СУБД (SSL-сертификат).

## 7. Что дальше

- Валидация Wildberries-схем через Pydantic (сейчас — `jsonb_typeof` проверка).
- Авто-расписание ETL (cron / Airflow / GitHub Actions).
- Экспорт отчётов в PDF / XLSX.
- Метрики качества (в приложении есть раздел «Качество данных»).

## 8. Лицензия и авторство

ВКР бакалавриата МИРЭА, 2026. Автор — Препелица П. П. (ИНБО-05-22).
Для учебных и демонстрационных целей.
