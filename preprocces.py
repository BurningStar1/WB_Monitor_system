import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы и конфигурация
# ---------------------------------------------------------------------------

WB_STATS_BASE = "https://statistics-api.wildberries.ru"

REQUIRED_FIELDS_ORDERS = {"srid", "date", "nmId", "supplierArticle", "priceWithDisc"}
REQUIRED_FIELDS_SALES = {"srid", "date", "nmId", "supplierArticle", "priceWithDisc", "saleID"}
REQUIRED_FIELDS_STOCKS = {"nmId", "supplierArticle", "quantity", "warehouseName"}
REQUIRED_FIELDS_COST = {"nm_id", "cost_price"}

NUMERIC_BOUNDS = {
    "priceWithDisc": (0.0, 10_000_000.0),
    "totalPrice":    (0.0, 10_000_000.0),
    "quantity":      (0,   100_000),
    "discountPercent": (0, 100),
}


def get_db_engine():
    dsn = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME')}"
    )
    return create_engine(dsn)


def get_wb_headers() -> dict:
    token = os.getenv("WB_API_TOKEN")
    if not token:
        raise EnvironmentError("WB_API_TOKEN не задан в переменных окружения")
    return {"Authorization": token, "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Шаг 1-2: Извлечение данных из WB API
# ---------------------------------------------------------------------------

def _fetch_with_pagination(endpoint: str, date_from: str, flag: int = 1) -> list:
    headers = get_wb_headers()
    params = {"dateFrom": date_from, "flag": flag}
    all_records = []

    while True:
        url = f"{WB_STATS_BASE}{endpoint}"
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("HTTP ошибка при запросе %s: %s", endpoint, exc)
            raise

        data = resp.json()
        if not data:
            break

        all_records.extend(data)

        last_change = data[-1].get("lastChangeDate")
        if not last_change:
            break
        params["dateFrom"] = last_change
        logger.info("Получено %d записей, продолжаем с %s", len(data), last_change)

    logger.info("Итого получено %d записей из %s", len(all_records), endpoint)
    return all_records


def fetch_orders(date_from: str) -> list:
    return _fetch_with_pagination("/api/v1/supplier/orders", date_from)


def fetch_sales(date_from: str) -> list:
    return _fetch_with_pagination("/api/v1/supplier/sales", date_from)


def fetch_stocks(date_from: str) -> list:
    return _fetch_with_pagination("/api/v1/supplier/stocks", date_from)


# ---------------------------------------------------------------------------
# Шаг 3: Чтение внутренних файлов продавца
# ---------------------------------------------------------------------------

def load_cost_reference(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл справочника не найден: {filepath}")

    df = pd.read_excel(filepath, dtype={"nm_id": str})

    missing = REQUIRED_FIELDS_COST - set(df.columns)
    if missing:
        raise ValueError(f"В файле себестоимости отсутствуют поля: {missing}")

    df["nm_id"] = df["nm_id"].str.strip()
    df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce")
    df["valid_from"] = pd.to_datetime(df.get("valid_from", pd.Timestamp("2000-01-01")), errors="coerce")
    df = df.dropna(subset=["nm_id", "cost_price"])
    logger.info("Загружено %d записей справочника себестоимости", len(df))
    return df


# ---------------------------------------------------------------------------
# Шаг 4: Сохранение в слой RAW
# ---------------------------------------------------------------------------

def save_to_raw(engine, records: list, source_type: str, loaded_at: datetime):
    if not records:
        logger.warning("Нет данных для сохранения в RAW (%s)", source_type)
        return

    rows = [
        {
            "source_type": source_type,
            "loaded_at": loaded_at.isoformat(),
            "payload": json.dumps(rec, ensure_ascii=False, default=str),
        }
        for rec in records
    ]
    df = pd.DataFrame(rows)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.wb_data (
                id         BIGSERIAL PRIMARY KEY,
                source_type VARCHAR(50) NOT NULL,
                loaded_at  TIMESTAMPTZ NOT NULL,
                payload    JSONB NOT NULL
            )
        """))

    df.to_sql("wb_data", engine, schema="raw", if_exists="append", index=False, method="multi")
    logger.info("RAW: сохранено %d записей типа '%s'", len(rows), source_type)


# ---------------------------------------------------------------------------
# Шаг 5: Проверка структуры
# ---------------------------------------------------------------------------

def validate_structure(records: list, required_fields: set, source_name: str) -> list:
    valid, invalid = [], 0
    for rec in records:
        missing = required_fields - rec.keys()
        if missing:
            invalid += 1
            logger.debug("Запись исключена (%s): отсутствуют поля %s", source_name, missing)
        else:
            valid.append(rec)

    if invalid:
        logger.warning(
            "%s: исключено %d записей из %d из-за отсутствия обязательных полей",
            source_name, invalid, len(records)
        )
    return valid


# ---------------------------------------------------------------------------
# Шаг 6: Очистка и нормализация
# ---------------------------------------------------------------------------

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит все колонки с датами к формату datetime64."""
    date_cols = [c for c in df.columns if "date" in c.lower() or "Date" in c]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _check_numeric_bounds(df: pd.DataFrame, bounds: dict, source_name: str) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for field, (lo, hi) in bounds.items():
        if field not in df.columns:
            continue
        col = pd.to_numeric(df[field], errors="coerce")
        out_of_range = (col < lo) | (col > hi) | col.isna()
        n_bad = out_of_range.sum()
        if n_bad:
            logger.warning(
                "%s: поле '%s' — %d записей вне диапазона [%s, %s], исключены",
                source_name, field, n_bad, lo, hi
            )
        mask &= ~out_of_range
    return df[mask].copy()


def clean_orders(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["nmId"] = df["nmId"].astype(str).str.strip()
    df["supplierArticle"] = df["supplierArticle"].astype(str).str.strip()
    df = _normalize_dates(df)
    df = _check_numeric_bounds(df, NUMERIC_BOUNDS, "orders")
    df["is_cancel"] = df.get("isCancel", False).fillna(False).astype(bool)
    logger.info("orders после очистки: %d записей", len(df))
    return df


def clean_sales(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["nmId"] = df["nmId"].astype(str).str.strip()
    df["supplierArticle"] = df["supplierArticle"].astype(str).str.strip()
    df = _normalize_dates(df)
    df = _check_numeric_bounds(df, NUMERIC_BOUNDS, "sales")
    df["operation_type"] = df["saleID"].str[:1].map({"S": "sale", "R": "return"}).fillna("unknown")
    logger.info("sales после очистки: %d записей", len(df))
    return df


def clean_stocks(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["nmId"] = df["nmId"].astype(str).str.strip()
    df["supplierArticle"] = df["supplierArticle"].astype(str).str.strip()
    df = _normalize_dates(df)
    df = _check_numeric_bounds(df, {"quantity": NUMERIC_BOUNDS["quantity"]}, "stocks")
    logger.info("stocks после очистки: %d записей", len(df))
    return df


# ---------------------------------------------------------------------------
# Устранение дублей
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame, key_cols: list, loaded_at_col: str = "lastChangeDate") -> pd.DataFrame:
    before = len(df)
    existing_cols = [c for c in key_cols if c in df.columns]
    if not existing_cols:
        return df

    if loaded_at_col in df.columns:
        df = df.sort_values(loaded_at_col, ascending=False)

    df = df.drop_duplicates(subset=existing_cols, keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Дедупликация: удалено %d дублирующихся записей", removed)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Согласование идентификаторов с внутренним справочником
# ---------------------------------------------------------------------------

def enrich_with_cost(
    df: pd.DataFrame,
    cost_ref: pd.DataFrame,
    period_col: str = "date"
) -> pd.DataFrame:
    if "nmId" not in df.columns or cost_ref is None or cost_ref.empty:
        return df

    period_col = period_col if period_col in df.columns else None
    result_rows = []

    for _, row in df.iterrows():
        nm = str(row["nmId"])
        candidates = cost_ref[cost_ref["nm_id"] == nm]

        if candidates.empty:
            row["cost_price"] = None
            result_rows.append(row)
            continue

        if period_col:
            op_date = row[period_col]
            if pd.notna(op_date):
                candidates = candidates[candidates["valid_from"] <= op_date]

        if candidates.empty:
            row["cost_price"] = None
        else:
            row["cost_price"] = candidates.sort_values("valid_from").iloc[-1]["cost_price"]

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    matched = result["cost_price"].notna().sum()
    logger.info(
        "Согласование себестоимости: %d из %d записей сопоставлены",
        matched, len(result)
    )
    return result


# ---------------------------------------------------------------------------
# Шаг 7: Загрузка в STG
# ---------------------------------------------------------------------------

def save_to_stg(engine, df: pd.DataFrame, table_name: str, loaded_at: datetime):
    if df.empty:
        logger.warning("STG: нет данных для загрузки в %s", table_name)
        return

    df = df.copy()
    df["stg_loaded_at"] = loaded_at

    for col in df.select_dtypes(include=["datetimetz", "datetime64[ns, UTC]"]).columns:
        df[col] = df[col].astype(str)

    df.to_sql(table_name, engine, schema="stg", if_exists="append", index=False, method="multi")
    logger.info("STG: загружено %d записей в stg.%s", len(df), table_name)


# ---------------------------------------------------------------------------
# Главная процедура обновления (Листинг 2.1)
# ---------------------------------------------------------------------------

def run_preprocessing(
    date_from: Optional[str] = None,
    cost_filepath: str = "cost_reference.xlsx",
):
    if date_from is None:
        date_from = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    loaded_at = datetime.utcnow()
    logger.info("=== Запуск предобработки. date_from=%s, loaded_at=%s ===", date_from, loaded_at)

    engine = get_db_engine()

    try:
        cost_ref = load_cost_reference(cost_filepath)
    except FileNotFoundError:
        logger.warning("Справочник себестоимости не найден, расчёт прибыли будет недоступен")
        cost_ref = None

    raw_orders = fetch_orders(date_from)
    save_to_raw(engine, raw_orders, "orders", loaded_at)
    valid_orders = validate_structure(raw_orders, REQUIRED_FIELDS_ORDERS, "orders")
    df_orders = clean_orders(valid_orders)
    df_orders = deduplicate(df_orders, key_cols=["srid", "nmId", "date"])
    df_orders = enrich_with_cost(df_orders, cost_ref)
    save_to_stg(engine, df_orders, "orders", loaded_at)

    raw_sales = fetch_sales(date_from)
    save_to_raw(engine, raw_sales, "sales", loaded_at)
    valid_sales = validate_structure(raw_sales, REQUIRED_FIELDS_SALES, "sales")
    df_sales = clean_sales(valid_sales)
    df_sales = deduplicate(df_sales, key_cols=["srid", "nmId", "saleID"])
    df_sales = enrich_with_cost(df_sales, cost_ref)
    save_to_stg(engine, df_sales, "sales", loaded_at)

    raw_stocks = fetch_stocks(date_from)
    save_to_raw(engine, raw_stocks, "stocks", loaded_at)
    valid_stocks = validate_structure(raw_stocks, REQUIRED_FIELDS_STOCKS, "stocks")
    df_stocks = clean_stocks(valid_stocks)
    df_stocks = deduplicate(df_stocks, key_cols=["nmId", "warehouseName"])
    save_to_stg(engine, df_stocks, "stocks", loaded_at)

    logger.info("=== Предобработка завершена успешно ===")
    return {
        "orders": len(df_orders),
        "sales": len(df_sales),
        "stocks": len(df_stocks),
        "loaded_at": loaded_at.isoformat(),
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print(result)
