import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Описание ожидаемой схемы данных (структурная согласованность)
# ---------------------------------------------------------------------------

EXPECTED_SCHEMA = {
    "orders": {
        "srid":             "object",
        "date":             "datetime64[ns, UTC]",
        "lastChangeDate":   "datetime64[ns, UTC]",
        "nmId":             "object",
        "supplierArticle":  "object",
        "priceWithDisc":    "float64",
        "warehouseName":    "object",
        "is_cancel":        "bool",
    },
    "sales": {
        "srid":             "object",
        "date":             "datetime64[ns, UTC]",
        "nmId":             "object",
        "supplierArticle":  "object",
        "saleID":           "object",
        "priceWithDisc":    "float64",
        "operation_type":   "object",
    },
    "stocks": {
        "nmId":             "object",
        "supplierArticle":  "object",
        "quantity":         "int64",
        "warehouseName":    "object",
    },
}

REQUIRED_FOR_ANALYTICS = {
    "orders": {"srid", "date", "nmId", "priceWithDisc"},
    "sales":  {"srid", "date", "nmId", "priceWithDisc", "saleID", "operation_type"},
    "stocks": {"nmId", "quantity", "warehouseName"},
}

NUMERIC_BOUNDS = {
    "priceWithDisc":   (0.0, 10_000_000.0),
    "totalPrice":      (0.0, 10_000_000.0),
    "quantity":        (0,   100_000),
    "discountPercent": (0,   100),
    "cost_price":      (0.0, 10_000_000.0),
}


# ---------------------------------------------------------------------------
# Структура отчёта о качестве
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    source_name: str
    total_records: int
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    n_required_values: int = 0
    n_filled_values: int = 0
    k_completeness: float = 0.0

    n_duplicates: int = 0
    k_uniqueness: float = 0.0

    schema_errors: list = field(default_factory=list)

    n_out_of_range: int = 0
    n_invalid_dates: int = 0
    logical_errors: list = field(default_factory=list)

    analytics_ready: bool = False
    missing_for_analytics: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "source": self.source_name,
            "total": self.total_records,
            "k_completeness": round(self.k_completeness, 4),
            "k_uniqueness": round(self.k_uniqueness, 4),
            "schema_errors": self.schema_errors,
            "n_out_of_range": self.n_out_of_range,
            "n_invalid_dates": self.n_invalid_dates,
            "logical_errors": self.logical_errors,
            "analytics_ready": self.analytics_ready,
            "missing_for_analytics": self.missing_for_analytics,
            "checked_at": self.checked_at,
        }


# ---------------------------------------------------------------------------
# 1. Коэффициент полноты: K_full = N_filled / N_required
# ---------------------------------------------------------------------------

def calc_completeness(df: pd.DataFrame, required_fields: set) -> tuple[float, int, int]:
    present_fields = [f for f in required_fields if f in df.columns]
    if not present_fields:
        return 0.0, 0, len(required_fields) * len(df)

    n_required = len(required_fields) * len(df)
    n_filled = 0

    for field_name in required_fields:
        if field_name in df.columns:
            n_filled += int(df[field_name].notna().sum())

    k_full = n_filled / n_required if n_required > 0 else 0.0
    return k_full, n_filled, n_required


# ---------------------------------------------------------------------------
# 2. Коэффициент уникальности: K_uniq = 1 - N_dup / N
# ---------------------------------------------------------------------------

def calc_uniqueness(df: pd.DataFrame, key_cols: list) -> tuple[float, int]:
    if df.empty:
        return 1.0, 0

    existing = [c for c in key_cols if c in df.columns]
    if not existing:
        return 1.0, 0

    n_total = len(df)
    n_unique = df.drop_duplicates(subset=existing).shape[0]
    n_dup = n_total - n_unique
    k_uniq = 1.0 - n_dup / n_total if n_total > 0 else 1.0
    return k_uniq, n_dup


# ---------------------------------------------------------------------------
# 3. Структурная согласованность
# ---------------------------------------------------------------------------

def check_schema(df: pd.DataFrame, source_name: str) -> list:
    errors = []
    expected = EXPECTED_SCHEMA.get(source_name, {})

    for field_name, expected_type in expected.items():
        if field_name not in df.columns:
            errors.append(f"Поле '{field_name}' отсутствует в наборе данных")
            continue

        actual_dtype = str(df[field_name].dtype)

        if "datetime" in expected_type and "datetime" not in actual_dtype:
            errors.append(
                f"Поле '{field_name}': ожидался datetime, получен {actual_dtype}"
            )
        elif expected_type in ("float64", "int64") and "float" not in actual_dtype and "int" not in actual_dtype:
            errors.append(
                f"Поле '{field_name}': ожидался числовой тип, получен {actual_dtype}"
            )

    return errors


# ---------------------------------------------------------------------------
# 4. Логическая непротиворечивость
# ---------------------------------------------------------------------------

def check_logical_validity(df: pd.DataFrame, source_name: str) -> tuple[int, int, list]:
    errors = []
    n_out_of_range = 0
    n_invalid_dates = 0

    for field_name, (lo, hi) in NUMERIC_BOUNDS.items():
        if field_name not in df.columns:
            continue
        col = pd.to_numeric(df[field_name], errors="coerce")
        out = ((col < lo) | (col > hi)).sum()
        if out > 0:
            n_out_of_range += int(out)
            errors.append(
                f"Поле '{field_name}': {out} значений вне допустимого диапазона [{lo}, {hi}]"
            )

    date_fields = [c for c in df.columns if "date" in c.lower() or "Date" in c]
    for field_name in date_fields:
        col = pd.to_datetime(df[field_name], errors="coerce", utc=True)
        n_nat = int(col.isna().sum())
        if n_nat > 0:
            n_invalid_dates += n_nat
            errors.append(
                f"Поле '{field_name}': {n_nat} записей с некорректной датой (NaT)"
            )

    if source_name == "sales" and "date" in df.columns:
        now = pd.Timestamp.utcnow()
        col = pd.to_datetime(df["date"], errors="coerce", utc=True)
        future = (col > now).sum()
        if future > 0:
            errors.append(f"Поле 'date' (sales): {future} записей с датой в будущем")

    return n_out_of_range, n_invalid_dates, errors


# ---------------------------------------------------------------------------
# 5. Пригодность для аналитики
# ---------------------------------------------------------------------------

def check_analytics_readiness(
    orders: pd.DataFrame,
    sales: pd.DataFrame,
    stocks: pd.DataFrame,
    cost_ref: Optional[pd.DataFrame] = None,
) -> tuple[bool, list]:

    issues = []

    for df, name in [(orders, "orders"), (sales, "sales"), (stocks, "stocks")]:
        required = REQUIRED_FOR_ANALYTICS[name]
        missing = required - set(df.columns)
        if missing:
            issues.append(f"{name}: отсутствуют поля для аналитики: {missing}")

    if "date" in sales.columns and not sales.empty:
        col = pd.to_datetime(sales["date"], errors="coerce", utc=True).dropna()
        if not col.empty:
            date_range = (col.max() - col.min()).days
            if date_range < 7:
                issues.append(
                    f"Данные о продажах покрывают только {date_range} дней "
                    f"(минимум 7 для недельных срезов)"
                )

    if cost_ref is None or cost_ref.empty:
        issues.append(
            "Справочник себестоимости отсутствует — расчёт прибыли и операционной прибыли недоступен"
        )
    elif "cost_price" in sales.columns:
        matched_pct = sales["cost_price"].notna().mean() * 100
        if matched_pct < 50:
            issues.append(
                f"Только {matched_pct:.1f}% записей продаж имеют данные о себестоимости"
            )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Главная процедура оценки качества
# ---------------------------------------------------------------------------

def run_quality_check(
    df_orders: pd.DataFrame,
    df_sales: pd.DataFrame,
    df_stocks: pd.DataFrame,
    cost_ref: Optional[pd.DataFrame] = None,
) -> dict:
    reports = {}

    datasets = {
        "orders": (df_orders, ["srid", "nmId", "date"]),
        "sales":  (df_sales,  ["srid", "nmId", "saleID"]),
        "stocks": (df_stocks, ["nmId", "warehouseName"]),
    }

    for source_name, (df, key_cols) in datasets.items():
        report = QualityReport(
            source_name=source_name,
            total_records=len(df),
        )

        if df.empty:
            logger.warning("Набор данных '%s' пуст", source_name)
            reports[source_name] = report.summary()
            continue

        required = REQUIRED_FOR_ANALYTICS[source_name]
        k_full, n_filled, n_required = calc_completeness(df, required)
        report.k_completeness = k_full
        report.n_filled_values = n_filled
        report.n_required_values = n_required

        k_uniq, n_dup = calc_uniqueness(df, key_cols)
        report.k_uniqueness = k_uniq
        report.n_duplicates = n_dup

        report.schema_errors = check_schema(df, source_name)

        n_range, n_dates, log_errors = check_logical_validity(df, source_name)
        report.n_out_of_range = n_range
        report.n_invalid_dates = n_dates
        report.logical_errors = log_errors

        reports[source_name] = report.summary()

        logger.info(
            "%s — K_full=%.4f, K_uniq=%.4f, схема: %d ошибок, диапазон: %d записей вне границ",
            source_name, k_full, k_uniq,
            len(report.schema_errors), n_range
        )

    analytics_ready, analytics_issues = check_analytics_readiness(
        df_orders, df_sales, df_stocks, cost_ref
    )
    reports["analytics_readiness"] = {
        "ready": analytics_ready,
        "issues": analytics_issues,
    }

    if analytics_ready:
        logger.info("Данные пригодны для формирования витринного слоя")
    else:
        for issue in analytics_issues:
            logger.warning("Проблема с пригодностью данных: %s", issue)

    return reports


# ---------------------------------------------------------------------------
# Опциональное сохранение отчёта в БД
# ---------------------------------------------------------------------------

def save_quality_report(engine, reports: dict, loaded_at: datetime):
    import json
    row = {
        "checked_at": loaded_at.isoformat(),
        "report": json.dumps(reports, ensure_ascii=False, default=str),
    }
    df = pd.DataFrame([row])
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw.quality_log (
                id         BIGSERIAL PRIMARY KEY,
                checked_at TIMESTAMPTZ NOT NULL,
                report     JSONB NOT NULL
            )
        """))
    df.to_sql("quality_log", engine, schema="raw", if_exists="append", index=False)
    logger.info("Отчёт о качестве сохранён в raw.quality_log")


# ---------------------------------------------------------------------------
# Вспомогательная функция: печать читаемого отчёта
# ---------------------------------------------------------------------------

def print_quality_report(reports: dict):
    print("\n" + "=" * 60)
    print("ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ")
    print("=" * 60)

    for source, data in reports.items():
        if source == "analytics_readiness":
            continue
        print(f"\n[{source.upper()}]  записей: {data['total']}")
        print(f"  Полнота      K_full  = {data['k_completeness']:.4f}")
        print(f"  Уникальность K_uniq  = {data['k_uniqueness']:.4f}")
        if data["schema_errors"]:
            print("  Ошибки схемы:")
            for e in data["schema_errors"]:
                print(f"    - {e}")
        if data["logical_errors"]:
            print("  Логические ошибки:")
            for e in data["logical_errors"]:
                print(f"    - {e}")

    ar = reports.get("analytics_readiness", {})
    print(f"\n[ПРИГОДНОСТЬ ДЛЯ АНАЛИТИКИ]  {'✓ Данные готовы' if ar.get('ready') else '✗ Есть проблемы'}")
    for issue in ar.get("issues", []):
        print(f"  - {issue}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Точка входа для самостоятельного запуска
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from appendix_a_preprocessing import (
        get_db_engine,
        load_cost_reference,
    )

    engine = get_db_engine()

    df_orders = pd.read_sql("SELECT * FROM stg.orders", engine)
    df_sales  = pd.read_sql("SELECT * FROM stg.sales",  engine)
    df_stocks = pd.read_sql("SELECT * FROM stg.stocks", engine)

    try:
        cost_ref = load_cost_reference("cost_reference.xlsx")
    except FileNotFoundError:
        cost_ref = None

    reports = run_quality_check(df_orders, df_sales, df_stocks, cost_ref)
    print_quality_report(reports)
    save_quality_report(engine, reports, datetime.utcnow())
