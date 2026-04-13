# --- Приложение А — Демонстрация предобработки данных WB API --- #
# Запуск: Google Colab или локально
# Результат: таблицы с количественными показателями предобработки

import requests
import pandas as pd
from datetime import datetime, timedelta

# --- Конфигурация --- #
WB_API_TOKEN = ""  # <-- ВСТАВЬ СВОЙ ТОКЕН
WB_API_HOST = "https://statistics-api.wildberries.ru"
DATE_FROM = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

ENDPOINTS = {
    "orders": "/api/v1/supplier/orders",
    "sales": "/api/v1/supplier/sales",
    "stocks": "/api/v1/supplier/stocks",
}

REQUIRED_FIELDS = {
    "orders": ["srid", "date", "nmId", "supplierArticle", "totalPrice", "warehouseName"],
    "sales": ["saleID", "date", "nmId", "supplierArticle", "totalPrice", "forPay", "finishedPrice"],
    "stocks": ["nmId", "supplierArticle", "warehouseName", "quantity", "quantityFull"],
}

DEDUP_KEYS = {
    "orders": "srid",
    "sales": "saleID",
    "stocks": None,
}

NUMERIC_FIELDS = {
    "orders": ["totalPrice", "discountPercent"],
    "sales": ["totalPrice", "forPay", "finishedPrice", "priceWithDisc", "spp"],
    "stocks": ["quantity", "quantityFull", "inWayToClient", "inWayFromClient", "Price"],
}


# --- Этап 1: Получение данных из WB API --- #
print("=" * 70)
print("ЭТАП 1: ПОЛУЧЕНИЕ ДАННЫХ ИЗ WB API")
print(f"Период: с {DATE_FROM}")
print("=" * 70)

raw_data = {}
for name, path in ENDPOINTS.items():
    url = f"{WB_API_HOST}{path}"
    params = {"dateFrom": DATE_FROM}
    headers = {"Authorization": WB_API_TOKEN}

    response = requests.get(url, params=params, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    raw_data[name] = data
    print(f"  {name:>10}: получено {len(data):>6} записей")


# --- Этап 2: Проверка структуры --- #
print("\n" + "=" * 70)
print("ЭТАП 2: ПРОВЕРКА СТРУКТУРЫ (наличие обязательных полей)")
print("=" * 70)

valid_data = {}
structure_stats = {}

for name, records in raw_data.items():
    required = REQUIRED_FIELDS[name]
    valid = []
    invalid_count = 0

    for rec in records:
        if all(rec.get(f) is not None for f in required):
            valid.append(rec)
        else:
            invalid_count += 1

    valid_data[name] = valid
    structure_stats[name] = {
        "Получено из API": len(records),
        "Прошло проверку структуры": len(valid),
        "Исключено (нет обязательных полей)": invalid_count,
    }
    print(f"  {name:>10}: {len(valid):>6} валидных, {invalid_count:>4} исключено")


# --- Этап 3: Приведение типов и форматов --- #
print("\n" + "=" * 70)
print("ЭТАП 3: ПРИВЕДЕНИЕ ТИПОВ И ФОРМАТОВ")
print("=" * 70)

dataframes = {}
type_issues = {}

for name, records in valid_data.items():
    df = pd.DataFrame(records)
    issues = 0

    for col in NUMERIC_FIELDS.get(name, []):
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            issues += (after_na - before_na)

    for col in df.columns:
        if "date" in col.lower() or col in ("date", "lastChangeDate", "cancel_dt"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    dataframes[name] = df
    type_issues[name] = issues
    print(f"  {name:>10}: {len(df)} записей, {issues} значений не прошли преобразование типов")


# --- Этап 4: Дедупликация --- #
print("\n" + "=" * 70)
print("ЭТАП 4: ДЕДУПЛИКАЦИЯ")
print("=" * 70)

dedup_stats = {}

for name, df in dataframes.items():
    key = DEDUP_KEYS[name]
    before = len(df)

    if key and key in df.columns:
        df = df.sort_values("lastChangeDate", ascending=False) if "lastChangeDate" in df.columns else df
        df = df.drop_duplicates(subset=[key], keep="first")

    after = len(df)
    duplicates = before - after
    dataframes[name] = df
    dedup_stats[name] = {
        "До дедупликации": before,
        "Дубликатов удалено": duplicates,
        "После дедупликации": after,
    }
    print(f"  {name:>10}: {before:>6} -> {after:>6} (удалено дубликатов: {duplicates})")


# --- Этап 5: Проверка допустимости числовых значений --- #
print("\n" + "=" * 70)
print("ЭТАП 5: ПРОВЕРКА ДОПУСТИМОСТИ ЧИСЛОВЫХ ЗНАЧЕНИЙ")
print("=" * 70)

range_stats = {}

for name, df in dataframes.items():
    before = len(df)
    removed = 0

    for col in NUMERIC_FIELDS.get(name, []):
        if col in df.columns:
            mask = df[col].fillna(0) < 0
            neg_count = mask.sum()
            if neg_count > 0:
                df = df[~mask]
                removed += neg_count

    dataframes[name] = df
    range_stats[name] = {
        "Записей до проверки": before,
        "Удалено (отрицательные значения)": removed,
        "Записей после проверки": len(df),
    }
    print(f"  {name:>10}: удалено {removed} записей с отрицательными значениями")


# --- Итоговая сводная таблица --- #
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ПРЕДОБРАБОТКИ")
print("=" * 70)

summary = pd.DataFrame({
    "Показатель": [
        "Получено записей из API",
        "Исключено при проверке структуры",
        "Ошибки преобразования типов",
        "Дубликатов удалено",
        "Удалено (недопустимые значения)",
        "Итого записей в STG",
    ],
    "Заказы (orders)": [
        structure_stats["orders"]["Получено из API"],
        structure_stats["orders"]["Исключено (нет обязательных полей)"],
        type_issues["orders"],
        dedup_stats["orders"]["Дубликатов удалено"],
        range_stats["orders"]["Удалено (отрицательные значения)"],
        len(dataframes["orders"]),
    ],
    "Продажи (sales)": [
        structure_stats["sales"]["Получено из API"],
        structure_stats["sales"]["Исключено (нет обязательных полей)"],
        type_issues["sales"],
        dedup_stats["sales"]["Дубликатов удалено"],
        range_stats["sales"]["Удалено (отрицательные значения)"],
        len(dataframes["sales"]),
    ],
    "Остатки (stocks)": [
        structure_stats["stocks"]["Получено из API"],
        structure_stats["stocks"]["Исключено (нет обязательных полей)"],
        type_issues["stocks"],
        dedup_stats["stocks"]["Дубликатов удалено"],
        range_stats["stocks"]["Удалено (отрицательные значения)"],
        len(dataframes["stocks"]),
    ],
})

print(summary.to_string(index=False))
print()


# --- Пример данных после предобработки --- #
print("=" * 70)
print("ПРИМЕР ДАННЫХ ПОСЛЕ ПРЕДОБРАБОТКИ (первые 5 записей)")
print("=" * 70)

for name, df in dataframes.items():
    print(f"\n--- {name.upper()} ---")
    display_cols = REQUIRED_FIELDS[name][:5]
    cols_to_show = [c for c in display_cols if c in df.columns]
    print(df[cols_to_show].head().to_string(index=False))
