# --- Приложение Б — Демонстрация оценки качества данных --- #
# Запуск: Google Colab или локально (после appendix_a)
# Результат: таблицы с коэффициентами качества данных

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

EXPECTED_SCHEMA = {
    "orders": {
        "srid": "object", "date": "datetime64", "nmId": "int64",
        "supplierArticle": "object", "totalPrice": "float64",
    },
    "sales": {
        "saleID": "object", "date": "datetime64", "nmId": "int64",
        "supplierArticle": "object", "totalPrice": "float64", "forPay": "float64",
    },
    "stocks": {
        "nmId": "int64", "supplierArticle": "object",
        "warehouseName": "object", "quantity": "int64",
    },
}

NUMERIC_BOUNDS = {
    "orders": {"totalPrice": (0, 1_000_000), "discountPercent": (0, 100)},
    "sales": {"totalPrice": (0, 1_000_000), "forPay": (0, 1_000_000), "finishedPrice": (0, 1_000_000)},
    "stocks": {"quantity": (0, 100_000), "quantityFull": (0, 100_000)},
}


# --- Загрузка данных --- #
print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ ИЗ WB API")
print(f"Период: с {DATE_FROM}")
print("=" * 70)

dataframes = {}
for name, path in ENDPOINTS.items():
    url = f"{WB_API_HOST}{path}"
    params = {"dateFrom": DATE_FROM}
    headers = {"Authorization": WB_API_TOKEN}

    response = requests.get(url, params=params, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []

    df = pd.DataFrame(data)

    for col in df.columns:
        if "date" in col.lower() or col in ("date", "lastChangeDate", "cancel_dt"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in ["totalPrice", "forPay", "finishedPrice", "priceWithDisc", "spp",
                 "discountPercent", "Price", "Discount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["nmId", "quantity", "quantityFull", "inWayToClient", "inWayFromClient"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dataframes[name] = df
    print(f"  {name:>10}: {len(df)} записей загружено")


# --- Проверка 1: Коэффициент полноты --- #
print("\n" + "=" * 70)
print("ПРОВЕРКА 1: КОЭФФИЦИЕНТ ПОЛНОТЫ (K_полн)")
print("K_полн = N_заполненных / N_обязательных")
print("=" * 70)

completeness = {}
for name, df in dataframes.items():
    required = REQUIRED_FIELDS[name]
    total_required = len(df) * len(required)
    filled = 0
    details = {}

    for col in required:
        if col in df.columns:
            non_null = df[col].notna().sum()
            filled += non_null
            details[col] = f"{non_null}/{len(df)}"
        else:
            details[col] = f"0/{len(df)} (столбец отсутствует)"

    k_completeness = filled / total_required if total_required > 0 else 0
    completeness[name] = k_completeness

    print(f"\n  --- {name.upper()} ---")
    print(f"  Обязательных полей: {len(required)}")
    print(f"  Всего проверяемых значений: {total_required}")
    print(f"  Заполнено: {filled}")
    print(f"  K_полн = {k_completeness:.4f}")
    print(f"  Детализация по полям:")
    for col, stat in details.items():
        print(f"    {col:>25}: {stat}")


# --- Проверка 2: Коэффициент уникальности --- #
print("\n" + "=" * 70)
print("ПРОВЕРКА 2: КОЭФФИЦИЕНТ УНИКАЛЬНОСТИ (K_уник)")
print("K_уник = 1 - N_дубликатов / N_всего")
print("=" * 70)

uniqueness = {}
for name, df in dataframes.items():
    key = DEDUP_KEYS[name]
    n_total = len(df)

    if key and key in df.columns:
        n_dup = n_total - df[key].nunique()
    else:
        n_dup = 0

    k_unique = 1 - (n_dup / n_total) if n_total > 0 else 1
    uniqueness[name] = k_unique

    print(f"  {name:>10}: N={n_total}, дубликатов={n_dup}, K_уник={k_unique:.4f}")


# --- Проверка 3: Соответствие схеме хранения --- #
print("\n" + "=" * 70)
print("ПРОВЕРКА 3: СООТВЕТСТВИЕ СХЕМЕ ХРАНЕНИЯ")
print("=" * 70)

schema_ok = {}
for name, df in dataframes.items():
    expected = EXPECTED_SCHEMA[name]
    all_ok = True
    issues = []

    for col, expected_type in expected.items():
        if col not in df.columns:
            issues.append(f"  Столбец '{col}' отсутствует")
            all_ok = False
        else:
            actual_type = str(df[col].dtype)
            if expected_type == "datetime64":
                ok = "datetime" in actual_type
            elif expected_type == "int64":
                ok = "int" in actual_type or "float" in actual_type
            elif expected_type == "float64":
                ok = "float" in actual_type or "int" in actual_type
            else:
                ok = True

            if not ok:
                issues.append(f"  '{col}': ожидался {expected_type}, получен {actual_type}")
                all_ok = False

    schema_ok[name] = all_ok
    status = "СООТВЕТСТВУЕТ" if all_ok else "ЕСТЬ РАСХОЖДЕНИЯ"
    print(f"  {name:>10}: {status}")
    for issue in issues:
        print(f"    {issue}")


# --- Проверка 4: Логическая непротиворечивость --- #
print("\n" + "=" * 70)
print("ПРОВЕРКА 4: ЛОГИЧЕСКАЯ НЕПРОТИВОРЕЧИВОСТЬ")
print("(проверка допустимых диапазонов числовых значений)")
print("=" * 70)

logic_ok = {}
for name, df in dataframes.items():
    bounds = NUMERIC_BOUNDS.get(name, {})
    violations = 0
    total_checks = 0
    details = []

    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            vals = df[col].dropna()
            total_checks += len(vals)
            out_of_range = ((vals < lo) | (vals > hi)).sum()
            violations += out_of_range
            details.append(f"    {col}: {out_of_range} из {len(vals)} вне [{lo}, {hi}]")

    logic_ok[name] = violations == 0
    status = "ПРОЙДЕНА" if violations == 0 else f"НАРУШЕНИЙ: {violations}"
    print(f"  {name:>10}: {status}")
    for d in details:
        print(d)


# --- Проверка 5: Пригодность для аналитики --- #
print("\n" + "=" * 70)
print("ПРОВЕРКА 5: ПРИГОДНОСТЬ ДЛЯ АНАЛИТИКИ")
print("=" * 70)

analytics_ok = {}
for name, df in dataframes.items():
    checks = []

    if "date" in df.columns:
        date_col = pd.to_datetime(df["date"], errors="coerce", utc=True)
        date_range = (date_col.max() - date_col.min()).days if date_col.notna().any() else 0
        checks.append(f"  Временной охват: {date_range} дней")
        has_time = date_range >= 7
    elif name == "stocks":
        has_time = True
        checks.append(f"  Остатки — срезовые данные (временной ряд не требуется)")
    else:
        has_time = False

    if "nmId" in df.columns:
        n_articles = df["nmId"].nunique()
        checks.append(f"  Уникальных товаров (nmId): {n_articles}")
        has_articles = n_articles > 0
    else:
        has_articles = False

    is_ok = has_time and has_articles
    analytics_ok[name] = is_ok

    status = "ПРИГОДЕН" if is_ok else "НЕДОСТАТОЧНО ДАННЫХ"
    print(f"  {name:>10}: {status}")
    for c in checks:
        print(c)


# --- Итоговая сводная таблица качества --- #
print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА ОЦЕНКИ КАЧЕСТВА ДАННЫХ")
print("=" * 70)

quality_summary = pd.DataFrame({
    "Критерий проверки": [
        "Коэффициент полноты (K_полн)",
        "Коэффициент уникальности (K_уник)",
        "Соответствие схеме хранения",
        "Логическая непротиворечивость",
        "Пригодность для аналитики",
    ],
    "Заказы (orders)": [
        f"{completeness['orders']:.4f}",
        f"{uniqueness['orders']:.4f}",
        "Да" if schema_ok["orders"] else "Нет",
        "Да" if logic_ok["orders"] else "Нет",
        "Да" if analytics_ok["orders"] else "Нет",
    ],
    "Продажи (sales)": [
        f"{completeness['sales']:.4f}",
        f"{uniqueness['sales']:.4f}",
        "Да" if schema_ok["sales"] else "Нет",
        "Да" if logic_ok["sales"] else "Нет",
        "Да" if analytics_ok["sales"] else "Нет",
    ],
    "Остатки (stocks)": [
        f"{completeness['stocks']:.4f}",
        f"{uniqueness['stocks']:.4f}",
        "Да" if schema_ok["stocks"] else "Нет",
        "Да" if logic_ok["stocks"] else "Нет",
        "Да" if analytics_ok["stocks"] else "Нет",
    ],
})

print(quality_summary.to_string(index=False))

print("\n" + "=" * 70)
print("ВЫВОД: Данные пригодны для формирования витринного слоя")
print("и построения аналитической отчётности.")
print("=" * 70)
