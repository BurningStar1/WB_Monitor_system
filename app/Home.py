"""Home — главная страница с обзором отчётов и управлением данными."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st
from datetime import date, timedelta
from pathlib import Path

from styles import inject_global_styles
from auth import check_auth, logout
from config import get_settings
from db import get_engine
from sqlalchemy import text

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

st.set_page_config(page_title="WB Analytics", layout="wide", page_icon="\U0001f4ca")
inject_global_styles()

if not check_auth():
    st.stop()
logout()

# ── Helpers ──────────────────────────────────────────────────

TOKEN_FILE = PROJECT_ROOT / "wb_api_key.txt"


def _read_token() -> str:
    """Read current API token (masked for display)."""
    settings = get_settings()
    return settings.wb_api_token


def _save_token(token: str) -> None:
    """Save API token to file."""
    TOKEN_FILE.write_text(token.strip(), encoding="utf-8")


def _get_data_status() -> list[dict]:
    """Get date ranges and row counts for all data layers."""
    engine = get_engine()
    queries = [
        ("Заказы", "mart.orders_daily", "order_date"),
        ("Продажи", "mart.sales_daily", "sales_date"),
        ("Финансы", "mart.finance_daily", "report_date"),
        ("Реклама", "mart.ads_daily", "ads_date"),
        ("Остатки", "mart.stocks_snapshot", "snapshot_date"),
    ]
    results = []
    with engine.connect() as conn:
        for label, table, col in queries:
            try:
                row = conn.execute(text(
                    f"SELECT MIN({col})::date, MAX({col})::date, COUNT(*) FROM {table}"
                )).fetchone()
                results.append({
                    "name": label,
                    "date_from": row[0],
                    "date_to": row[1],
                    "rows": row[2],
                })
            except Exception:
                results.append({"name": label, "date_from": None, "date_to": None, "rows": 0})

        # Last raw payload loaded_at
        try:
            last = conn.execute(text(
                "SELECT MAX(loaded_at) FROM raw.wb_api_payloads"
            )).scalar()
            results.append({"_last_loaded": last})
        except Exception:
            results.append({"_last_loaded": None})

    return results


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 12:
        return token[:4] + "..." + token[-4:]
    return token[:8] + "..." + token[-4:]


def _run_pipeline(days_back: int, skip_ads: bool = False):
    """Run ETL pipeline with Streamlit status updates."""
    from datetime import date, timedelta
    from api import WB_ENDPOINTS
    from etl.raw_loader import RawLoader
    from etl.stg_loader import StgLoader
    from etl.mart_loader import MartLoader

    date_from = date.today() - timedelta(days=days_back)
    date_to = date.today()

    status = st.status(f"Сбор данных за {days_back} дн. ({date_from} \u2192 {date_to})", expanded=True)

    try:
        raw = RawLoader()
        stg = StgLoader("INFO")
        mart = MartLoader("INFO")

        # Step 1: Orders
        status.update(label="Загрузка заказов...", state="running")
        status.write(f"\u23f3 Заказы с {date_from}...")
        payload = raw.load_endpoint("orders", date_from)
        status.write(f"\u2705 Заказы: **{len(payload)}** записей")

        # Step 2: Sales
        status.update(label="Загрузка продаж...", state="running")
        status.write(f"\u23f3 Продажи с {date_from}...")
        payload = raw.load_endpoint("sales", date_from)
        status.write(f"\u2705 Продажи: **{len(payload)}** записей")

        # Step 3: Stocks
        status.update(label="Загрузка остатков...", state="running")
        status.write(f"\u23f3 Остатки...")
        payload = raw.load_endpoint("stocks", date_from)
        status.write(f"\u2705 Остатки: **{len(payload)}** записей")

        # Step 4: Finance
        status.update(label="Загрузка финансов...", state="running")
        status.write(f"\u23f3 Финансовый отчёт {date_from} \u2192 {date_to}...")
        payload = raw.load_finance(date_from, date_to)
        status.write(f"\u2705 Финансы: **{len(payload)}** записей")

        # Step 5: Ads (optional, slow)
        if not skip_ads:
            status.update(label="Загрузка рекламы (может занять несколько минут)...", state="running")
            status.write(f"\u23f3 Рекламные кампании {date_from} \u2192 {date_to}...")
            try:
                payload = raw.load_ads(date_from, date_to)
                status.write(f"\u2705 Реклама: **{len(payload)}** записей")
            except Exception as e:
                status.write(f"\u26a0\ufe0f Реклама пропущена: {e}")
        else:
            status.write("\u23ed\ufe0f Реклама пропущена (отключена)")

        # Step 6: STG
        status.update(label="Трансформация (STG)...", state="running")
        status.write("\u23f3 Парсинг и нормализация данных...")
        stg.run()
        status.write("\u2705 STG таблицы обновлены")

        # Step 7: MART
        status.update(label="Агрегация витрин (MART)...", state="running")
        status.write("\u23f3 Расчёт витрин данных...")
        mart.run()
        status.write("\u2705 Витрины обновлены")

        status.update(label="Сбор данных завершён!", state="complete")
        return True

    except Exception as e:
        status.update(label=f"Ошибка: {e}", state="error")
        status.write(f"\u274c {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# PAGE CONTENT
# ═══════════════════════════════════════════════════════════════

st.title("Аналитический сервис Wildberries")

# ── Freshness badge ──────────────────────────────────────────
def _freshness_html() -> str:
    """Render a compact freshness badge next to the title."""
    try:
        eng = get_engine()
        with eng.connect() as conn:
            last = conn.execute(text("SELECT MAX(loaded_at) FROM raw.wb_api_payloads")).scalar()
    except Exception:
        return ""
    if not last:
        return (
            '<div style="display:inline-block;margin:-0.5rem 0 1rem 0;'
            'padding:4px 10px;border-radius:999px;background:#fef3c7;color:#92400e;'
            'font-size:12px;font-weight:500">⚠ Данные ещё не загружались</div>'
        )
    from datetime import datetime, timezone
    now = datetime.now(tz=last.tzinfo) if last.tzinfo else datetime.now()
    delta = now - last
    mins = int(delta.total_seconds() / 60)
    if mins < 60:
        label = f"{mins} мин назад"
        color = "#065f46"; bg = "#d1fae5"
    elif mins < 60 * 24:
        label = f"{mins // 60} ч назад"
        color = "#065f46"; bg = "#d1fae5"
    elif mins < 60 * 24 * 3:
        label = f"{mins // (60 * 24)} дн назад"
        color = "#92400e"; bg = "#fef3c7"
    else:
        label = f"{mins // (60 * 24)} дн назад"
        color = "#991b1b"; bg = "#fee2e2"
    return (
        f'<div style="display:inline-block;margin:-0.5rem 0 1rem 0;'
        f'padding:4px 10px;border-radius:999px;background:{bg};color:{color};'
        f'font-size:12px;font-weight:500">🔄 Данные: {label}</div>'
    )


@st.cache_data(ttl=600, show_spinner=False)
def _alert_counts() -> dict:
    """Return critical-event counts for the badge on the home page."""
    out = {"low_stock": 0, "losses": 0, "wow_drop": 0}
    try:
        eng = get_engine()
        with eng.connect() as conn:
            # Low stock: last snapshot date, qty_full < 30 AND positive orders last 30d
            row = conn.execute(text(
                """
                WITH recent_orders AS (
                    SELECT nm_id, SUM(orders_count) AS o
                    FROM mart.orders_daily
                    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY nm_id
                ),
                latest_stock AS (
                    SELECT nm_id, SUM(quantity_full) AS q
                    FROM mart.stocks_snapshot
                    WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM mart.stocks_snapshot)
                    GROUP BY nm_id
                )
                SELECT COUNT(*)
                FROM latest_stock ls
                JOIN recent_orders ro USING (nm_id)
                WHERE ro.o > 0 AND ls.q / GREATEST(ro.o / 30.0, 0.01) < 7
                """
            )).scalar()
            out["low_stock"] = int(row or 0)

            # Articles with negative profit in last 30 days (finance_daily)
            row = conn.execute(text(
                """
                SELECT COUNT(*) FROM (
                    SELECT nm_id, SUM(ppvz_for_pay - logistics_amount - storage_amount
                          - penalty_amount - acceptance_amount - acquiring_amount
                          - deduction_amount + additional_payment_amount
                          - COALESCE(cost_amount, 0) - COALESCE(tax_amount, 0)) AS p
                    FROM mart.finance_daily
                    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY nm_id
                    HAVING SUM(ppvz_for_pay - logistics_amount - storage_amount
                          - penalty_amount - acceptance_amount - acquiring_amount
                          - deduction_amount + additional_payment_amount
                          - COALESCE(cost_amount, 0) - COALESCE(tax_amount, 0)) < 0
                ) x
                """
            )).scalar()
            out["losses"] = int(row or 0)

            # WoW drop ≥ 30% (current 7 vs previous 7)
            row = conn.execute(text(
                """
                WITH cur AS (
                    SELECT nm_id, SUM(orders_count) AS o_cur FROM mart.orders_daily
                    WHERE order_date >= CURRENT_DATE - 6 GROUP BY nm_id
                ),
                prev AS (
                    SELECT nm_id, SUM(orders_count) AS o_prev FROM mart.orders_daily
                    WHERE order_date BETWEEN CURRENT_DATE - 13 AND CURRENT_DATE - 7
                    GROUP BY nm_id
                )
                SELECT COUNT(*)
                FROM prev p
                LEFT JOIN cur c USING (nm_id)
                WHERE p.o_prev >= 3
                  AND COALESCE(c.o_cur, 0) < p.o_prev * 0.7
                """
            )).scalar()
            out["wow_drop"] = int(row or 0)
    except Exception:
        pass
    return out


def _alerts_html(counts: dict) -> str:
    total = counts["low_stock"] + counts["losses"] + counts["wow_drop"]
    if total == 0:
        return (
            '<div style="display:inline-block;margin:-0.5rem 0 1rem 0.5rem;'
            'padding:4px 10px;border-radius:999px;background:#d1fae5;color:#065f46;'
            'font-size:12px;font-weight:500">✓ Нет алертов</div>'
        )
    tooltip = (
        f"Низкий запас: {counts['low_stock']}  ·  "
        f"Убыточных: {counts['losses']}  ·  "
        f"Падение WoW: {counts['wow_drop']}"
    )
    return (
        f'<div title="{tooltip}" style="display:inline-block;margin:-0.5rem 0 1rem 0.5rem;'
        f'padding:4px 10px;border-radius:999px;background:#fee2e2;color:#991b1b;'
        f'font-size:12px;font-weight:500">🚨 Алертов: {total}</div>'
    )


_badge = _freshness_html()
_alerts_badge = _alerts_html(_alert_counts())
if _badge or _alerts_badge:
    st.markdown(_badge + _alerts_badge, unsafe_allow_html=True)

tab_overview, tab_data = st.tabs(["\U0001f4ca Обзор", "\u2699\ufe0f Данные и API"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: Обзор
# ═══════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("""
    Выберите отчёт в боковом меню для просмотра данных.
    """)

    report_groups = [
        ("📊 Обзор и сводки", [
            ("\U0001f4c8", "KPI-дашборд", "Сводные показатели за период"),
            ("\U0001f4c5", "Еженедельный отчёт", "Динамика по неделям"),
            ("\U0001f4cb", "Отчёт за период", "Помесячная сводка"),
            ("\U0001f504", "Неделя к неделе", "Сравнение двух недель"),
            ("\U0001fac0", "Рука на Пульсе", "Ежедневная динамика по артикулам"),
        ]),
        ("💰 Финансы и прибыль", [
            ("\U0001f4ca", "ОПИУ", "Отчёт о прибылях и убытках"),
            ("\U0001f4b0", "Рентабельность", "Финансовый результат по товарам"),
            ("\U0001f9ee", "Юнит-экономика", "CM1 / CM2 / CM3 на один юнит"),
        ]),
        ("📦 Ассортимент и логистика", [
            ("\U0001f4e6", "Отчёт по артикулам", "Детализация до товара"),
            ("\U0001f3ed", "Остатки на складах", "Распределение и капитализация"),
            ("\U0001f524", "ABC-анализ", "Мульти-метрика: выручка, прибыль, продажи"),
            ("\U0001f465", "Когортный анализ", "Возраст артикула и retention"),
            ("\U0001f4e6", "Потребность", "Расчёт потребности в поставках"),
        ]),
        ("📢 Маркетинг и планирование", [
            ("\U0001f4c8", "Прогноз", "Прогноз заказов и прибыли"),
            ("\U0001f4e2", "Конверсия рекламы", "Аналитика рекламных кампаний"),
            ("\U0001f3f7\ufe0f", "Калькулятор акций", "Оценка участия в промо"),
        ]),
        ("⚙️ Служебное", [
            ("\U0001f6a8", "Алерты", "Низкие остатки, убытки, падение спроса"),
            ("\U0001f9ea", "Качество данных", "Проверки полноты и целостности витрин"),
            ("\U0001f4da", "Справочники", "Себестоимость, затраты, налоги"),
        ]),
    ]

    for group_title, reports in report_groups:
        st.markdown(f"#### {group_title}")
        cols = st.columns(3)
        for i, (icon, name, desc) in enumerate(reports):
            with cols[i % 3]:
                st.markdown(
                    f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;'
                    f'padding:12px 16px;margin-bottom:8px">'
                    f'<span style="font-size:18px">{icon}</span> '
                    f'<strong>{name}</strong><br>'
                    f'<span style="color:#64748b;font-size:13px">{desc}</span></div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════
# TAB 2: Данные и API
# ═══════════════════════════════════════════════════════════════

with tab_data:

    # ── API Key Section ──────────────────────────────────────
    st.markdown("### \U0001f511 API-ключ Wildberries")

    current_token = _read_token()
    has_token = bool(current_token)

    if has_token:
        st.success("\u2705 API-ключ настроен и сохранён")
        st.caption("Ключ хранится локально и не отображается в целях безопасности.")
        if st.button("\U0001f5d1\ufe0f Удалить ключ", key="delete_token"):
            TOKEN_FILE.unlink(missing_ok=True)
            st.rerun()
    else:
        st.warning("API-ключ не настроен. Укажите его ниже для сбора данных.")
        st.caption(
            "Получите ключ в личном кабинете WB: "
            "Настройки \u2192 Доступ к API \u2192 Статистика. "
            "Ключ сохраняется локально и не передаётся третьим лицам."
        )
        new_token = st.text_input(
            "WB API Token",
            type="password",
            placeholder="eyJhbGciOiJFUzI1NiIs...",
            key="api_token_input",
        )
        if st.button("Сохранить ключ", key="save_token", type="primary"):
            if new_token and len(new_token) > 20:
                _save_token(new_token)
                st.success("\u2705 Ключ сохранён!")
                st.rerun()
            else:
                st.error("Введите корректный API-ключ (длина > 20 символов)")

    st.divider()

    # ── Data Status Section ──────────────────────────────────
    st.markdown("### \U0001f4c1 Состояние данных")

    data_status = _get_data_status()
    last_loaded = None
    table_rows = []
    for item in data_status:
        if "_last_loaded" in item:
            last_loaded = item["_last_loaded"]
            continue
        table_rows.append(item)

    if last_loaded:
        st.caption(f"Последнее обновление: **{last_loaded.strftime('%d.%m.%Y %H:%M')}**")
    else:
        st.caption("Данные ещё не загружались")

    # Status cards
    cols = st.columns(len(table_rows))
    for i, row in enumerate(table_rows):
        with cols[i]:
            if row["date_from"] and row["date_to"]:
                days = (row["date_to"] - row["date_from"]).days
                st.metric(
                    row["name"],
                    f'{row["rows"]:,}'.replace(",", " "),
                    f'{row["date_from"].strftime("%d.%m.%y")} \u2014 {row["date_to"].strftime("%d.%m.%y")}',
                )
            else:
                st.metric(row["name"], "0", "Нет данных")

    st.divider()

    # ── Data Collection Section ──────────────────────────────
    st.markdown("### \U0001f504 Сбор данных")

    if not has_token:
        st.info("Сначала укажите API-ключ выше, затем запустите сбор данных.")
    else:
        st.caption(
            "Новые данные **дополняют** существующие \u2014 старые не удаляются. "
            "Заказы и продажи: API отдаёт до ~6 мес. назад. "
            "Финансы и реклама: до 2 лет."
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            period_opts = {
                "Последние 7 дней": 7,
                "Последние 30 дней": 30,
                "Последние 90 дней": 90,
                "Последние 180 дней": 180,
                "Максимум (2 года)": 730,
            }
            period_label = st.selectbox(
                "Период загрузки",
                list(period_opts.keys()),
                index=0,
                key="collect_period",
            )
            days_back = period_opts[period_label]
        with c2:
            skip_ads = st.checkbox("Пропустить рекламу", value=False, key="skip_ads",
                                   help="Реклама загружается медленно (rate limit WB). Пропустите для быстрого обновления.")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button(
                "\u25b6\ufe0f Обновить данные",
                type="primary",
                width="stretch",
                key="run_pipeline",
            )

        if run_btn:
            success = _run_pipeline(days_back, skip_ads=skip_ads)
            if success:
                st.balloons()
                st.rerun()
