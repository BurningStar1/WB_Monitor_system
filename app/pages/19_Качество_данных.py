"""Качество данных — проверки полноты и целостности витрин."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from db import get_engine
from sqlalchemy import text
from styles import (
    inject_global_styles, fmt_number,
    render_sortable_table, export_buttons,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()

st.title("🧪 Качество данных")
st.caption("Диагностика витрин: заполненность, пропуски, целостность справочников.")


# ── Overview: rows + date range per mart ─────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _mart_overview() -> pd.DataFrame:
    eng = get_engine()
    tables = [
        ("mart.orders_daily", "order_date", "Заказы"),
        ("mart.sales_daily", "sales_date", "Продажи"),
        ("mart.finance_daily", "report_date", "Финансы"),
        ("mart.ads_daily", "ads_date", "Реклама"),
        ("mart.stocks_snapshot", "snapshot_date", "Остатки"),
    ]
    rows = []
    with eng.connect() as c:
        for tbl, dcol, lbl in tables:
            try:
                row = c.execute(text(
                    f"SELECT MIN({dcol}), MAX({dcol}), COUNT(*), COUNT(DISTINCT {dcol})"
                    f" FROM {tbl}"
                )).fetchone()
                rows.append({
                    "Витрина": lbl,
                    "Таблица": tbl,
                    "Записей": int(row[2] or 0),
                    "Дат": int(row[3] or 0),
                    "С": row[0],
                    "По": row[1],
                })
            except Exception as e:
                rows.append({
                    "Витрина": lbl, "Таблица": tbl, "Записей": 0,
                    "Дат": 0, "С": None, "По": None,
                    "Ошибка": str(e)[:60],
                })
    return pd.DataFrame(rows)


ov = _mart_overview()
st.markdown("### Обзор витрин")
hdr = "<tr><th>Витрина</th><th>Таблица</th><th>Записей</th><th>Дат</th><th>С</th><th>По</th></tr>"
rows_html = ""
for _, r in ov.iterrows():
    rows_html += (
        f'<tr><td><b>{r["Витрина"]}</b></td>'
        f'<td style="color:#64748b;font-family:monospace;font-size:12px">{r["Таблица"]}</td>'
        f'<td class="num">{fmt_number(r["Записей"])}</td>'
        f'<td class="num">{fmt_number(r["Дат"])}</td>'
        f'<td class="ctr">{r["С"] or "—"}</td>'
        f'<td class="ctr">{r["По"] or "—"}</td></tr>'
    )
render_sortable_table("qov", hdr, rows_html, height=280)


# ── Missing dates per mart ───────────────────────────────────
st.markdown("### Пропуски в днях")
st.caption("Дни без данных в пределах min/max диапазона каждой витрины.")


@st.cache_data(ttl=600, show_spinner=False)
def _missing_days() -> pd.DataFrame:
    eng = get_engine()
    tables = [
        ("mart.orders_daily", "order_date", "Заказы"),
        ("mart.sales_daily", "sales_date", "Продажи"),
        ("mart.finance_daily", "report_date", "Финансы"),
        ("mart.stocks_snapshot", "snapshot_date", "Остатки"),
    ]
    rows = []
    with eng.connect() as c:
        for tbl, dcol, lbl in tables:
            try:
                row = c.execute(text(
                    f"SELECT MIN({dcol}), MAX({dcol})"
                    f" FROM {tbl}"
                )).fetchone()
                if not row or not row[0] or not row[1]:
                    continue
                d_from, d_to = row[0], row[1]
                all_days = (d_to - d_from).days + 1
                present = c.execute(text(
                    f"SELECT COUNT(DISTINCT {dcol}) FROM {tbl}"
                    f" WHERE {dcol} BETWEEN :a AND :b"
                ), {"a": d_from, "b": d_to}).scalar()
                missing = all_days - int(present or 0)
                pct = 100 * (int(present or 0)) / all_days if all_days else 0
                rows.append({
                    "Витрина": lbl,
                    "Период": f"{d_from} — {d_to}",
                    "Дней в периоде": all_days,
                    "С данными": int(present or 0),
                    "Пропущено": missing,
                    "% заполненности": round(pct, 1),
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


md = _missing_days()
if md.empty:
    st.info("Недостаточно данных для анализа пропусков")
else:
    hdr = (
        "<tr><th>Витрина</th><th>Период</th><th>Дней в периоде</th>"
        "<th>С данными</th><th>Пропущено</th><th>% заполн.</th></tr>"
    )
    rows_html = ""
    for _, r in md.iterrows():
        pct = r["% заполненности"]
        cls = "pos" if pct >= 95 else ("neg" if pct < 80 else "")
        rows_html += (
            f'<tr><td><b>{r["Витрина"]}</b></td>'
            f'<td style="color:#64748b">{r["Период"]}</td>'
            f'<td class="num">{r["Дней в периоде"]}</td>'
            f'<td class="num">{r["С данными"]}</td>'
            f'<td class="num {"neg" if r["Пропущено"] > 0 else ""}">{r["Пропущено"]}</td>'
            f'<td class="ctr {cls}">{pct:.1f}%</td></tr>'
        )
    render_sortable_table("qmd", hdr, rows_html, height=280)


# ── Articles without cost reference ─────────────────────────
st.markdown("### Артикулы без себестоимости")
st.caption(
    "Активные артикулы (с продажами за последние 90 дней), для которых нет записи "
    "в ``dict.cost_reference``. Для таких товаров прибыль считается как 0 — это искажает отчёты."
)


@st.cache_data(ttl=600, show_spinner=False)
def _articles_no_cost(days: int = 90) -> pd.DataFrame:
    """Return active articles missing cost reference."""
    eng = get_engine()
    q = text("""
        SELECT
            s.nm_id,
            MAX(s.supplier_article) AS supplier_article,
            MAX(s.subject) AS subject,
            MAX(s.brand) AS brand,
            SUM(s.sales_count) AS units_sold,
            SUM(s.net_revenue) AS revenue
        FROM mart.sales_daily s
        LEFT JOIN dict.cost_reference cr ON cr.nm_id = s.nm_id
        WHERE s.sales_date >= CURRENT_DATE - CAST(:d AS INTEGER) * INTERVAL '1 day'
          AND cr.nm_id IS NULL
          AND s.sales_count > 0
        GROUP BY s.nm_id
        ORDER BY SUM(s.net_revenue) DESC NULLS LAST
        LIMIT 500
    """)
    try:
        with eng.connect() as c:
            res = c.execute(q, {"d": days}).fetchall()
            cols = ["nm_id", "supplier_article", "subject", "brand", "units_sold", "revenue"]
            return pd.DataFrame(res, columns=cols)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)[:200]]})


lookback = st.slider("Окно (дни)", min_value=30, max_value=180, value=90, step=30, key="nc_lb")
nc = _articles_no_cost(lookback)

if nc.empty or "error" in nc.columns:
    if "error" in nc.columns:
        st.warning(f"Ошибка запроса: {nc.iloc[0, 0]}")
    else:
        st.success("Все активные артикулы имеют себестоимость — отлично!")
else:
    c1, c2, c3 = st.columns(3)
    total_units = int(pd.to_numeric(nc["units_sold"], errors="coerce").fillna(0).sum())
    total_rev = float(pd.to_numeric(nc["revenue"], errors="coerce").fillna(0).sum())
    c1.metric("Артикулов без себестоимости", len(nc))
    c2.metric("Юнитов продано", fmt_number(total_units))
    c3.metric("Выручка без cost", fmt_number(total_rev))

    from styles import wb_link
    hdr = (
        "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
        "<th>Продано</th><th>Выручка</th></tr>"
    )
    rows_html = ""
    for i, r in nc.iterrows():
        rows_html += (
            f'<tr><td class="ctr" style="color:#94a3b8">{i + 1}</td>'
            f'<td><b>{wb_link(r["nm_id"], r.get("supplier_article"))}</b></td>'
            f'<td>{r.get("subject") or ""}</td>'
            f'<td>{r.get("brand") or ""}</td>'
            f'<td class="num">{fmt_number(r["units_sold"])}</td>'
            f'<td class="num">{fmt_number(r["revenue"])}</td></tr>'
        )
    render_sortable_table("qnc", hdr, rows_html, height=440)

    export_buttons(nc, "articles_no_cost", sheet_name="NoCost")


# ── Sanity checks ────────────────────────────────────────────
st.markdown("### Санити-проверки")


@st.cache_data(ttl=600, show_spinner=False)
def _sanity_checks() -> list[dict]:
    eng = get_engine()
    checks = []
    with eng.connect() as c:
        # 1. Orders without sales (30d)
        try:
            q = text("""
                SELECT COUNT(DISTINCT o.nm_id)
                FROM mart.orders_daily o
                LEFT JOIN mart.sales_daily s
                  ON s.nm_id = o.nm_id
                 AND s.sales_date BETWEEN o.order_date AND o.order_date + 30
                WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
                  AND s.nm_id IS NULL
            """)
            n = c.execute(q).scalar() or 0
            checks.append({
                "Проверка": "Артикулы с заказами, но без продаж (30 дн)",
                "Значение": n,
                "Норма": "~10% и менее",
                "OK": "✓" if n < 300 else "⚠",
            })
        except Exception as e:
            checks.append({"Проверка": "Заказы без продаж", "Значение": f"error: {e}", "Норма": "", "OK": "✗"})

        # 2. Negative profit rows
        try:
            q = text("SELECT COUNT(*) FROM mart.finance_daily WHERE tax_amount < 0")
            n = c.execute(q).scalar() or 0
            checks.append({
                "Проверка": "Строк с отрицательным налогом",
                "Значение": n,
                "Норма": "0",
                "OK": "✓" if n == 0 else "⚠",
            })
        except Exception as e:
            checks.append({"Проверка": "Отрицательный налог", "Значение": f"error: {e}", "Норма": "", "OK": "✗"})

        # 3. Stocks zero
        try:
            q = text("SELECT COUNT(DISTINCT nm_id) FROM mart.stocks_snapshot WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM mart.stocks_snapshot) AND quantity = 0")
            n = c.execute(q).scalar() or 0
            checks.append({
                "Проверка": "Артикулов с нулевым остатком (последний снимок)",
                "Значение": n,
                "Норма": "зависит от ассортимента",
                "OK": "✓" if n >= 0 else "—",
            })
        except Exception as e:
            checks.append({"Проверка": "Нулевые остатки", "Значение": f"error: {e}", "Норма": "", "OK": "✗"})

        # 4. Ads without matching article
        try:
            q = text("""
                SELECT COUNT(DISTINCT a.nm_id)
                FROM mart.ads_daily a
                LEFT JOIN mart.sales_daily s ON s.nm_id = a.nm_id
                WHERE a.ads_date >= CURRENT_DATE - INTERVAL '30 days'
                  AND s.nm_id IS NULL
            """)
            n = c.execute(q).scalar() or 0
            checks.append({
                "Проверка": "Артикулы в рекламе без продаж (30 дн)",
                "Значение": n,
                "Норма": "~0 (небольшое число допустимо)",
                "OK": "✓" if n < 50 else "⚠",
            })
        except Exception as e:
            checks.append({"Проверка": "Реклама без продаж", "Значение": f"error: {e}", "Норма": "", "OK": "✗"})

    return checks


checks = _sanity_checks()
hdr = "<tr><th>Проверка</th><th>Значение</th><th>Норма</th><th>Статус</th></tr>"
rows_html = ""
for ch in checks:
    status = ch["OK"]
    color = "#10b981" if status == "✓" else ("#f59e0b" if status == "⚠" else "#ef4444")
    rows_html += (
        f'<tr><td><b>{ch["Проверка"]}</b></td>'
        f'<td class="num">{ch["Значение"]}</td>'
        f'<td style="color:#64748b">{ch["Норма"]}</td>'
        f'<td class="ctr" style="color:{color};font-weight:700">{status}</td></tr>'
    )
render_sortable_table("qsn", hdr, rows_html, height=300)
