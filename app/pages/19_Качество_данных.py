"""Качество данных — проверки полноты и целостности витрин + инлайн-фикс."""
import sys
import pathlib
from datetime import date

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

_hc1, _hc2 = st.columns([5, 1])
with _hc1:
    st.title("🧪 Качество данных")
    st.caption("Диагностика витрин: заполненность, пропуски, целостность справочников.")
with _hc2:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Обновить", key="qd_refresh", help="Очистить кеш и перечитать проверки"):
        st.cache_data.clear()
        st.rerun()

with st.expander("ℹ️ Что проверяет эта страница", expanded=False):
    st.markdown(
        """
        **4 блока проверок:**

        1. **📋 Обзор витрин** — для каждой таблицы `mart.*` считает количество
           строк и диапазон дат. *Свежесть* = дни от последней даты до сегодня.
           Норма — не более 2 дней (ETL работает стабильно).

        2. **📅 Пропуски дней** — дни внутри периода витрины без данных.
           Может означать: (а) выходные/праздники (редкие пустые дни нормальны),
           (б) падение ETL (много подряд), (в) реальное отсутствие активности.

        3. **🏷️ Артикулы без себестоимости** — продавались, но нет записи
           в `dict.cost_reference`. Без себестоимости прибыль рассчитывается
           некорректно (как вся выручка минус услуги). Нужно добавить в справочник.

        4. **🔬 Sanity-проверки** — логические аномалии:
           - Заказы без продаж (отменённые или долгие)
           - Отрицательный налог
           - Нулевые остатки (признак распродажи)
           - Реклама без соответствия в артикулах

        **Используйте эту страницу** если числа в отчётах кажутся странными —
        тут видно причину.
        """
    )


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
    "в ``dict.cost_reference``. Прибыль для них считается как 0 — это искажает отчёты. "
    "🛠️ **Заполните ячейку «Себестоимость» прямо в таблице и нажмите «Сохранить».**"
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


def _save_cost_bulk(rows: list[dict]) -> int:
    """Insert rows into dict.cost_reference. Returns number saved."""
    eng = get_engine()
    saved = 0
    with eng.begin() as conn:
        for r in rows:
            try:
                conn.execute(text("""
                    INSERT INTO dict.cost_reference
                        (nm_id, supplier_article, unit_cost, valid_from, valid_to)
                    VALUES (:nm, :sa, :c, :vf, :vt)
                """), {
                    "nm": int(r["nm_id"]),
                    "sa": str(r.get("supplier_article") or ""),
                    "c": float(r["unit_cost"]),
                    "vf": str(date.today()),
                    "vt": "2999-12-31",
                })
                saved += 1
            except Exception as e:
                st.warning(f"Ошибка для nm_id={r['nm_id']}: {e}")
    return saved


lookback = st.slider("Окно (дни)", min_value=30, max_value=180, value=90, step=30, key="nc_lb")
nc = _articles_no_cost(lookback)

if "error" in nc.columns:
    st.warning(f"Ошибка запроса: {nc.iloc[0, 0]}")
elif nc.empty:
    st.success("✅ Все активные артикулы имеют себестоимость — отлично!")
else:
    c1, c2, c3, c4 = st.columns(4)
    total_units = int(pd.to_numeric(nc["units_sold"], errors="coerce").fillna(0).sum())
    total_rev = float(pd.to_numeric(nc["revenue"], errors="coerce").fillna(0).sum())
    c1.metric("Артикулов без себестоимости", len(nc))
    c2.metric("Юнитов продано", fmt_number(total_units))
    c3.metric("Выручка без cost", fmt_number(total_rev))
    if st.session_state.get("_qd_cost_saved"):
        c4.success(f"💾 Сохранено: {st.session_state['_qd_cost_saved']}")
        st.session_state.pop("_qd_cost_saved", None)

    # Prepare editable dataframe
    edit_df = nc.copy()
    edit_df["unit_cost"] = 0.0
    edit_df = edit_df[[
        "nm_id", "supplier_article", "subject", "brand",
        "units_sold", "revenue", "unit_cost",
    ]]

    st.caption(
        "👇 Введите себестоимость в колонке «Себестоимость, ₽» для артикулов, "
        "которые хотите добавить в справочник. Строки с `0` будут пропущены."
    )
    edited = st.data_editor(
        edit_df,
        hide_index=True,
        use_container_width=True,
        height=440,
        column_config={
            "nm_id": st.column_config.NumberColumn(
                "nm_id", disabled=True, format="%d"
            ),
            "supplier_article": st.column_config.TextColumn(
                "Артикул", disabled=True, width="medium",
            ),
            "subject": st.column_config.TextColumn("Предмет", disabled=True),
            "brand": st.column_config.TextColumn("Бренд", disabled=True),
            "units_sold": st.column_config.NumberColumn(
                "Продано", disabled=True, format="%d",
            ),
            "revenue": st.column_config.NumberColumn(
                "Выручка", disabled=True, format="%.0f ₽",
            ),
            "unit_cost": st.column_config.NumberColumn(
                "Себестоимость, ₽",
                help="Введите закупочную цену в рублях",
                min_value=0.0, max_value=1_000_000.0, step=10.0,
                format="%.2f",
            ),
        },
        key="nc_editor",
    )

    # Action buttons
    _bc1, _bc2, _bc3 = st.columns([1, 1, 2])
    with _bc1:
        save_btn = st.button(
            "💾 Сохранить в справочник",
            type="primary", key="nc_save",
            use_container_width=True,
        )
    with _bc2:
        st.page_link(
            "pages/14_Справочники.py",
            label="🔧 Открыть справочник",
            use_container_width=True,
        )

    if save_btn:
        to_save = edited[edited["unit_cost"] > 0].to_dict("records")
        if not to_save:
            st.warning("Нет строк с положительной себестоимостью.")
        else:
            saved_n = _save_cost_bulk(to_save)
            st.session_state["_qd_cost_saved"] = saved_n
            st.cache_data.clear()
            st.rerun()

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
