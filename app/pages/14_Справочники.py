"""Справочники — управление себестоимостью, затратами и налоговыми ставками."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime

from db import get_engine
from marts import fetch_dataframe
from styles import inject_global_styles, fmt_number, table_css
from auth import check_auth, logout
from sqlalchemy import text

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()

st.title("\U0001f4da Справочники")
st.caption("Загрузка и редактирование справочных данных: себестоимость, доп. затраты, налоги")

# ── Helpers ──────────────────────────────────────────────────

def _get_engine():
    return get_engine()


def _load_cost_reference():
    return fetch_dataframe(
        "SELECT id, nm_id, supplier_article, unit_cost, valid_from, valid_to "
        "FROM dict.cost_reference ORDER BY nm_id, valid_from DESC"
    )


def _load_extra_expenses():
    return fetch_dataframe(
        "SELECT id, expense_date, expense_category, amount, comment "
        "FROM dict.extra_expenses ORDER BY expense_date DESC"
    )


def _load_tax_reference():
    return fetch_dataframe(
        "SELECT id, tax_name, tax_rate_percent, valid_from, valid_to "
        "FROM dict.tax_reference ORDER BY valid_from DESC"
    )


def _load_articles():
    """Get unique articles from sales data for autocomplete."""
    return fetch_dataframe(
        "SELECT DISTINCT nm_id, supplier_article, subject, brand "
        "FROM mart.sales_daily ORDER BY supplier_article"
    )


TABLE_CSS_REF = table_css("ref") + (
    '<style>'
    '.ref .act{text-align:center}'
    '.ref td:first-child{text-align:center;color:#94a3b8;font-size:11px}'
    '</style>'
)

# ── Tabs ─────────────────────────────────────────────────────

tab_cost, tab_expenses, tab_tax = st.tabs([
    "\U0001f4e6 Себестоимость",
    "\U0001f4b8 Доп. затраты",
    "\U0001f4ca Налоги",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1: Себестоимость
# ═══════════════════════════════════════════════════════════════

with tab_cost:
    st.markdown("### Себестоимость товаров")
    st.caption("Укажите себестоимость единицы товара. Загрузите из Excel или добавьте вручную.")

    # ── Upload ───────────────────────────────────────────────
    with st.expander("\U0001f4e4 Загрузить из Excel / CSV", expanded=False):
        st.markdown("""
        **Формат файла** (колонки):
        | Артикул WB (nm_id) | Артикул поставщика | Себестоимость | Дата начала | Дата окончания |
        |---|---|---|---|---|

        Минимально нужны: `nm_id` + `себестоимость`. Остальные — опционально.
        """)

        tmpl = pd.DataFrame({
            "nm_id": [123456, 789012],
            "supplier_article": ["ART-001", "ART-002"],
            "unit_cost": [350.00, 520.00],
            "valid_from": [date.today().isoformat(), date.today().isoformat()],
            "valid_to": ["2999-12-31", "2999-12-31"],
        })
        st.download_button(
            "\u2b07 Скачать шаблон",
            tmpl.to_csv(index=False).encode("utf-8-sig"),
            "cost_template.csv",
            "text/csv",
        )

        uploaded = st.file_uploader(
            "Выберите файл", type=["xlsx", "xls", "csv"],
            key="cost_upload",
        )

        if uploaded:
            if uploaded.name.endswith(".csv"):
                udf = pd.read_csv(uploaded)
            else:
                udf = pd.read_excel(uploaded)

            # Normalize column names
            col_map = {
                "nm_id": "nm_id", "артикул wb": "nm_id", "артикул_wb": "nm_id",
                "nmid": "nm_id", "nm id": "nm_id",
                "supplier_article": "supplier_article", "артикул поставщика": "supplier_article",
                "артикул_поставщика": "supplier_article",
                "unit_cost": "unit_cost", "себестоимость": "unit_cost",
                "cost": "unit_cost", "цена закупки": "unit_cost",
                "valid_from": "valid_from", "дата начала": "valid_from",
                "дата_начала": "valid_from",
                "valid_to": "valid_to", "дата окончания": "valid_to",
                "дата_окончания": "valid_to",
            }
            udf.columns = [col_map.get(c.strip().lower(), c.strip().lower()) for c in udf.columns]

            if "nm_id" not in udf.columns or "unit_cost" not in udf.columns:
                st.error("Файл должен содержать колонки `nm_id` и `unit_cost` (себестоимость)")
            else:
                udf["nm_id"] = pd.to_numeric(udf["nm_id"], errors="coerce")
                udf["unit_cost"] = pd.to_numeric(udf["unit_cost"], errors="coerce")
                udf = udf.dropna(subset=["nm_id", "unit_cost"])
                udf["nm_id"] = udf["nm_id"].astype(int)

                if "valid_from" not in udf.columns:
                    udf["valid_from"] = date.today()
                if "valid_to" not in udf.columns:
                    udf["valid_to"] = date(2999, 12, 31)
                if "supplier_article" not in udf.columns:
                    udf["supplier_article"] = ""

                st.success(f"Распознано **{len(udf)}** записей")
                st.dataframe(udf.head(20), use_container_width=True)

                if st.button(f"\U0001f4be Сохранить {len(udf)} записей в БД", key="save_cost"):
                    engine = _get_engine()
                    saved = 0
                    with engine.begin() as conn:
                        for _, r in udf.iterrows():
                            conn.execute(text("""
                                INSERT INTO dict.cost_reference (nm_id, supplier_article, unit_cost, valid_from, valid_to)
                                VALUES (:nm_id, :sa, :cost, :vf, :vt)
                            """), {
                                "nm_id": int(r["nm_id"]),
                                "sa": str(r.get("supplier_article", "")),
                                "cost": float(r["unit_cost"]),
                                "vf": str(r["valid_from"]),
                                "vt": str(r["valid_to"]),
                            })
                            saved += 1
                    st.success(f"\u2705 Сохранено {saved} записей")
                    st.rerun()

    # ── Add single record ────────────────────────────────────
    with st.expander("\u2795 Добавить вручную", expanded=False):
        articles_df = _load_articles()
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            nm_input = st.number_input("nm_id (артикул WB)", min_value=1, step=1, key="cost_nm")
        with c2:
            sa_input = st.text_input("Артикул поставщика", key="cost_sa")
        with c3:
            cost_input = st.number_input("Себестоимость, \u20bd", min_value=0.0, step=10.0, key="cost_val")

        c4, c5 = st.columns(2)
        with c4:
            vf = st.date_input("Действует с", value=date.today(), key="cost_vf")
        with c5:
            vt = st.date_input("Действует до", value=date(2999, 12, 31), key="cost_vt")

        if st.button("\U0001f4be Сохранить", key="save_cost_single"):
            if nm_input and cost_input > 0:
                engine = _get_engine()
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO dict.cost_reference (nm_id, supplier_article, unit_cost, valid_from, valid_to)
                        VALUES (:nm_id, :sa, :cost, :vf, :vt)
                    """), {
                        "nm_id": int(nm_input), "sa": sa_input,
                        "cost": float(cost_input),
                        "vf": str(vf), "vt": str(vt),
                    })
                st.success("\u2705 Запись добавлена")
                st.rerun()
            else:
                st.warning("Укажите nm_id и себестоимость > 0")

    # ── Current data ─────────────────────────────────────────
    st.markdown("### Текущие данные")
    cost_df = _load_cost_reference()
    if cost_df.empty:
        st.info("Справочник себестоимости пуст. Загрузите данные выше.")
    else:
        st.caption(f"Всего записей: **{len(cost_df)}**")
        st.dataframe(
            cost_df.style.format({
                "unit_cost": "{:,.2f} \u20bd",
            }),
            use_container_width=True,
            height=400,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 2: Доп. затраты
# ═══════════════════════════════════════════════════════════════

with tab_expenses:
    st.markdown("### Дополнительные затраты")
    st.caption("Затраты, не привязанные к конкретному артикулу: логистика до склада, фото, упаковка и т.д.")

    # ── Upload ───────────────────────────────────────────────
    with st.expander("\U0001f4e4 Загрузить из Excel / CSV", expanded=False):
        st.markdown("""
        **Формат файла**:
        | Дата | Категория | Сумма | Комментарий |
        |---|---|---|---|

        Категории: `Логистика`, `Фото`, `Упаковка`, `Маркетинг`, `Прочее`
        """)

        tmpl_exp = pd.DataFrame({
            "expense_date": [date.today().isoformat(), date.today().isoformat()],
            "expense_category": ["Логистика", "Упаковка"],
            "amount": [5000.00, 2000.00],
            "comment": ["Доставка до склада WB", "Упаковочный материал"],
        })
        st.download_button(
            "\u2b07 Скачать шаблон",
            tmpl_exp.to_csv(index=False).encode("utf-8-sig"),
            "expenses_template.csv",
            "text/csv",
        )

        uploaded_exp = st.file_uploader(
            "Выберите файл", type=["xlsx", "xls", "csv"],
            key="exp_upload",
        )

        if uploaded_exp:
            if uploaded_exp.name.endswith(".csv"):
                edf = pd.read_csv(uploaded_exp)
            else:
                edf = pd.read_excel(uploaded_exp)

            col_map_exp = {
                "expense_date": "expense_date", "дата": "expense_date",
                "date": "expense_date",
                "expense_category": "expense_category", "категория": "expense_category",
                "category": "expense_category",
                "amount": "amount", "сумма": "amount",
                "comment": "comment", "комментарий": "comment",
            }
            edf.columns = [col_map_exp.get(c.strip().lower(), c.strip().lower()) for c in edf.columns]

            required = {"expense_date", "expense_category", "amount"}
            if not required.issubset(set(edf.columns)):
                st.error(f"Нужны колонки: {required}")
            else:
                edf["amount"] = pd.to_numeric(edf["amount"], errors="coerce")
                edf = edf.dropna(subset=["amount"])
                if "comment" not in edf.columns:
                    edf["comment"] = ""

                st.success(f"Распознано **{len(edf)}** записей")
                st.dataframe(edf.head(20), use_container_width=True)

                if st.button(f"\U0001f4be Сохранить {len(edf)} затрат в БД", key="save_exp"):
                    engine = _get_engine()
                    saved = 0
                    with engine.begin() as conn:
                        for _, r in edf.iterrows():
                            conn.execute(text("""
                                INSERT INTO dict.extra_expenses (expense_date, expense_category, amount, comment)
                                VALUES (:d, :cat, :amt, :cmt)
                            """), {
                                "d": str(r["expense_date"]),
                                "cat": str(r["expense_category"]),
                                "amt": float(r["amount"]),
                                "cmt": str(r.get("comment", "")),
                            })
                            saved += 1
                    st.success(f"\u2705 Сохранено {saved} записей")
                    st.rerun()

    # ── Add single ───────────────────────────────────────────
    with st.expander("\u2795 Добавить вручную", expanded=False):
        categories = ["Логистика", "Фото", "Упаковка", "Маркетинг", "Хранение", "Прочее"]
        c1, c2 = st.columns(2)
        with c1:
            exp_date = st.date_input("Дата", value=date.today(), key="exp_date")
        with c2:
            exp_cat = st.selectbox("Категория", categories, key="exp_cat")

        c3, c4 = st.columns(2)
        with c3:
            exp_amt = st.number_input("Сумма, \u20bd", min_value=0.0, step=100.0, key="exp_amt")
        with c4:
            exp_cmt = st.text_input("Комментарий", key="exp_cmt")

        if st.button("\U0001f4be Сохранить", key="save_exp_single"):
            if exp_amt > 0:
                engine = _get_engine()
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO dict.extra_expenses (expense_date, expense_category, amount, comment)
                        VALUES (:d, :cat, :amt, :cmt)
                    """), {"d": str(exp_date), "cat": exp_cat, "amt": float(exp_amt), "cmt": exp_cmt})
                st.success("\u2705 Запись добавлена")
                st.rerun()
            else:
                st.warning("Укажите сумму > 0")

    # ── Current data ─────────────────────────────────────────
    st.markdown("### Текущие данные")
    exp_df = _load_extra_expenses()
    if exp_df.empty:
        st.info("Затрат пока нет. Добавьте выше.")
    else:
        st.caption(f"Всего записей: **{len(exp_df)}**")
        # Summary by category
        summary = exp_df.groupby("expense_category")["amount"].sum().sort_values(ascending=False)
        cols = st.columns(min(len(summary), 4))
        for i, (cat, total) in enumerate(summary.items()):
            with cols[i % len(cols)]:
                st.metric(cat, f"{total:,.0f} \u20bd".replace(",", " "))

        st.dataframe(
            exp_df.style.format({"amount": "{:,.2f} \u20bd"}),
            use_container_width=True,
            height=400,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 3: Налоги
# ═══════════════════════════════════════════════════════════════

with tab_tax:
    st.markdown("### Налоговые ставки")
    st.caption("Настройка налоговых ставок для расчёта чистой прибыли")

    with st.expander("\u2795 Добавить / обновить ставку", expanded=False):
        tc1, tc2 = st.columns(2)
        with tc1:
            tax_name = st.text_input("Название налога", value="УСН", key="tax_name")
        with tc2:
            tax_rate = st.number_input("Ставка, %", min_value=0.0, max_value=100.0,
                                        value=6.0, step=0.5, key="tax_rate")
        tc3, tc4 = st.columns(2)
        with tc3:
            tax_vf = st.date_input("Действует с", value=date.today(), key="tax_vf")
        with tc4:
            tax_vt = st.date_input("Действует до", value=date(2999, 12, 31), key="tax_vt")

        if st.button("\U0001f4be Сохранить", key="save_tax"):
            engine = _get_engine()
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO dict.tax_reference (tax_name, tax_rate_percent, valid_from, valid_to)
                    VALUES (:name, :rate, :vf, :vt)
                """), {"name": tax_name, "rate": float(tax_rate), "vf": str(tax_vf), "vt": str(tax_vt)})
            st.success("\u2705 Ставка сохранена")
            st.rerun()

    st.markdown("### Текущие ставки")
    tax_df = _load_tax_reference()
    if tax_df.empty:
        st.info("Налоговые ставки не заданы.")
    else:
        st.dataframe(
            tax_df.style.format({"tax_rate_percent": "{:.2f}%"}),
            use_container_width=True,
        )
