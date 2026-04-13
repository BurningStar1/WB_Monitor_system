"""Справочники — управление себестоимостью, затратами и налоговыми ставками.

Дизайн вдохновлён portal.rask.pro: таблица на первом плане, тулбар с поиском
и кнопками сверху, загрузка файлов через отдельную панель.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import io
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from db import get_engine
from marts import fetch_dataframe
from styles import inject_global_styles
from auth import check_auth, logout
from sqlalchemy import text

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()

# ── Page-level CSS (rask.pro style) ─────────────────────────
st.markdown("""
<style>
/* Compact header */
.ref-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 4px;
}
.ref-header h2 { margin: 0; font-size: 22px; font-weight: 600; color: #1e293b; }

/* Toolbar row */
.ref-toolbar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 12px;
}

/* Table styling — clean, white, compact */
div[data-testid="stDataFrame"] table {
    font-size: 13px !important;
}
div[data-testid="stDataFrame"] th {
    background: #f8fafc !important; color: #475569 !important;
    font-weight: 600 !important; font-size: 12px !important;
    text-transform: uppercase; letter-spacing: 0.3px;
    border-bottom: 2px solid #e2e8f0 !important;
}
div[data-testid="stDataFrame"] td {
    border-bottom: 1px solid #f1f5f9 !important;
    padding: 6px 10px !important;
}

/* Upload panel */
.upload-panel {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 20px; margin: 12px 0;
}
.upload-panel h4 { margin: 0 0 12px 0; color: #334155; font-size: 16px; }
.step-label {
    font-size: 13px; color: #64748b; font-weight: 500;
    margin: 10px 0 4px 0;
}

/* Record count badge */
.rec-badge {
    display: inline-block; background: #f1f5f9; color: #475569;
    font-size: 12px; padding: 2px 10px; border-radius: 12px;
    margin-left: 8px;
}

/* Hide default Streamlit tab padding for a cleaner look */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 14px; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="ref-header"><h2>Справочники</h2></div>', unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────

def _engine():
    return get_engine()


def _load_cost_enriched():
    """Cost reference enriched with brand/subject from sales data."""
    df = fetch_dataframe("""
        SELECT
            c.id,
            COALESCE(s.brand, '')  AS "Бренд",
            COALESCE(s.subject, '') AS "Товар",
            c.nm_id            AS "SKU",
            c.supplier_article AS "Артикул",
            c.unit_cost        AS "Себестоимость",
            c.valid_from       AS "Дата начала",
            c.valid_to         AS "Дата окончания",
            c.updated_at::date AS "Обновлено"
        FROM dict.cost_reference c
        LEFT JOIN LATERAL (
            SELECT DISTINCT ON (nm_id) brand, subject
            FROM mart.sales_daily
            WHERE nm_id = c.nm_id
            ORDER BY nm_id, sales_date DESC
            LIMIT 1
        ) s ON true
        ORDER BY c.nm_id, c.valid_from DESC
    """)
    return df


def _load_extra_expenses():
    return fetch_dataframe("""
        SELECT
            e.id,
            e.expense_date    AS "Дата",
            e.expense_category AS "Категория",
            e.amount          AS "Сумма",
            e.nm_id           AS "SKU",
            e.supplier_article AS "Артикул",
            e.comment         AS "Комментарий"
        FROM dict.extra_expenses e
        ORDER BY e.expense_date DESC
    """)


def _load_tax_reference():
    return fetch_dataframe("""
        SELECT
            t.id,
            t.tax_name         AS "Налог",
            t.tax_rate_percent AS "Ставка %",
            t.valid_from       AS "Дата начала",
            t.valid_to         AS "Дата окончания",
            t.updated_at::date AS "Обновлено"
        FROM dict.tax_reference t
        ORDER BY t.valid_from DESC
    """)


def _to_excel(df: pd.DataFrame) -> bytes:
    """Export DataFrame to Excel bytes for download."""
    buf = io.BytesIO()
    out = df.copy()
    # Strip timezone info — openpyxl doesn't support tz-aware datetimes
    for col in out.select_dtypes(include=["datetimetz"]).columns:
        out[col] = out[col].dt.tz_localize(None)
    out.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


# Column mapping for uploaded files
COL_MAP_COST = {
    "nm_id": "nm_id", "nmid": "nm_id", "nm id": "nm_id",
    "артикул wb": "nm_id", "артикул_wb": "nm_id", "sku": "nm_id",
    "supplier_article": "supplier_article", "артикул поставщика": "supplier_article",
    "артикул_поставщика": "supplier_article", "артикул": "supplier_article",
    "unit_cost": "unit_cost", "себестоимость": "unit_cost",
    "cost": "unit_cost", "цена закупки": "unit_cost",
    "valid_from": "valid_from", "дата начала": "valid_from",
    "дата_начала": "valid_from", "дата": "valid_from",
    "valid_to": "valid_to", "дата окончания": "valid_to",
    "дата_окончания": "valid_to",
}

COL_MAP_EXP = {
    "expense_date": "expense_date", "дата": "expense_date", "date": "expense_date",
    "expense_category": "expense_category", "категория": "expense_category",
    "category": "expense_category",
    "amount": "amount", "сумма": "amount",
    "nm_id": "nm_id", "nmid": "nm_id", "артикул wb": "nm_id",
    "артикул_wb": "nm_id", "sku": "nm_id",
    "supplier_article": "supplier_article", "артикул поставщика": "supplier_article",
    "артикул_поставщика": "supplier_article",
    "comment": "comment", "комментарий": "comment",
}

# ── Tabs ─────────────────────────────────────────────────────

tab_cost, tab_exp_general, tab_exp_marketing, tab_tax = st.tabs([
    "Себестоимость",
    "Затраты общие",
    "Затраты маркетинг",
    "Налоги",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: Себестоимость
# ═══════════════════════════════════════════════════════════════

with tab_cost:
    cost_df = _load_cost_enriched()

    # ── Toolbar ──────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.columns([3, 2, 1.2, 1.2, 1.2])
    with t1:
        search_q = st.text_input(
            "Поиск", placeholder="SKU, артикул, бренд...",
            label_visibility="collapsed", key="cost_search",
        )
    with t2:
        brand_opts = ["Все"] + sorted(
            cost_df["Бренд"].dropna().unique().tolist()
        ) if not cost_df.empty else ["Все"]
        brand_filter = st.selectbox(
            "Бренд", brand_opts, label_visibility="collapsed", key="cost_brand",
        )
    with t3:
        add_cost = st.button("+ Добавить", use_container_width=True, key="btn_add_cost")
    with t4:
        upload_cost = st.button("Загрузить Excel", use_container_width=True, key="btn_upload_cost")
    with t5:
        if not cost_df.empty:
            st.download_button(
                "Экспорт Excel",
                _to_excel(cost_df.drop(columns=["id"], errors="ignore")),
                "cost_reference.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("Экспорт Excel", disabled=True, use_container_width=True)

    # ── Add form (shown on button click) ─────────────────────
    if add_cost:
        st.session_state["show_add_cost"] = not st.session_state.get("show_add_cost", False)
    if upload_cost:
        st.session_state["show_upload_cost"] = not st.session_state.get("show_upload_cost", False)

    if st.session_state.get("show_add_cost"):
        st.markdown('<div class="upload-panel"><h4>Добавить запись</h4>', unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns([2, 2, 1])
        with ac1:
            nm_input = st.number_input("nm_id (SKU)", min_value=1, step=1, key="cost_nm")
        with ac2:
            sa_input = st.text_input("Артикул поставщика", key="cost_sa")
        with ac3:
            cost_input = st.number_input("Себестоимость", min_value=0.0, step=10.0, key="cost_val")
        ac4, ac5, ac6 = st.columns([1, 1, 1])
        with ac4:
            vf = st.date_input("Действует с", value=date.today(), key="cost_vf")
        with ac5:
            vt = st.date_input("Действует до", value=date(2999, 12, 31), key="cost_vt")
        with ac6:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Сохранить", key="save_cost_single", type="primary", use_container_width=True):
                if nm_input and cost_input > 0:
                    with _engine().begin() as conn:
                        conn.execute(text("""
                            INSERT INTO dict.cost_reference (nm_id, supplier_article, unit_cost, valid_from, valid_to)
                            VALUES (:nm_id, :sa, :cost, :vf, :vt)
                        """), {
                            "nm_id": int(nm_input), "sa": sa_input,
                            "cost": float(cost_input), "vf": str(vf), "vt": str(vt),
                        })
                    st.session_state["show_add_cost"] = False
                    st.rerun()
                else:
                    st.warning("Укажите nm_id и себестоимость > 0")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload panel ─────────────────────────────────────────
    if st.session_state.get("show_upload_cost"):
        st.markdown('<div class="upload-panel"><h4>Загрузка себестоимости</h4>', unsafe_allow_html=True)

        st.markdown('<div class="step-label">Шаг 1. Скачайте и заполните шаблон</div>', unsafe_allow_html=True)
        tmpl = pd.DataFrame({
            "nm_id": [123456, 789012],
            "supplier_article": ["ART-001", "ART-002"],
            "unit_cost": [350.00, 520.00],
            "valid_from": [date.today().isoformat(), date.today().isoformat()],
            "valid_to": ["2999-12-31", "2999-12-31"],
        })
        st.download_button(
            "Скачать шаблон", tmpl.to_csv(index=False).encode("utf-8-sig"),
            "cost_template.csv", "text/csv",
        )

        st.markdown('<div class="step-label">Шаг 2. Загрузите заполненный шаблон</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Перенесите файл в зону загрузки или нажмите для выбора",
            type=["xlsx", "xls", "csv"], key="cost_upload",
        )

        if uploaded:
            udf = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            udf.columns = [COL_MAP_COST.get(c.strip().lower(), c.strip().lower()) for c in udf.columns]

            if "nm_id" not in udf.columns or "unit_cost" not in udf.columns:
                st.error("Файл должен содержать колонки nm_id и unit_cost (себестоимость)")
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
                st.dataframe(udf.head(15), use_container_width=True, height=200)

                if st.button(
                    f"Сохранить {len(udf)} записей в БД",
                    key="save_cost_upload", type="primary",
                ):
                    saved = 0
                    with _engine().begin() as conn:
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
                    st.session_state["show_upload_cost"] = False
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Data table ───────────────────────────────────────────
    if cost_df.empty:
        st.info("Справочник себестоимости пуст. Нажмите «Загрузить Excel» для импорта данных.")
    else:
        # Apply filters
        df_show = cost_df.copy()
        if search_q:
            q = search_q.lower()
            mask = df_show.apply(
                lambda row: q in str(row.get("SKU", "")).lower()
                or q in str(row.get("Артикул", "")).lower()
                or q in str(row.get("Бренд", "")).lower()
                or q in str(row.get("Товар", "")).lower(),
                axis=1,
            )
            df_show = df_show[mask]
        if brand_filter and brand_filter != "Все":
            df_show = df_show[df_show["Бренд"] == brand_filter]

        cnt = len(df_show)
        st.markdown(
            f'Записей: <span class="rec-badge">{cnt}</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df_show.drop(columns=["id"], errors="ignore").style.format({
                "Себестоимость": "{:,.0f}",
            }),
            use_container_width=True,
            height=500,
        )


# ═══════════════════════════════════════════════════════════════
# Helper for both expense tabs
# ═══════════════════════════════════════════════════════════════

def _render_expenses_tab(
    tab_key: str,
    category_list: list[str],
    title: str,
):
    """Render an expense tab (shared between general and marketing)."""
    exp_df = _load_extra_expenses()

    # Filter by categories for this tab
    if not exp_df.empty:
        exp_df = exp_df[exp_df["Категория"].isin(category_list)]

    # ── Toolbar ──────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.columns([3, 2, 1.2, 1.2, 1.2])
    with t1:
        search_e = st.text_input(
            "Поиск", placeholder="Категория, артикул, комментарий...",
            label_visibility="collapsed", key=f"{tab_key}_search",
        )
    with t2:
        cat_opts = ["Все"] + category_list
        cat_filter = st.selectbox(
            "Категория", cat_opts, label_visibility="collapsed", key=f"{tab_key}_cat",
        )
    with t3:
        add_exp = st.button("+ Добавить", use_container_width=True, key=f"btn_add_{tab_key}")
    with t4:
        upload_exp = st.button("Загрузить Excel", use_container_width=True, key=f"btn_upload_{tab_key}")
    with t5:
        if not exp_df.empty:
            st.download_button(
                "Экспорт Excel",
                _to_excel(exp_df.drop(columns=["id"], errors="ignore")),
                f"{tab_key}_expenses.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("Экспорт Excel", disabled=True, use_container_width=True, key=f"exp_dis_{tab_key}")

    # ── Toggle states ────────────────────────────────────────
    if add_exp:
        st.session_state[f"show_add_{tab_key}"] = not st.session_state.get(f"show_add_{tab_key}", False)
    if upload_exp:
        st.session_state[f"show_upload_{tab_key}"] = not st.session_state.get(f"show_upload_{tab_key}", False)

    # ── Add form ─────────────────────────────────────────────
    if st.session_state.get(f"show_add_{tab_key}"):
        st.markdown(f'<div class="upload-panel"><h4>Добавить затрату</h4>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            exp_date = st.date_input("Дата", value=date.today(), key=f"{tab_key}_date")
        with c2:
            exp_cat = st.selectbox("Категория", category_list, key=f"{tab_key}_cat_input")
        c3, c4 = st.columns(2)
        with c3:
            exp_amt = st.number_input("Сумма, \u20bd", min_value=0.0, step=100.0, key=f"{tab_key}_amt")
        with c4:
            exp_cmt = st.text_input("Комментарий", key=f"{tab_key}_cmt")

        st.caption("Привязка к артикулу (оставьте 0 для Нераспределённого)")
        c5, c6, c7 = st.columns([2, 2, 1])
        with c5:
            exp_nm = st.number_input("nm_id", min_value=0, step=1, value=0, key=f"{tab_key}_nm")
        with c6:
            exp_sa = st.text_input("Артикул поставщика", key=f"{tab_key}_sa")
        with c7:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Сохранить", key=f"save_{tab_key}_single", type="primary", use_container_width=True):
                if exp_amt > 0:
                    nm_val = int(exp_nm) if exp_nm > 0 else None
                    sa_val = exp_sa.strip() or None
                    with _engine().begin() as conn:
                        conn.execute(text("""
                            INSERT INTO dict.extra_expenses
                                (expense_date, expense_category, amount, nm_id, supplier_article, comment)
                            VALUES (:d, :cat, :amt, :nm, :sa, :cmt)
                        """), {
                            "d": str(exp_date), "cat": exp_cat,
                            "amt": float(exp_amt), "nm": nm_val,
                            "sa": sa_val, "cmt": exp_cmt,
                        })
                    st.session_state[f"show_add_{tab_key}"] = False
                    st.rerun()
                else:
                    st.warning("Укажите сумму > 0")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload panel ─────────────────────────────────────────
    if st.session_state.get(f"show_upload_{tab_key}"):
        st.markdown(f'<div class="upload-panel"><h4>Загрузка затрат</h4>', unsafe_allow_html=True)

        st.markdown('<div class="step-label">Шаг 1. Скачайте и заполните шаблон</div>', unsafe_allow_html=True)
        tmpl_exp = pd.DataFrame({
            "expense_date": [date.today().isoformat()] * 3,
            "expense_category": category_list[:3] if len(category_list) >= 3 else category_list,
            "amount": [5000.00, 2000.00, 3000.00][:len(category_list[:3])],
            "nm_id": ["123456", "", ""][:len(category_list[:3])],
            "supplier_article": ["ART-001", "", ""][:len(category_list[:3])],
            "comment": ["Привязано к товару", "Нераспределённое", "Нераспределённое"][:len(category_list[:3])],
        })
        st.download_button(
            "Скачать шаблон", tmpl_exp.to_csv(index=False).encode("utf-8-sig"),
            f"{tab_key}_template.csv", "text/csv",
        )

        st.markdown('<div class="step-label">Шаг 2. Загрузите заполненный шаблон</div>', unsafe_allow_html=True)
        uploaded_exp = st.file_uploader(
            "Перенесите файл или нажмите для выбора",
            type=["xlsx", "xls", "csv"], key=f"{tab_key}_upload",
        )

        if uploaded_exp:
            edf = pd.read_csv(uploaded_exp) if uploaded_exp.name.endswith(".csv") else pd.read_excel(uploaded_exp)
            edf.columns = [COL_MAP_EXP.get(c.strip().lower(), c.strip().lower()) for c in edf.columns]

            required = {"expense_date", "expense_category", "amount"}
            if not required.issubset(set(edf.columns)):
                st.error(f"Нужны колонки: {required}")
            else:
                edf["amount"] = pd.to_numeric(edf["amount"], errors="coerce")
                edf = edf.dropna(subset=["amount"])
                if "nm_id" in edf.columns:
                    edf["nm_id"] = pd.to_numeric(edf["nm_id"], errors="coerce")
                else:
                    edf["nm_id"] = np.nan
                if "supplier_article" not in edf.columns:
                    edf["supplier_article"] = ""
                if "comment" not in edf.columns:
                    edf["comment"] = ""

                n_linked = edf["nm_id"].notna().sum()
                n_unalloc = edf["nm_id"].isna().sum()
                st.success(
                    f"Распознано **{len(edf)}** записей: "
                    f"**{n_linked}** привязано, **{n_unalloc}** нераспределённых"
                )
                st.dataframe(edf.head(15), use_container_width=True, height=200)

                if st.button(f"Сохранить {len(edf)} записей", key=f"save_{tab_key}_upload", type="primary"):
                    saved = 0
                    with _engine().begin() as conn:
                        for _, r in edf.iterrows():
                            nm = int(r["nm_id"]) if pd.notna(r["nm_id"]) else None
                            sa = str(r.get("supplier_article", "")) if pd.notna(r.get("supplier_article")) else None
                            conn.execute(text("""
                                INSERT INTO dict.extra_expenses
                                    (expense_date, expense_category, amount, nm_id, supplier_article, comment)
                                VALUES (:d, :cat, :amt, :nm, :sa, :cmt)
                            """), {
                                "d": str(r["expense_date"]),
                                "cat": str(r["expense_category"]),
                                "amt": float(r["amount"]),
                                "nm": nm, "sa": sa or None,
                                "cmt": str(r.get("comment", "")),
                            })
                            saved += 1
                    st.session_state[f"show_upload_{tab_key}"] = False
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary metrics ──────────────────────────────────────
    if not exp_df.empty:
        summary = exp_df.groupby("Категория")["Сумма"].sum().sort_values(ascending=False)
        mc = st.columns(min(len(summary), 5))
        for i, (cat, total) in enumerate(summary.items()):
            with mc[i % len(mc)]:
                st.metric(cat, f"{total:,.0f} \u20bd".replace(",", " "))

    # ── Data table ───────────────────────────────────────────
    if exp_df.empty:
        st.info(f"Затрат в категории «{title}» пока нет.")
    else:
        df_show = exp_df.copy()

        # Allocation display
        df_show["Привязка"] = df_show["SKU"].apply(
            lambda x: f"Арт. {int(x)}" if pd.notna(x) else "Нераспределённое"
        )

        # Apply filters
        if search_e:
            q = search_e.lower()
            mask = df_show.apply(
                lambda row: q in str(row.get("Категория", "")).lower()
                or q in str(row.get("Артикул", "")).lower()
                or q in str(row.get("Комментарий", "")).lower()
                or q in str(row.get("SKU", "")).lower(),
                axis=1,
            )
            df_show = df_show[mask]
        if cat_filter and cat_filter != "Все":
            df_show = df_show[df_show["Категория"] == cat_filter]

        cnt = len(df_show)
        st.markdown(
            f'Записей: <span class="rec-badge">{cnt}</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            df_show[["Дата", "Категория", "Сумма", "Привязка", "Артикул", "Комментарий"]]
            .style.format({"Сумма": "{:,.0f}"}),
            use_container_width=True,
            height=450,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2: Затраты общие
# ═══════════════════════════════════════════════════════════════

with tab_exp_general:
    _render_expenses_tab(
        tab_key="exp_gen",
        category_list=["Логистика", "Фото", "Упаковка", "Хранение", "Прочее"],
        title="Затраты общие",
    )

# ═══════════════════════════════════════════════════════════════
# TAB 3: Затраты маркетинг
# ═══════════════════════════════════════════════════════════════

with tab_exp_marketing:
    _render_expenses_tab(
        tab_key="exp_mkt",
        category_list=["Маркетинг", "Реклама WB", "Реклама внешняя", "Блогеры", "Промо"],
        title="Затраты маркетинг",
    )

# ═══════════════════════════════════════════════════════════════
# TAB 4: Налоги
# ═══════════════════════════════════════════════════════════════

with tab_tax:
    tax_df = _load_tax_reference()

    # ── Toolbar ──────────────────────────────────────────────
    t1, t2, t3 = st.columns([5, 1.2, 1.2])
    with t2:
        add_tax = st.button("+ Добавить", use_container_width=True, key="btn_add_tax")
    with t3:
        if not tax_df.empty:
            st.download_button(
                "Экспорт Excel",
                _to_excel(tax_df.drop(columns=["id"], errors="ignore")),
                "tax_reference.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("Экспорт Excel", disabled=True, use_container_width=True, key="tax_exp_dis")

    if add_tax:
        st.session_state["show_add_tax"] = not st.session_state.get("show_add_tax", False)

    # ── Add form ─────────────────────────────────────────────
    if st.session_state.get("show_add_tax"):
        st.markdown('<div class="upload-panel"><h4>Добавить налоговую ставку</h4>', unsafe_allow_html=True)
        tc1, tc2 = st.columns(2)
        with tc1:
            tax_name = st.text_input("Название налога", value="УСН", key="tax_name")
        with tc2:
            tax_rate = st.number_input("Ставка, %", min_value=0.0, max_value=100.0,
                                       value=6.0, step=0.5, key="tax_rate")
        tc3, tc4, tc5 = st.columns([1, 1, 1])
        with tc3:
            tax_vf = st.date_input("Действует с", value=date.today(), key="tax_vf")
        with tc4:
            tax_vt = st.date_input("Действует до", value=date(2999, 12, 31), key="tax_vt")
        with tc5:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Сохранить", key="save_tax", type="primary", use_container_width=True):
                with _engine().begin() as conn:
                    conn.execute(text("""
                        INSERT INTO dict.tax_reference (tax_name, tax_rate_percent, valid_from, valid_to)
                        VALUES (:name, :rate, :vf, :vt)
                    """), {"name": tax_name, "rate": float(tax_rate), "vf": str(tax_vf), "vt": str(tax_vt)})
                st.session_state["show_add_tax"] = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Data table ───────────────────────────────────────────
    if tax_df.empty:
        st.info("Налоговые ставки не заданы. Нажмите «+ Добавить» для создания.")
    else:
        cnt = len(tax_df)
        st.markdown(
            f'Записей: <span class="rec-badge">{cnt}</span>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            tax_df.drop(columns=["id"], errors="ignore").style.format({
                "Ставка %": "{:.2f}%",
            }),
            use_container_width=True,
        )
