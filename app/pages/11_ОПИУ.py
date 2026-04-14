"""ОПИУ — Отчёт о прибылях и убытках (P&L Report).

Supports three granularity levels: monthly, weekly (ISO), and daily.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from marts import (
    fetch_dataframe,
    PNL_MONTHLY_QUERY,
    PNL_SALES_MONTHLY_QUERY,
    FIN_WEEKLY_QUERY,
    FINANCE_DAILY_QUERY,
)
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, table_css,
    PLOTLY_LAYOUT, plotly_defaults, PLOTLY_COLORS, SORT_JS, render_table,
    export_buttons,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📊 Отчёт о прибылях и убытках")

with st.expander("ℹ️ Как читать ОПИУ", expanded=False):
    st.markdown(
        """
        **ОПИУ** — отчёт о прибылях и убытках по правилам Raskка, в трёх
        разрезах (месяцы / недели / дни). Источник — финансовые отчёты WB.

        **Структура отчёта:**
        1. *Реализация до СПП* = `sales_amount − returns_amount`
        2. **− СПП** (скидка постоянного покупателя WB)
        3. *= Реализация после СПП* = `retail_amount`
        4. **− Плановая комиссия WB**
        5. *= К перечислению* = `ppvz_for_pay`
        6. **− Расходы**: логистика, хранение, штрафы, приёмка, эквайринг,
           удержания, себестоимость; **+ Доплаты** WB
        7. *= Валовая маржа (EBITDA)*
        8. **− Налог** (УСН)
        9. *= Чистая прибыль*

        **Показатели внизу:**
        - *ЧП / Реализация до СПП, %* — основная маржа (Raskка-совместимо)
        - *ЧП / Себестоимость, %* — ROI по закупке

        Проценты в таблице — доля от **Реализации до СПП** (база).
        Суммы в ₽ без НДС.
        """
    )

# ── Russian month names ──────────────────────────────────────
_RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}

def _ru_month(dt):
    return f"{_RU_MONTHS[dt.month]} {dt.year}"

# ── Load data ─────────────────────────────────────────────────

pnl_fin = fetch_dataframe(PNL_MONTHLY_QUERY, {})
pnl_sales = fetch_dataframe(PNL_SALES_MONTHLY_QUERY, {})

use_finance = not pnl_fin.empty
use_sales = not pnl_sales.empty

if not use_finance and not use_sales:
    st.info("Нет данных для построения ОПИУ")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────

tab_month, tab_week, tab_detail = st.tabs(["По месяцам", "По неделям", "По дням"])

# ── Formatting helpers ────────────────────────────────────────

def _pct(part, total):
    if not total or total == 0:
        return 0
    return round(part / total * 100, 1)


# ── CSS ───────────────────────────────────────────────────────

PNL_CSS = table_css("pnl") + (
    '<style>'
    '.pnl th{padding:8px 12px;font-weight:700}'
    '.pnl td{padding:6px 12px}'
    '.pnl td:last-child,.pnl th:last-child{border-right:none}'
    '.pnl .lbl{font-weight:600;color:#1e293b;padding-left:12px}'
    '.pnl .sub{padding-left:28px;color:#475569}'
    '.pnl .subsub{padding-left:44px;color:#64748b;font-size:11px}'
    '.pnl .total-row td{background:#f1f5f9;font-weight:700;border-top:2px solid #cbd5e1}'
    '.pnl .subtotal-row td{background:#f8fafc;font-weight:600;border-top:1px solid #e2e8f0}'
    '</style>'
)


# ══════════════════════════════════════════════════════════════
# Shared P&L renderer
# ══════════════════════════════════════════════════════════════

def _render_pnl_table(df, col_labels, base_key="net_sales_before_spp"):
    """Render a full P&L HTML table for the given dataframe.

    df: one row per period, columns match PNL_MONTHLY_QUERY output.
    col_labels: list of header strings for each period column.
    """
    totals = df.sum(numeric_only=True)

    def _cell(val, base):
        v = float(val)
        vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
        pct = _pct(abs(v), abs(float(base))) if base else 0
        return (
            f'<td class="num {vcls}">'
            f'{fmt_number(v)}<br>'
            f'<span class="pct">{fmt_pct_tbl(pct)}</span></td>'
        )

    def _cell_bold(val, base):
        v = float(val)
        vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
        pct = _pct(abs(v), abs(float(base))) if base else 0
        return (
            f'<td class="num {vcls}">'
            f'<b>{fmt_number(v)}</b><br>'
            f'<span class="pct">{fmt_pct_tbl(pct)}</span></td>'
        )

    def _cell_plain(val):
        v = float(val)
        vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
        return f'<td class="num {vcls}">{fmt_number(v)}</td>'

    def _cell_pct_only(val):
        v = float(val)
        vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
        return f'<td class="num {vcls}">{fmt_pct_tbl(v)}</td>'

    def pnl_row(label, key, cls="sub", sign=1):
        cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
        for _, r in df.iterrows():
            cells += _cell(float(r.get(key, 0)) * sign, r.get(base_key, 0))
        tv = float(totals.get(key, 0)) * sign
        cells += _cell(tv, totals.get(base_key, 0))
        return f"<tr>{cells}</tr>"

    def pnl_subtotal(label, key, cls="subtotal-row"):
        cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
        for _, r in df.iterrows():
            cells += _cell_bold(float(r.get(key, 0)), r.get(base_key, 0))
        tv = float(totals.get(key, 0))
        cells += _cell_bold(tv, totals.get(base_key, 0))
        return f'<tr class="{cls}">{cells}</tr>'

    def pnl_section(label, cls="subtotal-row"):
        cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
        for _ in range(len(df)):
            cells += "<td></td>"
        cells += "<td></td>"
        return f'<tr class="{cls}">{cells}</tr>'

    def pnl_count_row(label, key, cls="sub"):
        cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
        for _, r in df.iterrows():
            cells += _cell_plain(r.get(key, 0))
        cells += _cell_plain(totals.get(key, 0))
        return f"<tr>{cells}</tr>"

    def pnl_pct_row(label, values_per_period, total_val, cls="sub"):
        cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
        for v in values_per_period:
            cells += _cell_pct_only(v)
        cells += _cell_pct_only(total_val)
        return f"<tr>{cells}</tr>"

    # Header
    hdr = '<tr><th>Статья</th>'
    for ml in col_labels:
        hdr += f'<th>{ml}</th>'
    hdr += '<th>ИТОГО</th></tr>'

    # Compute EBITDA (Валовая маржа) matching the reference:
    # ppvz_for_pay - cost - logistics - storage - penalty - acceptance - acquiring - deduction + additional_payment
    def _ebitda(r):
        return (
            float(r.get("ppvz_for_pay", 0))
            - float(r.get("cost_amount", 0))
            - float(r.get("logistics", 0))
            - float(r.get("storage", 0))
            - float(r.get("penalty", 0))
            - float(r.get("acceptance", 0))
            - float(r.get("acquiring", 0))
            - float(r.get("deduction", 0))
            + float(r.get("additional_payment", 0))
        )

    def _net_profit_calc(r):
        return _ebitda(r) - float(r.get("tax_amount", 0))

    def pnl_computed_row(label, func, cls="subtotal-row"):
        """Bold row with computed values per period."""
        cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
        for _, r in df.iterrows():
            v = func(r)
            base = float(r.get(base_key, 0))
            cells += _cell_bold(v, base)
        tv = sum(func(r) for _, r in df.iterrows())
        cells += _cell_bold(tv, totals.get(base_key, 0))
        return f'<tr class="{cls}">{cells}</tr>'

    # Computed SPP per period (Реализация до СПП − после СПП)
    def _spp(r):
        pre = float(r.get("net_sales_before_spp", 0))
        post = float(r.get("retail_amount", 0))
        return max(pre - post, 0) if post else 0

    # Build rows
    rows_html = ""

    # --- ВЫРУЧКА ---
    rows_html += pnl_section("ВЫРУЧКА")
    rows_html += pnl_row("Реализация (до СПП)", "sales_before_spp")
    rows_html += pnl_row("Возвраты", "returns_amount", sign=-1)
    rows_html += pnl_subtotal("= Нетто реализация до СПП", "net_sales_before_spp")
    rows_html += pnl_computed_row("− СПП (скидка постоянного покупателя)",
                                   lambda r: -_spp(r), cls="sub")
    rows_html += pnl_subtotal("= Реализация после СПП", "retail_amount")

    # --- КОМИССИЯ ---
    rows_html += pnl_row("Плановая комиссия WB", "commission", sign=-1)
    rows_html += pnl_subtotal("= К ПЕРЕЧИСЛЕНИЮ", "ppvz_for_pay")

    # --- РАСХОДЫ ---
    rows_html += pnl_section("РАСХОДЫ")
    rows_html += pnl_row("Себестоимость", "cost_amount")
    rows_html += pnl_row("Логистика", "logistics")
    rows_html += pnl_row("Хранение", "storage")
    rows_html += pnl_row("Штрафы", "penalty")
    rows_html += pnl_row("Платная приёмка", "acceptance")
    rows_html += pnl_row("Эквайринг", "acquiring")
    rows_html += pnl_row("Удержания", "deduction")
    rows_html += pnl_row("Доп. платежи", "additional_payment")

    # --- EBITDA / Валовая маржа ---
    rows_html += pnl_computed_row("= Валовая маржа (EBITDA)", _ebitda)

    rows_html += pnl_row("Налог (УСН)", "tax_amount")
    rows_html += pnl_computed_row("= Чистая прибыль", _net_profit_calc)

    # --- ПОКАЗАТЕЛИ ---
    rows_html += pnl_section("ПОКАЗАТЕЛИ")
    rows_html += pnl_count_row("Продажи, шт", "sales_count")
    rows_html += pnl_count_row("Возвраты, шт", "returns_count")

    # ЧП / Реализация до СПП
    margin_vals = []
    for _, r in df.iterrows():
        net = float(r.get("net_sales_before_spp", 0))
        np_ = _net_profit_calc(r)
        margin_vals.append(round(np_ / net * 100, 1) if net else 0)
    t_net = float(totals.get("net_sales_before_spp", 0))
    t_np = sum(_net_profit_calc(r) for _, r in df.iterrows())
    margin_total = round(t_np / t_net * 100, 1) if t_net else 0
    rows_html += pnl_pct_row("ЧП / Реализация до СПП, %", margin_vals, margin_total)

    # ЧП / Себестоимость (рентабельность)
    rent_vals = []
    for _, r in df.iterrows():
        cost = float(r.get("cost_amount", 0))
        np_ = _net_profit_calc(r)
        rent_vals.append(round(np_ / cost * 100, 1) if cost else 0)
    t_cost = float(totals.get("cost_amount", 0))
    rent_total = round(t_np / t_cost * 100, 1) if t_cost else 0
    rows_html += pnl_pct_row("ЧП / Себестоимость, %", rent_vals, rent_total)

    html = (
        f'{PNL_CSS}'
        f'<div class="pnl-wrap"><table class="pnl">'
        f'<thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>'
    )
    return html


# ══════════════════════════════════════════════════════════════
# TAB 1: Monthly P&L
# ══════════════════════════════════════════════════════════════

with tab_month:
    if use_finance:
        df = pnl_fin.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month", ascending=False)
        month_labels = [_ru_month(m) for m in df["month"]]

        st.markdown(
            _render_pnl_table(df, month_labels),
            unsafe_allow_html=True,
        )

    elif use_sales:
        df = pnl_sales.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month", ascending=False)
        month_labels = [_ru_month(m) for m in df["month"]]
        totals = df.sum(numeric_only=True)

        def srow(label, key, cls="sub", sign=1):
            cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
            for _, r in df.iterrows():
                v = float(r.get(key, 0)) * sign
                vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
                cells += f'<td class="num {vcls}">{fmt_number(v)}</td>'
            tv = float(totals.get(key, 0)) * sign
            tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
            cells += f'<td class="num {tcls}">{fmt_number(tv)}</td>'
            return f"<tr>{cells}</tr>"

        def ssep(label, key=None, cls="subtotal-row"):
            cells = f'<td class="{cls} lbl"><b>{label}</b></td>'
            for _, r in df.iterrows():
                v = float(r.get(key, 0)) if key else 0
                vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
                cells += f'<td class="num {vcls}"><b>{fmt_number(v)}</b></td>' if key else '<td></td>'
            if key:
                tv = float(totals.get(key, 0))
                tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
                cells += f'<td class="num {tcls}"><b>{fmt_number(tv)}</b></td>'
            else:
                cells += '<td></td>'
            return f'<tr class="{cls}">{cells}</tr>'

        hdr = '<tr><th>Статья</th>'
        for ml in month_labels:
            hdr += f'<th>{ml}</th>'
        hdr += '<th>ИТОГО</th></tr>'

        rows_html = ""
        rows_html += ssep("Выручка (до СПП)", "gross_revenue")
        rows_html += ssep("Выручка (после СПП)", "net_revenue")
        rows_html += srow("Комиссия", "commission")
        rows_html += srow("Себестоимость", "cost_amount")
        rows_html += srow("Налоги", "tax_amount")
        rows_html += ssep("Прибыль", "profit_amount", "total-row")
        rows_html += ssep("Операционная прибыль", "operating_profit", "total-row")
        rows_html += srow("Заказы, шт", "orders_count")
        rows_html += srow("Продажи, шт", "sales_count")
        rows_html += srow("Возвраты, шт", "returns_count")

        html = f'{PNL_CSS}<div class="pnl-wrap"><table class="pnl"><thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)

    st.caption("Данные на основе финансовых отчётов Wildberries")

# ══════════════════════════════════════════════════════════════
# TAB 2: Weekly P&L
# ══════════════════════════════════════════════════════════════

with tab_week:
    if not use_finance:
        st.info("Для недельного ОПИУ необходимы данные из финансовых отчётов (mart.finance_daily)")
    else:
        # Load weekly data — use last 365 days
        weekly_params = {
            "d_from": str(date.today() - timedelta(days=365)),
            "d_to": str(date.today()),
        }
        wdf = fetch_dataframe(FIN_WEEKLY_QUERY, weekly_params)

        if wdf.empty:
            st.info("Нет данных за выбранный период")
        else:
            # FIN_WEEKLY_QUERY returns different column names — adapt to shared renderer
            wdf = wdf.rename(columns={
                "sales_amount": "sales_before_spp",
                "total_wb_fees": "total_fees",
                "realization_pre_spp": "net_sales_before_spp",
            })
            # acquiring and additional_payment are in the query; ensure fallback
            for col in ["acquiring", "additional_payment", "returns_amount", "retail_amount"]:
                if col not in wdf.columns:
                    wdf[col] = 0
                else:
                    wdf[col] = pd.to_numeric(wdf[col], errors="coerce").fillna(0)
            wdf["gross_profit"] = wdf["ppvz_for_pay"] - wdf["cost_amount"]
            wdf["net_profit"] = wdf["profit"]

            wdf = wdf.sort_values("year_week", ascending=False)

            # Limit to last N weeks
            n_weeks_opt = st.selectbox("Количество недель", [8, 12, 16, 26, 52], index=1, key="pnl_n_weeks")
            wdf_display = wdf.head(n_weeks_opt)

            # Build week labels: "DD.MM–DD.MM"
            week_labels = []
            for _, r in wdf_display.iterrows():
                ws = pd.to_datetime(r["week_start"]).strftime("%d.%m")
                we = pd.to_datetime(r["week_end"]).strftime("%d.%m")
                week_labels.append(f"{ws}–{we}")

            st.markdown(
                _render_pnl_table(wdf_display, week_labels),
                unsafe_allow_html=True,
            )
            st.caption("Данные агрегированы по ISO-неделям")

# ══════════════════════════════════════════════════════════════
# TAB 3: Daily detail
# ══════════════════════════════════════════════════════════════

with tab_detail:
    fcol1, _ = st.columns([1, 3])
    with fcol1:
        detail_days = st.selectbox("Период", [30, 60, 90, 180], index=0)

    params = {
        "d_from": str(date.today() - timedelta(days=detail_days)),
        "d_to": str(date.today()),
    }
    fin_df = fetch_dataframe(FINANCE_DAILY_QUERY, params)

    if fin_df.empty:
        st.info("Нет данных по финансовым отчётам за выбранный период")
    else:
        # Aggregate by date
        daily = (
            fin_df.groupby("report_date")
            .agg(
                sales_amt=("sales_amount", "sum"),
                returns_amt=("returns_amount", "sum"),
                commission=("commission_amount", "sum"),
                logistics=("logistics_amount", "sum"),
                storage=("storage_amount", "sum"),
                penalty=("penalty_amount", "sum"),
                acceptance=("acceptance_amount", "sum"),
                acquiring=("acquiring_amount", "sum"),
                deduction=("deduction_amount", "sum"),
                additional=("additional_payment_amount", "sum"),
                ppvz=("ppvz_for_pay", "sum"),
                sales_ct=("sales_count", "sum"),
                returns_ct=("returns_count", "sum"),
            )
            .reset_index()
            .sort_values("report_date", ascending=False)
        )
        daily["net_sales"] = daily["sales_amt"] - daily["returns_amt"]
        daily["total_fees"] = (
            daily["commission"] + daily["logistics"] + daily["storage"]
            + daily["penalty"] + daily["acceptance"] + daily["acquiring"]
            + daily["deduction"] - daily["additional"]
        )

        hdr = (
            '<tr><th>Дата</th><th>Продажи</th><th>Возвраты</th><th>Нетто</th>'
            '<th>Комиссия</th><th>Логистика</th><th>Хранение</th><th>Штрафы</th>'
            '<th>Приёмка</th><th>Эквайринг</th><th>Удержания</th><th>Допл.</th>'
            '<th>Итого услуги</th>'
            '<th>К перечислению</th><th>Прод.&nbsp;шт</th><th>Возвр.&nbsp;шт</th></tr>'
        )
        rows = ""
        for _, r in daily.iterrows():
            dt = pd.to_datetime(r["report_date"]).strftime("%d.%m.%Y")
            ppvz = float(r["ppvz"])
            pcls = "pos" if ppvz > 0 else ("neg" if ppvz < 0 else "")
            rows += (
                f'<tr><td>{dt}</td>'
                f'<td class="num">{fmt_number(r["sales_amt"])}</td>'
                f'<td class="num neg">{fmt_number(r["returns_amt"])}</td>'
                f'<td class="num">{fmt_number(r["net_sales"])}</td>'
                f'<td class="num">{fmt_number(r["commission"])}</td>'
                f'<td class="num">{fmt_number(r["logistics"])}</td>'
                f'<td class="num">{fmt_number(r["storage"])}</td>'
                f'<td class="num">{fmt_number(r["penalty"])}</td>'
                f'<td class="num">{fmt_number(r["acceptance"])}</td>'
                f'<td class="num">{fmt_number(r["acquiring"])}</td>'
                f'<td class="num">{fmt_number(r["deduction"])}</td>'
                f'<td class="num">{fmt_number(r["additional"])}</td>'
                f'<td class="num">{fmt_number(r["total_fees"])}</td>'
                f'<td class="num {pcls}">{fmt_number(ppvz)}</td>'
                f'<td class="num">{int(r["sales_ct"])}</td>'
                f'<td class="num">{int(r["returns_ct"])}</td>'
                f'</tr>'
            )

        html = (
            f'{PNL_CSS}<div class="pnl-wrap"><table class="pnl" data-sortable>'
            f'<thead>{hdr}</thead><tbody>{rows}</tbody></table></div>{SORT_JS}'
        )
        render_table(html)

        export_buttons(daily, "pnl_daily", sheet_name="P&L")
