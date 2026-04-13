"""ОПИУ — Отчёт о прибылях и убытках (P&L Report)."""
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
    FINANCE_DAILY_QUERY,
)
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, table_css,
    PLOTLY_LAYOUT, plotly_defaults, PLOTLY_COLORS,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📊 Отчёт о прибылях и убытках")

# ── Russian month names ──────────────────────────────────────
_RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}

def _ru_month(dt):
    return f"{_RU_MONTHS[dt.month]} {dt.year}"

# ── Load data ─────────────────────────────────────────────────

d_from = str(date.today() - timedelta(days=365))
d_to = str(date.today())

pnl_fin = fetch_dataframe(PNL_MONTHLY_QUERY, {})
pnl_sales = fetch_dataframe(PNL_SALES_MONTHLY_QUERY, {})

use_finance = not pnl_fin.empty
use_sales = not pnl_sales.empty

if not use_finance and not use_sales:
    st.info("Нет данных для построения ОПИУ")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────

tab_month, tab_detail = st.tabs(["По месяцам", "Детализация по дням"])

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
# TAB 1: Monthly P&L
# ══════════════════════════════════════════════════════════════

with tab_month:
    if use_finance:
        df = pnl_fin.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month", ascending=False)
        months = df["month"].tolist()
        month_labels = [_ru_month(m) for m in months]
        totals = df.sum(numeric_only=True)

        # ── Helper: value + % of net sales for one cell ──────────
        def _cell(val, base):
            """Return formatted value with % share underneath."""
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
            """Cell showing just a percentage value."""
            v = float(val)
            vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
            return f'<td class="num {vcls}">{fmt_pct_tbl(v)}</td>'

        # ── Row builders ─────────────────────────────────────────
        def pnl_row(label, key, cls="sub", sign=1):
            """Regular P&L row with values and % of net sales."""
            cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
            for _, r in df.iterrows():
                cells += _cell(float(r.get(key, 0)) * sign,
                               r.get("net_sales_before_spp", 0))
            tv = float(totals.get(key, 0)) * sign
            cells += _cell(tv, totals.get("net_sales_before_spp", 0))
            return f"<tr>{cells}</tr>"

        def pnl_subtotal(label, key, cls="subtotal-row"):
            """Bold subtotal row with values and % of net sales."""
            cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
            for _, r in df.iterrows():
                cells += _cell_bold(float(r.get(key, 0)),
                                    r.get("net_sales_before_spp", 0))
            tv = float(totals.get(key, 0))
            cells += _cell_bold(tv, totals.get("net_sales_before_spp", 0))
            return f'<tr class="{cls}">{cells}</tr>'

        def pnl_section(label, cls="subtotal-row"):
            """Section header row — no values."""
            cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
            for _ in months:
                cells += "<td></td>"
            cells += "<td></td>"
            return f'<tr class="{cls}">{cells}</tr>'

        def pnl_count_row(label, key, cls="sub"):
            """Row showing integer counts without percentages."""
            cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
            for _, r in df.iterrows():
                cells += _cell_plain(r.get(key, 0))
            cells += _cell_plain(totals.get(key, 0))
            return f"<tr>{cells}</tr>"

        def pnl_pct_row(label, values_per_month, total_val, cls="sub"):
            """Row showing percentage metrics."""
            cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
            for v in values_per_month:
                cells += _cell_pct_only(v)
            cells += _cell_pct_only(total_val)
            return f"<tr>{cells}</tr>"

        # ── Header ───────────────────────────────────────────────
        hdr = '<tr><th>Статья</th>'
        for ml in month_labels:
            hdr += f'<th>{ml}</th>'
        hdr += '<th>ИТОГО</th></tr>'

        # ── Build rows ───────────────────────────────────────────
        rows_html = ""

        # --- ВЫРУЧКА ---
        rows_html += pnl_section("ВЫРУЧКА")
        rows_html += pnl_row("Реализация (до СПП)", "sales_before_spp")
        rows_html += pnl_row("Возвраты", "returns_amount", sign=-1)
        rows_html += pnl_subtotal("= Нетто реализация", "net_sales_before_spp")

        # --- УСЛУГИ WILDBERRIES ---
        rows_html += pnl_section("УСЛУГИ WILDBERRIES")
        rows_html += pnl_row("Комиссия", "commission")
        rows_html += pnl_row("Логистика", "logistics")
        rows_html += pnl_row("Хранение", "storage")
        rows_html += pnl_row("Штрафы", "penalty")
        rows_html += pnl_row("Платная приёмка", "acceptance")
        rows_html += pnl_row("Эквайринг", "acquiring")
        rows_html += pnl_row("Удержания", "deduction")
        rows_html += pnl_row("Доп. платежи", "additional_payment")
        rows_html += pnl_subtotal("= Итого услуги WB", "total_fees")

        # --- К ПЕРЕЧИСЛЕНИЮ ---
        rows_html += pnl_subtotal("К ПЕРЕЧИСЛЕНИЮ", "ppvz_for_pay")

        # --- РАСХОДЫ ---
        rows_html += pnl_section("РАСХОДЫ")
        rows_html += pnl_row("Себестоимость", "cost_amount")
        rows_html += pnl_subtotal("= Валовая прибыль", "gross_profit")

        rows_html += pnl_row("Налог (УСН)", "tax_amount")
        rows_html += pnl_subtotal("= Чистая прибыль", "net_profit")

        # --- ПОКАЗАТЕЛИ ---
        rows_html += pnl_section("ПОКАЗАТЕЛИ")
        rows_html += pnl_count_row("Продажи, шт", "sales_count")
        rows_html += pnl_count_row("Возвраты, шт", "returns_count")

        # Маржинальность = чистая_прибыль / нетто_реализация * 100
        margin_vals = []
        for _, r in df.iterrows():
            net = float(r.get("net_sales_before_spp", 0))
            np_ = float(r.get("net_profit", 0))
            margin_vals.append(round(np_ / net * 100, 1) if net else 0)
        t_net = float(totals.get("net_sales_before_spp", 0))
        t_np = float(totals.get("net_profit", 0))
        margin_total = round(t_np / t_net * 100, 1) if t_net else 0
        rows_html += pnl_pct_row("Маржинальность, %", margin_vals, margin_total)

        # Рентабельность = чистая_прибыль / себестоимость * 100
        rent_vals = []
        for _, r in df.iterrows():
            cost = float(r.get("cost_amount", 0))
            np_ = float(r.get("net_profit", 0))
            rent_vals.append(round(np_ / cost * 100, 1) if cost else 0)
        t_cost = float(totals.get("cost_amount", 0))
        rent_total = round(t_np / t_cost * 100, 1) if t_cost else 0
        rows_html += pnl_pct_row("Рентабельность, %", rent_vals, rent_total)

        html = (
            f'{PNL_CSS}'
            f'<div class="pnl-wrap"><table class="pnl">'
            f'<thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    elif use_sales:
        # Fallback: use sales_daily aggregation (simpler structure)
        df = pnl_sales.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month", ascending=False)
        months = df["month"].tolist()
        month_labels = [_ru_month(m) for m in months]
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
# TAB 2: Daily detail
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
                deduction=("deduction_amount", "sum"),
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
            + daily["penalty"] + daily["acceptance"] + daily["deduction"]
        )

        hdr = (
            '<tr><th>Дата</th><th>Продажи</th><th>Возвраты</th><th>Нетто</th>'
            '<th>Комиссия</th><th>Логистика</th><th>Хранение</th><th>Штрафы</th>'
            '<th>Приёмка</th><th>Удержания</th><th>Итого услуги</th>'
            '<th>К перечислению</th><th>Продажи шт</th><th>Возвраты шт</th></tr>'
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
                f'<td class="num">{fmt_number(r["deduction"])}</td>'
                f'<td class="num">{fmt_number(r["total_fees"])}</td>'
                f'<td class="num {pcls}">{fmt_number(ppvz)}</td>'
                f'<td class="num">{int(r["sales_ct"])}</td>'
                f'<td class="num">{int(r["returns_ct"])}</td>'
                f'</tr>'
            )

        html = f'{PNL_CSS}<div class="pnl-wrap"><table class="pnl"><thead>{hdr}</thead><tbody>{rows}</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)

        st.download_button(
            "📥 Скачать CSV",
            daily.to_csv(index=False).encode("utf-8-sig"),
            "pnl_daily.csv",
            "text/csv",
        )
