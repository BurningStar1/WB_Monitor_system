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
from styles import inject_global_styles

inject_global_styles()
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

def _fmt(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:,.0f}".replace(",", " ")


def _fmtp(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:.1f}%"


def _pct(part, total):
    if not total or total == 0:
        return 0
    return round(part / total * 100, 1)


# ── CSS ───────────────────────────────────────────────────────

PNL_CSS = """
<style>
.pnl-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin:1rem 0;border:1px solid #e2e8f0}
.pnl{border-collapse:collapse;width:100%;font-size:12px;font-family:Inter,system-ui,sans-serif;
  background:#fff;color:#1e293b}
.pnl th{background:#f1f5f9;padding:8px 12px;border-bottom:2px solid #cbd5e1;
  border-right:1px solid #e2e8f0;font-weight:700;font-size:11px;color:#475569;
  text-align:center;white-space:nowrap}
.pnl td{padding:6px 12px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  white-space:nowrap;font-size:12px}
.pnl td:last-child,.pnl th:last-child{border-right:none}
.pnl .num{text-align:right}
.pnl .lbl{font-weight:600;color:#1e293b;padding-left:12px}
.pnl .sub{padding-left:28px;color:#475569}
.pnl .subsub{padding-left:44px;color:#64748b;font-size:11px}
.pnl .total-row td{background:#f1f5f9;font-weight:700;border-top:2px solid #cbd5e1}
.pnl .subtotal-row td{background:#f8fafc;font-weight:600;border-top:1px solid #e2e8f0}
.pnl .pos{color:#16a34a;font-weight:700}
.pnl .neg{color:#dc2626;font-weight:700}
.pnl .pct{color:#64748b;font-size:10px}
.pnl tbody tr:hover td{background:#eef2ff}
</style>
"""

# ══════════════════════════════════════════════════════════════
# TAB 1: Monthly P&L
# ══════════════════════════════════════════════════════════════

with tab_month:
    if use_finance:
        df = pnl_fin.copy()
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month", ascending=False)
        months = df["month"].tolist()

        # Build ОПИУ table — rows are P&L line items, columns are months
        month_labels = [_ru_month(m) for m in months]

        # Calculate totals
        totals = df.sum(numeric_only=True)

        def row(label, key, cls="sub", sign=1, total_val=None):
            """Build one HTML row for a P&L line."""
            cells = f'<td class="{cls} lbl" style="min-width:220px">{label}</td>'
            for _, r in df.iterrows():
                v = float(r.get(key, 0)) * sign
                vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
                base = float(r.get("net_sales_before_spp", 0)) or 1
                pct = _pct(abs(float(r.get(key, 0))), abs(base))
                cells += f'<td class="num {vcls}">{_fmt(v)}<br><span class="pct">{_fmtp(pct)}</span></td>'
            # Total column
            tv = float(total_val if total_val is not None else totals.get(key, 0)) * sign
            tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
            cells += f'<td class="num {tcls}">{_fmt(tv)}</td>'
            return f"<tr>{cells}</tr>"

        def separator(label, key=None, cls="total-row"):
            cells = f'<td class="{cls} lbl" style="min-width:220px"><b>{label}</b></td>'
            for _, r in df.iterrows():
                v = float(r.get(key, 0)) if key else 0
                vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
                cells += f'<td class="num {vcls}"><b>{_fmt(v)}</b></td>' if key else '<td></td>'
            if key:
                tv = float(totals.get(key, 0))
                tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
                cells += f'<td class="num {tcls}"><b>{_fmt(tv)}</b></td>'
            else:
                cells += '<td></td>'
            return f'<tr class="{cls}">{cells}</tr>'

        # Header
        hdr = '<tr><th>Статья</th>'
        for ml in month_labels:
            hdr += f'<th>{ml}</th>'
        hdr += '<th>ИТОГО</th></tr>'

        # Revenue block
        rows_html = ""
        rows_html += separator("Реализация (до СПП)", "net_sales_before_spp", "subtotal-row")
        rows_html += row("Продажи до СПП", "sales_before_spp")
        rows_html += row("Возвраты", "returns_amount", sign=-1)

        # Fees block
        rows_html += separator("Услуги Wildberries", "total_fees", "subtotal-row")
        rows_html += row("Комиссия", "commission")
        rows_html += row("Логистика", "logistics")
        rows_html += row("Хранение", "storage")
        rows_html += row("Штрафы", "penalty")
        rows_html += row("Платная приёмка", "acceptance")
        rows_html += row("Эквайринг", "acquiring")
        rows_html += row("Удержания", "deduction")
        rows_html += row("Доп. платежи", "additional_payment")

        # К перечислению
        rows_html += separator("К перечислению", "ppvz_for_pay", "subtotal-row")

        # Volume
        rows_html += separator("Объём", cls="subtotal-row")
        rows_html += row("Продажи, шт", "sales_count")
        rows_html += row("Возвраты, шт", "returns_count")

        html = f'{PNL_CSS}<div class="pnl-wrap"><table class="pnl"><thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>'
        st.markdown(html, unsafe_allow_html=True)

    elif use_sales:
        # Fallback: use sales_daily aggregation
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
                cells += f'<td class="num {vcls}">{_fmt(v)}</td>'
            tv = float(totals.get(key, 0)) * sign
            tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
            cells += f'<td class="num {tcls}">{_fmt(tv)}</td>'
            return f"<tr>{cells}</tr>"

        def ssep(label, key=None, cls="subtotal-row"):
            cells = f'<td class="{cls} lbl"><b>{label}</b></td>'
            for _, r in df.iterrows():
                v = float(r.get(key, 0)) if key else 0
                vcls = "pos" if v > 0 else ("neg" if v < 0 else "")
                cells += f'<td class="num {vcls}"><b>{_fmt(v)}</b></td>' if key else '<td></td>'
            if key:
                tv = float(totals.get(key, 0))
                tcls = "pos" if tv > 0 else ("neg" if tv < 0 else "")
                cells += f'<td class="num {tcls}"><b>{_fmt(tv)}</b></td>'
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

        # Margins
        for _, r in df.iterrows():
            nr = float(r.get("gross_revenue", 0))
            pr = float(r.get("profit_amount", 0))
            if nr:
                r["margin_pct"] = round(pr / nr * 100, 1)
            else:
                r["margin_pct"] = 0

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
    with st.sidebar:
        st.header("Детализация")
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
                f'<td class="num">{_fmt(r["sales_amt"])}</td>'
                f'<td class="num neg">{_fmt(r["returns_amt"])}</td>'
                f'<td class="num">{_fmt(r["net_sales"])}</td>'
                f'<td class="num">{_fmt(r["commission"])}</td>'
                f'<td class="num">{_fmt(r["logistics"])}</td>'
                f'<td class="num">{_fmt(r["storage"])}</td>'
                f'<td class="num">{_fmt(r["penalty"])}</td>'
                f'<td class="num">{_fmt(r["acceptance"])}</td>'
                f'<td class="num">{_fmt(r["deduction"])}</td>'
                f'<td class="num">{_fmt(r["total_fees"])}</td>'
                f'<td class="num {pcls}">{_fmt(ppvz)}</td>'
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
