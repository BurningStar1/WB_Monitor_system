"""РнП — Рука на Пульсе: ежедневная операционная сводка по артикулам.

Показывает заказы/продажи/прибыль по каждому артикулу за каждый день,
с возможностью перехода к недельной детализации.
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
    FIN_PROFIT_QUERY,
    ORDERS_DAILY_AMOUNT_QUERY,
    STOCKS_QUERY,
)
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, table_css,
    date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, wb_link, render_table, paginate,
    export_buttons,
)
from auth import check_auth, logout

# ── Page setup ────────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🫀 Рука на Пульсе")
st.caption("Операционная сводка: ежедневная динамика по артикулам")

# ── Filters ──────────────────────────────────────────────────
d_from, d_to = date_filter_bar("rnp", default_days=14)

params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Load data ────────────────────────────────────────────────
fin_df = fetch_dataframe(FIN_PROFIT_QUERY, params)
ord_df = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)
stk_df = fetch_dataframe(STOCKS_QUERY, {})

if fin_df.empty and ord_df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Entity filters ───────────────────────────────────────────
ref = fin_df if not fin_df.empty else ord_df
brands = sorted(ref["brand"].dropna().unique()) if "brand" in ref.columns else []
subjects = sorted(ref["subject"].dropna().unique()) if "subject" in ref.columns else []
_fc1, _fc2 = st.columns(2)
with _fc1:
    sel_brands = st.multiselect("Бренд", brands, key="rnp_brand")
with _fc2:
    sel_subjects = st.multiselect("Предмет", subjects, key="rnp_subj")

if sel_brands:
    if not fin_df.empty:
        fin_df = fin_df[fin_df["brand"].isin(sel_brands)]
    if not ord_df.empty and "brand" in ord_df.columns:
        ord_df = ord_df[ord_df["brand"].isin(sel_brands)]
if sel_subjects:
    if not fin_df.empty:
        fin_df = fin_df[fin_df["subject"].isin(sel_subjects)]
    if not ord_df.empty and "subject" in ord_df.columns:
        ord_df = ord_df[ord_df["subject"].isin(sel_subjects)]

# ── Metric selector ──────────────────────────────────────────
metric_opt = st.radio(
    "Показатель в тепловой карте",
    ["Заказы шт", "Заказы ₽", "Продажи шт", "Прибыль"],
    horizontal=True, key="rnp_metric",
)

# ── Build daily pivot ────────────────────────────────────────

# Orders pivot
if not ord_df.empty:
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    ord_pivot = (
        ord_df.groupby(["nm_id", "supplier_article", "order_date"])
        .agg(
            orders_count=("orders_count", "sum"),
            orders_amount=("orders_amount", "sum"),
        )
        .reset_index()
    )
    # Article meta
    ord_meta = (
        ord_df.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            total_orders=("orders_count", "sum"),
            total_orders_amt=("orders_amount", "sum"),
        )
        .reset_index()
    )
else:
    ord_pivot = pd.DataFrame()
    ord_meta = pd.DataFrame()

# Finance pivot
if not fin_df.empty:
    fin_df["report_date"] = pd.to_datetime(fin_df["report_date"])
    fin_pivot = (
        fin_df.groupby(["nm_id", "supplier_article", "report_date"])
        .agg(
            sales_count=("sales_count", "sum"),
            profit=("profit", "sum"),
            ppvz_for_pay=("ppvz_for_pay", "sum"),
        )
        .reset_index()
    )
    fin_meta = (
        fin_df.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            total_sales=("sales_count", "sum"),
            total_profit=("profit", "sum"),
            total_ppvz=("ppvz_for_pay", "sum"),
        )
        .reset_index()
    )
else:
    fin_pivot = pd.DataFrame()
    fin_meta = pd.DataFrame()

# Stocks
if not stk_df.empty:
    stk_agg = stk_df.groupby("nm_id").agg(stock=("quantity_full", "sum")).reset_index()
else:
    stk_agg = pd.DataFrame(columns=["nm_id", "stock"])

# ── Build article master list ─────────────────────────────────
if not fin_meta.empty:
    arts = fin_meta[["nm_id", "supplier_article", "subject", "brand", "total_sales", "total_profit", "total_ppvz"]].copy()
else:
    arts = ord_meta[["nm_id", "supplier_article", "subject", "brand"]].copy()
    arts["total_sales"] = 0
    arts["total_profit"] = 0
    arts["total_ppvz"] = 0

if not ord_meta.empty:
    arts = arts.merge(
        ord_meta[["nm_id", "supplier_article", "total_orders", "total_orders_amt"]],
        on=["nm_id", "supplier_article"], how="outer",
    )
    # Fill meta from orders if missing
    if not fin_meta.empty:
        for col in ["subject", "brand"]:
            if f"{col}_x" in arts.columns:
                arts[col] = arts[f"{col}_x"].fillna(arts.get(f"{col}_y", ""))
                arts.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True, errors="ignore")
else:
    arts["total_orders"] = 0
    arts["total_orders_amt"] = 0

arts = arts.fillna(0)
arts = arts.merge(stk_agg, on="nm_id", how="left").fillna(0)

# Sort by orders amount
sort_col = "total_orders_amt" if arts["total_orders_amt"].sum() > 0 else "total_ppvz"
arts = arts.sort_values(sort_col, ascending=False).reset_index(drop=True)

# ── Determine date range for columns ─────────────────────────
all_dates = set()
if not ord_pivot.empty:
    all_dates.update(ord_pivot["order_date"].dt.date.unique())
if not fin_pivot.empty:
    all_dates.update(fin_pivot["report_date"].dt.date.unique())
all_dates = sorted(all_dates)

if not all_dates:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Summary KPIs ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Артикулов", len(arts))
k2.metric("Заказы шт", fmt_number(int(arts["total_orders"].sum())) or "0")
k3.metric("Заказы ₽", fmt_number(arts["total_orders_amt"].sum()) or "0")
k4.metric("Продажи шт", fmt_number(int(arts["total_sales"].sum())) or "0")
k5.metric("Прибыль", fmt_number(arts["total_profit"].sum()) or "0")

# ── Build heatmap-style HTML table ───────────────────────────

TABLE_CSS = table_css("rnp") + '''<style>
.rnp th.day-h{font-size:9px;padding:4px 3px;min-width:34px;text-transform:none;letter-spacing:0}
.rnp .heat{text-align:center;font-size:11px;font-weight:600;min-width:34px;padding:3px 2px}
.rnp .heat-0{background:#f8fafc;color:#cbd5e1}
.rnp .heat-1{background:#dbeafe;color:#1e40af}
.rnp .heat-2{background:#93c5fd;color:#1e3a8a}
.rnp .heat-3{background:#3b82f6;color:#fff}
.rnp .heat-4{background:#1d4ed8;color:#fff}
.rnp .heat-neg{background:#fee2e2;color:#dc2626;font-weight:700}
.rnp .art-cell{min-width:140px}
</style>'''

# Header
hdr = '<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Остаток</th><th>Итого</th>'
for d in all_dates:
    wd = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][d.weekday()]
    hdr += f'<th class="day-h">{d.strftime("%d.%m")}<br>{wd}</th>'
hdr += '</tr>'

# Precompute lookup dicts for performance
_ord_lookup = {}
if not ord_pivot.empty:
    for _, r in ord_pivot.iterrows():
        key = (int(r["nm_id"]), r["order_date"].date())
        _ord_lookup[key] = {
            "orders_count": int(r["orders_count"]),
            "orders_amount": float(r["orders_amount"]),
        }

_fin_lookup = {}
if not fin_pivot.empty:
    for _, r in fin_pivot.iterrows():
        key = (int(r["nm_id"]), r["report_date"].date())
        _fin_lookup[key] = {
            "sales_count": int(r["sales_count"]),
            "profit": float(r["profit"]),
        }

# Compute global max for heat coloring
def _get_metric_val(nm_id, dt):
    o = _ord_lookup.get((nm_id, dt), {})
    f = _fin_lookup.get((nm_id, dt), {})
    if metric_opt == "Заказы шт":
        return o.get("orders_count", 0)
    elif metric_opt == "Заказы ₽":
        return o.get("orders_amount", 0)
    elif metric_opt == "Продажи шт":
        return f.get("sales_count", 0)
    else:  # Прибыль
        return f.get("profit", 0)

all_vals = []
for _, a in arts.iterrows():
    nm = int(a["nm_id"])
    for d in all_dates:
        all_vals.append(_get_metric_val(nm, d))

max_val = max(all_vals) if all_vals else 1
min_val = min(all_vals) if all_vals else 0

def _heat_cls(v):
    if v < 0:
        return "heat heat-neg"
    if v == 0:
        return "heat heat-0"
    ratio = v / max_val if max_val > 0 else 0
    if ratio < 0.25:
        return "heat heat-1"
    elif ratio < 0.5:
        return "heat heat-2"
    elif ratio < 0.75:
        return "heat heat-3"
    return "heat heat-4"

# Rows
display, _start, _end, _total = paginate(arts, "rnp_art", default_size=50)
rows_html = ""
for _i, (_, a) in enumerate(display.iterrows()):
    idx = _start + _i + 1
    nm = int(a["nm_id"])
    art = a.get("supplier_article", "")
    subj = a.get("subject", "")
    stock = int(a.get("stock", 0))

    # Total for the metric
    if metric_opt == "Заказы шт":
        total = int(a.get("total_orders", 0))
    elif metric_opt == "Заказы ₽":
        total = a.get("total_orders_amt", 0)
    elif metric_opt == "Продажи шт":
        total = int(a.get("total_sales", 0))
    else:
        total = a.get("total_profit", 0)

    tcls = "pos" if total > 0 else ("neg" if total < 0 else "")

    row = f'<tr><td class="ctr" style="color:#94a3b8">{idx}</td>'
    row += f'<td class="art-cell"><b>{wb_link(nm, art)}</b></td>'
    row += f'<td>{subj}</td>'
    row += f'<td class="num">{stock}</td>'
    row += f'<td class="num {tcls}"><b>{fmt_number(total)}</b></td>'

    for d in all_dates:
        v = _get_metric_val(nm, d)
        cls = _heat_cls(v)
        txt = fmt_number(v) if v != 0 else ""
        row += f'<td class="{cls}">{txt}</td>'

    row += '</tr>'
    rows_html += row

# Footer totals (sum over all articles in the report, not only visible page)
ftr = '<tr><td></td><td><b>Итого</b></td><td></td><td></td><td></td>'
for d in all_dates:
    day_sum = sum(_get_metric_val(int(a["nm_id"]), d) for _, a in arts.iterrows())
    ftr += f'<td class="heat" style="background:#f1f5f9"><b>{fmt_number(day_sum)}</b></td>'
ftr += '</tr>'

html = (
    f'{TABLE_CSS}<div class="rnp-wrap"><table class="rnp" data-sortable>'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>{SORT_JS}'
)
render_table(html)

# ── Weekly drill-down ─────────────────────────────────────────
st.markdown("---")
st.markdown("### Недельная детализация")

# Group dates by ISO week
week_data = {}
for d in all_dates:
    iso = d.isocalendar()
    wk = f"{iso.year}-W{iso.week:02d}"
    if wk not in week_data:
        week_data[wk] = {"start": d, "end": d, "dates": []}
    week_data[wk]["end"] = d
    week_data[wk]["dates"].append(d)

if week_data:
    week_labels = {
        k: f"{v['start'].strftime('%d.%m')}–{v['end'].strftime('%d.%m')}"
        for k, v in week_data.items()
    }
    sel_week = st.selectbox(
        "Выберите неделю",
        list(week_data.keys()),
        format_func=lambda k: f"{k} ({week_labels[k]})",
        key="rnp_week",
    )

    wk_dates = week_data[sel_week]["dates"]

    # Build weekly aggregation per article
    wk_rows = []
    for _, a in arts.iterrows():
        nm = int(a["nm_id"])
        wk_orders = sum(_ord_lookup.get((nm, d), {}).get("orders_count", 0) for d in wk_dates)
        wk_orders_amt = sum(_ord_lookup.get((nm, d), {}).get("orders_amount", 0) for d in wk_dates)
        wk_sales = sum(_fin_lookup.get((nm, d), {}).get("sales_count", 0) for d in wk_dates)
        wk_profit = sum(_fin_lookup.get((nm, d), {}).get("profit", 0) for d in wk_dates)
        wk_rows.append({
            "nm_id": nm,
            "supplier_article": a.get("supplier_article", ""),
            "subject": a.get("subject", ""),
            "brand": a.get("brand", ""),
            "orders_count": wk_orders,
            "orders_amount": wk_orders_amt,
            "sales_count": wk_sales,
            "profit": wk_profit,
            "stock": int(a.get("stock", 0)),
        })

    wk_df = pd.DataFrame(wk_rows)
    wk_df = wk_df.sort_values("orders_amount", ascending=False).reset_index(drop=True)

    # Weekly summary table
    wk_hdr = (
        '<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>'
        '<th>Заказы шт</th><th>Заказы ₽</th>'
        '<th>Продажи шт</th><th>Прибыль</th><th>Остаток</th></tr>'
    )
    wk_display, _ws, _we, _wt = paginate(wk_df, "rnp_wk", default_size=50)
    wk_rows_html = ""
    for i, (_, r) in enumerate(wk_display.iterrows(), _ws + 1):
        pcls = "pos" if r["profit"] > 0 else ("neg" if r["profit"] < 0 else "")
        wk_rows_html += (
            f'<tr><td class="ctr" style="color:#94a3b8">{i}</td>'
            f'<td><b>{wb_link(r["nm_id"], r["supplier_article"])}</b></td>'
            f'<td>{r["subject"]}</td>'
            f'<td>{r.get("brand", "")}</td>'
            f'<td class="num">{fmt_number(r["orders_count"])}</td>'
            f'<td class="num">{fmt_number(r["orders_amount"])}</td>'
            f'<td class="num">{fmt_number(r["sales_count"])}</td>'
            f'<td class="num {pcls}">{fmt_number(r["profit"])}</td>'
            f'<td class="num">{fmt_number(r["stock"])}</td></tr>'
        )

    WK_CSS = table_css("rnpw")
    wk_html = (
        f'{WK_CSS}<div class="rnpw-wrap"><table class="rnpw" data-sortable>'
        f'<thead>{wk_hdr}</thead><tbody>{wk_rows_html}</tbody></table></div>{SORT_JS}'
    )
    render_table(wk_html)

# ── Export ────────────────────────────────────────────────────
export_buttons(arts, "rnp_report", sheet_name="RnP")
