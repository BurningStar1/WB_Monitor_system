"""Юнит-экономика — пошаговый P&L на один проданный товар (CM1/CM2/CM3)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from marts import fetch_dataframe, FIN_PROFIT_QUERY
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, format_currency, format_pct,
    date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, wb_link,
    export_buttons, paginate, plotly_defaults, render_sortable_table,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🧮 Юнит-экономика")
st.caption("Разложение одного проданного юнита на составляющие: CM1 → CM2 → CM3")

# ── Filters ──────────────────────────────────────────────────
d_from, d_to = date_filter_bar("unit", default_days=30)
params = {"d_from": str(d_from), "d_to": str(d_to)}

df = fetch_dataframe(FIN_PROFIT_QUERY, params)
if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# Ensure numeric
for col in [
    "sales_count", "ppvz_for_pay", "commission_amount", "logistics_amount",
    "storage_amount", "penalty_amount", "cost_amount", "profit",
    "acceptance_amount", "acquiring_amount", "deduction_amount",
    "additional_payment_amount", "tax_amount",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# ── Aggregate per article ───────────────────────────────────
agg = df.groupby(["nm_id", "supplier_article"]).agg(
    subject=("subject", "first"),
    brand=("brand", "first"),
    units=("sales_count", "sum"),
    revenue=("ppvz_for_pay", "sum"),
    commission=("commission_amount", "sum"),
    logistics=("logistics_amount", "sum"),
    storage=("storage_amount", "sum"),
    penalty=("penalty_amount", "sum"),
    cost=("cost_amount", "sum"),
    tax=("tax_amount", "sum"),
    profit=("profit", "sum"),
).reset_index()

# Only keep articles with sales
agg = agg[agg["units"] > 0].copy()
if agg.empty:
    st.info("Нет артикулов с продажами в выбранный период")
    st.stop()

# ── Per-unit metrics ────────────────────────────────────────
for metric in ("revenue", "commission", "logistics", "storage", "penalty", "cost", "tax", "profit"):
    agg[f"{metric}_per_unit"] = agg[metric] / agg["units"]

# Contribution margins
agg["cm1"] = agg["revenue_per_unit"] - agg["commission_per_unit"]  # after WB commission
agg["cm2"] = agg["cm1"] - agg["logistics_per_unit"] - agg["storage_per_unit"] - agg["penalty_per_unit"]
agg["cm3"] = agg["cm2"] - agg["cost_per_unit"]  # after COGS
agg["cm1_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm1"] / agg["revenue_per_unit"] * 100, 0)
agg["cm2_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm2"] / agg["revenue_per_unit"] * 100, 0)
agg["cm3_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm3"] / agg["revenue_per_unit"] * 100, 0)

# ── Aggregate totals for waterfall ──────────────────────────
total_units = float(agg["units"].sum())
avg_rev = agg["revenue"].sum() / total_units
avg_comm = agg["commission"].sum() / total_units
avg_logi = agg["logistics"].sum() / total_units
avg_stor = agg["storage"].sum() / total_units
avg_pen = agg["penalty"].sum() / total_units
avg_cost = agg["cost"].sum() / total_units
avg_tax = agg["tax"].sum() / total_units
avg_profit = agg["profit"].sum() / total_units
avg_cm1 = avg_rev - avg_comm
avg_cm2 = avg_cm1 - avg_logi - avg_stor - avg_pen
avg_cm3 = avg_cm2 - avg_cost

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Юнитов", f"{int(total_units):,}".replace(",", " "))
c2.metric("Средний чек", format_currency(avg_rev))
c3.metric("CM1 / ед.", format_currency(avg_cm1), f"{avg_cm1 / avg_rev * 100:.1f}%" if avg_rev else "–")
c4.metric("CM2 / ед.", format_currency(avg_cm2), f"{avg_cm2 / avg_rev * 100:.1f}%" if avg_rev else "–")
c5.metric("Прибыль / ед.", format_currency(avg_profit), f"{avg_profit / avg_rev * 100:.1f}%" if avg_rev else "–")

# ── Waterfall: unit-level decomposition ──────────────────────
st.markdown("### Юнит-экономика (средний юнит)")
_labels = [
    "Средний чек", "− Комиссия WB", "= CM1",
    "− Логистика", "− Хранение", "− Штрафы", "= CM2",
    "− Себестоимость", "= CM3", "− Налог", "= Прибыль",
]
_vals = [
    avg_rev, -avg_comm, 0,  # CM1 is total
    -avg_logi, -avg_stor, -avg_pen, 0,  # CM2 total
    -avg_cost, 0,  # CM3 total
    -avg_tax, 0,  # Profit total
]
_measure = [
    "absolute", "relative", "total",
    "relative", "relative", "relative", "total",
    "relative", "total",
    "relative", "total",
]
_text = [fmt_number(abs(v)) if v else "" for v in _vals]
# Fix total labels with explicit values
_vals[2] = avg_cm1
_vals[6] = avg_cm2
_vals[8] = avg_cm3
_vals[10] = avg_profit
_text[2] = fmt_number(avg_cm1)
_text[6] = fmt_number(avg_cm2)
_text[8] = fmt_number(avg_cm3)
_text[10] = fmt_number(avg_profit)

fig = go.Figure(go.Waterfall(
    x=_labels,
    y=_vals,
    measure=_measure,
    connector=dict(line=dict(color="#cbd5e1", width=1, dash="dash")),
    increasing_marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color="white", width=1.5)),
    decreasing_marker=dict(color=PLOTLY_COLORS["rose"], line=dict(color="white", width=1.5)),
    totals_marker=dict(color=PLOTLY_COLORS["blue_dark"], line=dict(color="white", width=1.5)),
    text=_text,
    textposition="outside",
    textfont=dict(size=11, color="#334155", family="Inter, system-ui, sans-serif"),
    hovertemplate="<b>%{x}</b><br>%{y:,.2f} ₽<extra></extra>",
))
fig.update_layout(
    **PLOTLY_LAYOUT,
    yaxis_title="₽ на юнит", showlegend=False,
    margin=dict(l=10, r=10, t=30, b=40),
    bargap=0.25,
)
plotly_defaults(fig)
st.plotly_chart(fig, width="stretch")

# ── Per-article table ───────────────────────────────────────
st.markdown("### Детализация по артикулам")
st.caption("CM1 = Выручка − Комиссия · CM2 = CM1 − (Логистика + Хранение + Штрафы) · CM3 = CM2 − Себестоимость")

# Sort and filter
_fc1, _fc2 = st.columns(2)
with _fc1:
    sort_col = st.selectbox(
        "Сортировать по",
        ["cm3", "cm2", "cm1", "profit", "units", "revenue_per_unit", "profit_per_unit"],
        format_func=lambda x: {
            "cm3": "CM3 (убывание)", "cm2": "CM2", "cm1": "CM1",
            "profit": "Прибыль", "units": "Юниты", "revenue_per_unit": "Цена за юнит",
            "profit_per_unit": "Прибыль за юнит",
        }[x],
    )
with _fc2:
    only_negative = st.checkbox("Только убыточные (CM3 < 0)")

shown = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
if only_negative:
    shown = shown[shown["cm3"] < 0].reset_index(drop=True)

if shown.empty:
    st.info("Нет артикулов по выбранному фильтру")
else:
    display, start, end, total = paginate(shown, "unit_art", default_size=50)

    hdr = (
        "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Юниты</th>"
        "<th>Цена / ед.</th><th>Комисс. / ед.</th><th>CM1 / ед.</th><th>%</th>"
        "<th>Лог. / ед.</th><th>Хран. / ед.</th><th>CM2 / ед.</th><th>%</th>"
        "<th>Себест. / ед.</th><th>CM3 / ед.</th><th>%</th></tr>"
    )
    rows = ""
    for i, (_, r) in enumerate(display.iterrows()):
        idx = start + i + 1
        cm1_cls = "pos" if r["cm1"] > 0 else "neg"
        cm2_cls = "pos" if r["cm2"] > 0 else "neg"
        cm3_cls = "pos" if r["cm3"] > 0 else "neg"
        rows += (
            f'<tr><td class="ctr" style="color:#94a3b8">{idx}</td>'
            f'<td><b>{wb_link(r["nm_id"], r["supplier_article"])}</b></td>'
            f'<td>{r["subject"]}</td>'
            f'<td class="num">{int(r["units"])}</td>'
            f'<td class="num">{fmt_number(r["revenue_per_unit"])}</td>'
            f'<td class="num">{fmt_number(r["commission_per_unit"])}</td>'
            f'<td class="num {cm1_cls}"><b>{fmt_number(r["cm1"])}</b></td>'
            f'<td class="ctr {cm1_cls}">{fmt_pct_tbl(r["cm1_pct"])}</td>'
            f'<td class="num">{fmt_number(r["logistics_per_unit"])}</td>'
            f'<td class="num">{fmt_number(r["storage_per_unit"])}</td>'
            f'<td class="num {cm2_cls}"><b>{fmt_number(r["cm2"])}</b></td>'
            f'<td class="ctr {cm2_cls}">{fmt_pct_tbl(r["cm2_pct"])}</td>'
            f'<td class="num">{fmt_number(r["cost_per_unit"])}</td>'
            f'<td class="num {cm3_cls}"><b>{fmt_number(r["cm3"])}</b></td>'
            f'<td class="ctr {cm3_cls}">{fmt_pct_tbl(r["cm3_pct"])}</td>'
            f'</tr>'
        )
    render_sortable_table("ue", hdr, rows)

    export_buttons(agg, "unit_economics", sheet_name="UnitEconomics")
