"""Отчёт о прибыли — детализация с водопадной диаграммой и HTML-таблицей."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from marts import fetch_dataframe, PROFIT_QUERY, default_date_range
from styles import inject_global_styles, format_currency, format_pct, fmt_number, fmt_pct_tbl, table_css, PLOTLY_LAYOUT
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("💰 Отчёт о прибыли")

# ── Helpers ──────────────────────────────────────────────────


# ── Sidebar ──────────────────────────────────────────────────
_fc1, _fc2 = st.columns(2)
d_def = default_date_range()
with _fc1:
    d_from = st.date_input("Дата начала", value=d_def[0])
with _fc2:
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(PROFIT_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# Entity filters
categories = sorted(df["subject"].dropna().unique())
brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
_fe1, _fe2 = st.columns(2)
with _fe1:
    sel_cat = st.multiselect("Категория", categories, default=[])
with _fe2:
    sel_brands = st.multiselect("Бренд", brands, default=[])
if sel_cat:
    df = df[df["subject"].isin(sel_cat)]
if sel_brands:
    df = df[df["brand"].isin(sel_brands)]

# ── KPIs ─────────────────────────────────────────────────────

total_rev = float(df["net_revenue"].sum())
total_cost = float(df["cost_amount"].sum())
total_comm = float(df["commission_amount"].sum())
total_tax = float(df["tax_amount"].sum()) if "tax_amount" in df.columns else 0
total_extra = float(df["extra_expenses_amount"].sum()) if "extra_expenses_amount" in df.columns else 0
total_profit = float(df["profit_amount"].sum())
total_op = float(df["operating_profit_amount"].sum()) if "operating_profit_amount" in df.columns else total_profit
margin = total_profit / total_rev * 100 if total_rev else 0
op_margin = total_op / total_rev * 100 if total_rev else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Выручка", format_currency(total_rev))
c2.metric("Себестоимость", format_currency(total_cost))
c3.metric("Комиссия", format_currency(total_comm))
c4.metric("Прибыль", format_currency(total_profit))
c5.metric("Маржинальность", format_pct(margin))

# ── Waterfall chart ──────────────────────────────────────────

st.markdown("### Структура финансового результата")
fig_wf = go.Figure(go.Waterfall(
    x=["Выручка", "Себестоимость", "Комиссия WB", "Доп. расходы", "Налоги", "Операц. прибыль"],
    y=[total_rev, -total_cost, -total_comm, -total_extra, -total_tax, total_op],
    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
    connector_line_color="#94a3b8",
    increasing_marker_color="#3b82f6",
    decreasing_marker_color="#dc2626",
    totals_marker_color="#1e40af",
    text=[fmt_number(total_rev), fmt_number(total_cost), fmt_number(total_comm),
          fmt_number(total_extra), fmt_number(total_tax), fmt_number(total_op)],
    textposition="outside",
))
fig_wf.update_layout(
    **PLOTLY_LAYOUT,
    yaxis_title="Сумма, ₽", showlegend=False, margin=dict(t=30),
)
st.plotly_chart(fig_wf, use_container_width=True)

# ── Daily profit trend ───────────────────────────────────────

st.markdown("### Динамика прибыли по дням")
daily = df.groupby("sales_date").agg(
    net_revenue=("net_revenue", "sum"),
    profit_amount=("profit_amount", "sum"),
    cost_amount=("cost_amount", "sum"),
).reset_index().sort_values("sales_date")

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=daily["sales_date"], y=daily["net_revenue"],
    name="Выручка", marker_color="#3b82f6", opacity=0.4,
))
fig_trend.add_trace(go.Bar(
    x=daily["sales_date"], y=daily["profit_amount"],
    name="Прибыль", marker_color="#22c55e",
))
fig_trend.update_layout(
    **PLOTLY_LAYOUT,
    barmode="overlay",
    xaxis_title="",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    margin=dict(t=40),
)
st.plotly_chart(fig_trend, use_container_width=True)

# ── Profit by article (aggregated) ──────────────────────────

st.markdown("### Прибыль по артикулам")

_agg_dict = {
    "subject": ("subject", "first"),
    "brand": ("brand", "first"),
    "net_revenue": ("net_revenue", "sum"),
    "cost_amount": ("cost_amount", "sum"),
    "commission_amount": ("commission_amount", "sum"),
    "profit_amount": ("profit_amount", "sum"),
}
if "sales_count" in df.columns:
    _agg_dict["sales_count"] = ("sales_count", "sum")
if "returns_count" in df.columns:
    _agg_dict["returns_count"] = ("returns_count", "sum")
art = df.groupby(["nm_id", "supplier_article"]).agg(**_agg_dict).reset_index()
for _c in ("sales_count", "returns_count"):
    if _c not in art.columns:
        art[_c] = 0
art["margin_pct"] = np.where(
    art["net_revenue"] > 0,
    (art["profit_amount"] / art["net_revenue"] * 100).round(1),
    0,
)
art = art.sort_values("profit_amount", ascending=False).reset_index(drop=True)

TABLE_CSS = table_css("prf")

hdr = (
    "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
    "<th>Продажи</th><th>Возвраты</th><th>Выручка</th>"
    "<th>Себестоимость</th><th>Комиссия</th><th>Прибыль</th><th>Маржа</th></tr>"
)

rows_html = ""
for idx, (_, r) in enumerate(art.head(100).iterrows(), 1):
    profit = float(r["profit_amount"])
    pcls = "pos" if profit > 0 else ("neg" if profit < 0 else "")
    m = float(r["margin_pct"])
    mcls = "pos" if m > 0 else ("neg" if m < 0 else "")

    rows_html += (
        f"<tr>"
        f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
        f'<td style="font-weight:600">{r["supplier_article"]}</td>'
        f'<td>{r["subject"]}</td>'
        f'<td>{r.get("brand", "")}</td>'
        f'<td class="num">{int(r["sales_count"])}</td>'
        f'<td class="num">{int(r["returns_count"])}</td>'
        f'<td class="num">{fmt_number(r["net_revenue"])}</td>'
        f'<td class="num">{fmt_number(r["cost_amount"])}</td>'
        f'<td class="num">{fmt_number(r["commission_amount"])}</td>'
        f'<td class="num {pcls}">{fmt_number(profit)}</td>'
        f'<td class="ctr {mcls}">{fmt_pct_tbl(m)}</td>'
        f"</tr>"
    )

# Footer
ftr_profit = art["profit_amount"].sum()
ftr_cls = "pos" if ftr_profit > 0 else ("neg" if ftr_profit < 0 else "")
ftr_margin = art["profit_amount"].sum() / art["net_revenue"].sum() * 100 if art["net_revenue"].sum() else 0
ftr_mcls = "pos" if ftr_margin > 0 else ("neg" if ftr_margin < 0 else "")
ftr = (
    f'<tr><td></td><td><b>Итого</b></td><td></td><td></td>'
    f'<td class="num">{int(art["sales_count"].sum())}</td>'
    f'<td class="num">{int(art["returns_count"].sum())}</td>'
    f'<td class="num">{fmt_number(art["net_revenue"].sum())}</td>'
    f'<td class="num">{fmt_number(art["cost_amount"].sum())}</td>'
    f'<td class="num">{fmt_number(art["commission_amount"].sum())}</td>'
    f'<td class="num {ftr_cls}">{fmt_number(ftr_profit)}</td>'
    f'<td class="ctr {ftr_mcls}">{fmt_pct_tbl(ftr_margin)}</td>'
    f'</tr>'
)

html = (
    f'{TABLE_CSS}<div class="prf-wrap"><table class="prf">'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>'
)
st.markdown(html, unsafe_allow_html=True)
st.caption(f"Показано {min(100, len(art))} из {len(art)} артикулов")

st.download_button(
    "📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
    "profit_report.csv", "text/csv",
)
