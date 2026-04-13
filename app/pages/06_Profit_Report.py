import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from marts import fetch_dataframe, PROFIT_QUERY, default_date_range
from styles import inject_global_styles, format_currency, format_pct

inject_global_styles()
st.title("💰 Отчёт о прибыли")

with st.sidebar:
    st.header("Фильтры")
    d_def = default_date_range()
    d_from = st.date_input("Дата начала", value=d_def[0])
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(PROFIT_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# Category filter
categories = sorted(df["subject"].dropna().unique())
with st.sidebar:
    sel_cat = st.multiselect("Категория", categories, default=[])
if sel_cat:
    df = df[df["subject"].isin(sel_cat)]

# Summary KPIs
total_rev = df["net_revenue"].sum()
total_cost = df["cost_amount"].sum()
total_profit = df["profit_amount"].sum()
margin = (total_rev - total_cost) / total_rev * 100 if total_rev else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Выручка", format_currency(total_rev))
c2.metric("Себестоимость", format_currency(total_cost))
c3.metric("Прибыль", format_currency(total_profit))
c4.metric("Маржинальность", format_pct(margin))

# Waterfall chart
st.markdown("### Структура финансового результата")
total_commission = df["commission_amount"].sum()
total_tax = df["tax_amount"].sum()
total_extra = df["extra_expenses_amount"].sum()
total_op = df["operating_profit_amount"].sum()

fig_wf = go.Figure(go.Waterfall(
    x=["Выручка", "Себестоимость", "Комиссия WB", "Доп. расходы", "Налоги", "Операц. прибыль"],
    y=[total_rev, -total_cost, -total_commission, -total_extra, -total_tax, total_op],
    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
    connector_line_color="#94a3b8",
    increasing_marker_color="#2563eb",
    decreasing_marker_color="#dc2626",
    totals_marker_color="#1e40af",
))
fig_wf.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                     yaxis_title="Сумма, \u20bd", showlegend=False)
st.plotly_chart(fig_wf, use_container_width=True)

# Daily profit trend
st.markdown("### Динамика прибыли по дням")
daily = df.groupby("sales_date").agg(
    net_revenue=("net_revenue", "sum"),
    profit_amount=("profit_amount", "sum"),
).reset_index()
fig_trend = px.area(daily, x="sales_date", y="profit_amount",
                    labels={"sales_date": "Дата", "profit_amount": "Прибыль, \u20bd"},
                    color_discrete_sequence=["#2563eb"])
fig_trend.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="", hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)

# Table
st.markdown("### Детализация")
cols_map = {"sales_date": "Дата", "supplier_article": "Артикул", "subject": "Категория",
            "net_revenue": "Выручка", "cost_amount": "Себестоимость",
            "commission_amount": "Комиссия", "tax_amount": "Налог",
            "profit_amount": "Прибыль", "margin_pct": "Маржа, %"}
show = [c for c in cols_map if c in df.columns]
st.dataframe(df[show].rename(columns=cols_map), use_container_width=True, hide_index=True)

st.download_button("📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "profit_report.csv", "text/csv")
