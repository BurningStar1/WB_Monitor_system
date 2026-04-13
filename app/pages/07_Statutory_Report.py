import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

from marts import fetch_dataframe, STATUTORY_QUERY
from styles import inject_global_styles, format_currency

inject_global_styles()
st.title("📋 Отчёт за период")

df = fetch_dataframe(STATUTORY_QUERY)

if df.empty:
    st.info("Нет данных")
    st.stop()

# Summary of latest month
latest = df.iloc[0]
st.markdown(f"### Последний месяц: {latest['period_month']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Выручка", format_currency(latest.get("net_revenue", 0)))
c2.metric("Прибыль", format_currency(latest.get("profit_amount", 0)))
c3.metric("Продажи", f"{int(latest.get('sales_count', 0)):,}".replace(",", " "))
c4.metric("Возвраты", f"{int(latest.get('returns_count', 0)):,}".replace(",", " "))

# Monthly revenue trend
st.markdown("### Помесячная динамика")
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df["period_month"], y=df["net_revenue"],
    name="Выручка", marker_color="#3b82f6",
))
fig.add_trace(go.Bar(
    x=df["period_month"], y=df["profit_amount"],
    name="Прибыль", marker_color="#1e40af",
))
fig.add_trace(go.Scatter(
    x=df["period_month"], y=df["operating_profit_amount"],
    name="Операц. прибыль", line=dict(color="#dc2626", width=2),
))
fig.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Месяц",
    yaxis_title="Сумма, \u20bd",
    legend_title="",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# Table
st.markdown("### Детализация")
cols_map = {"period_month": "Месяц", "orders_count": "Заказы", "sales_count": "Продажи",
            "returns_count": "Возвраты", "gross_revenue": "Валовая выручка",
            "net_revenue": "Чистая выручка", "commission_amount": "Комиссия",
            "cost_amount": "Себестоимость", "tax_amount": "Налог",
            "profit_amount": "Прибыль", "operating_profit_amount": "Операц. прибыль"}
show = [c for c in cols_map if c in df.columns]
st.dataframe(df[show].rename(columns=cols_map), use_container_width=True, hide_index=True)

st.download_button("📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "statutory_report.csv", "text/csv")
