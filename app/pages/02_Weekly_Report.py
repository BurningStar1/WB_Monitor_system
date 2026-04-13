import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

from marts import fetch_dataframe, WEEKLY_QUERY, default_date_range
from styles import inject_global_styles

inject_global_styles()
st.title("📅 Еженедельный отчёт")

with st.sidebar:
    st.header("Фильтры")
    d_def = default_date_range()
    d_from = st.date_input("Дата начала", value=d_def[0])
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(WEEKLY_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

st.markdown("### Выручка и прибыль по неделям")
fig = go.Figure()
fig.add_trace(go.Bar(x=df["year_week"], y=df["net_revenue"], name="Выручка", marker_color="#3b82f6"))
fig.add_trace(go.Bar(x=df["year_week"], y=df["profit_amount"], name="Прибыль", marker_color="#1e40af"))
fig.update_layout(barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  xaxis_title="Неделя", yaxis_title="Сумма, \u20bd", legend_title="", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Продажи и возвраты")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=df["year_week"], y=df["sales_count"], name="Продажи", marker_color="#2563eb"))
fig2.add_trace(go.Bar(x=df["year_week"], y=df["returns_count"], name="Возвраты", marker_color="#dc2626"))
fig2.update_layout(barmode="stack", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                  xaxis_title="Неделя", yaxis_title="Количество", legend_title="")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("### Детализация")
cols_map = {"year_week": "Неделя", "orders_count": "Заказы", "sales_count": "Продажи",
            "returns_count": "Возвраты", "net_revenue": "Выручка", "profit_amount": "Прибыль",
            "cost_amount": "Себестоимость"}
show = [c for c in cols_map if c in df.columns]
st.dataframe(df[show].rename(columns=cols_map), use_container_width=True, hide_index=True)

st.download_button("📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "weekly_report.csv", "text/csv")
