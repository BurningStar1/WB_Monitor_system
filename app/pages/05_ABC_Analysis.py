import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from marts import fetch_dataframe, ABC_QUERY, default_date_range
from styles import inject_global_styles, format_currency

inject_global_styles()
st.title("🔤 ABC-анализ")

with st.sidebar:
    st.header("Фильтры")
    d_def = default_date_range()
    d_from = st.date_input("Дата начала", value=d_def[0])
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(ABC_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# Category summary
abc_colors = {"A": "#1e40af", "B": "#3b82f6", "C": "#93c5fd"}
summary = df.groupby("abc_category").agg(
    count=("nm_id", "count"),
    revenue=("total_revenue", "sum"),
).reset_index()

c1, c2, c3 = st.columns(3)
for col, cat in zip([c1, c2, c3], ["A", "B", "C"]):
    row = summary[summary["abc_category"] == cat]
    cnt = int(row["count"].values[0]) if len(row) else 0
    rev = float(row["revenue"].values[0]) if len(row) else 0
    col.metric(f"Категория {cat}", f"{cnt} артикулов", format_currency(rev))

# Pie chart
st.markdown("### Доля выручки по категориям")
fig_pie = px.pie(summary, names="abc_category", values="revenue",
                 color="abc_category", color_discrete_map=abc_colors)
fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_pie, use_container_width=True)

# Pareto chart
st.markdown("### Кривая Парето")
fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(
    x=df["supplier_article"].head(30),
    y=df["total_revenue"].head(30),
    name="Выручка",
    marker_color=[abc_colors.get(c, "#93c5fd") for c in df["abc_category"].head(30)],
))
fig_pareto.add_trace(go.Scatter(
    x=df["supplier_article"].head(30),
    y=df["cumulative_share"].head(30),
    name="Нарастающий итог, %",
    yaxis="y2",
    line=dict(color="#dc2626", width=2),
))
fig_pareto.update_layout(
    yaxis=dict(title="Выручка, \u20bd"),
    yaxis2=dict(title="Нарастающий итог, %", overlaying="y", side="right", range=[0, 105]),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis_tickangle=-45, hovermode="x unified", legend_title="",
)
st.plotly_chart(fig_pareto, use_container_width=True)

# Filter by category
with st.sidebar:
    sel_abc = st.multiselect("Показать категории", ["A", "B", "C"], default=["A", "B", "C"])
filtered = df[df["abc_category"].isin(sel_abc)]

st.markdown("### Детализация")
cols_map = {"supplier_article": "Артикул", "subject": "Категория", "brand": "Бренд",
            "total_revenue": "Выручка", "revenue_share": "Доля, %",
            "cumulative_share": "Нарастающий итог, %", "abc_category": "Класс"}
show = [c for c in cols_map if c in filtered.columns]
st.dataframe(filtered[show].rename(columns=cols_map), use_container_width=True, hide_index=True)

st.download_button("📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "abc_analysis.csv", "text/csv")
