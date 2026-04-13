import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px

from marts import fetch_dataframe, STOCKS_QUERY, STOCKS_BY_WH_QUERY
from styles import inject_global_styles, format_currency

inject_global_styles()
st.title("🏭 Остатки на складах")

df = fetch_dataframe(STOCKS_QUERY)
if df.empty:
    st.info("Нет данных об остатках")
    st.stop()

# Warehouse filter
warehouses = sorted(df["warehouse_name"].dropna().unique()) if "warehouse_name" in df.columns else []
with st.sidebar:
    st.header("Фильтры")
    sel_wh = st.multiselect("Склад", warehouses, default=[])
if sel_wh:
    df = df[df["warehouse_name"].isin(sel_wh)]

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("Позиций", len(df))
c2.metric("Общий остаток", f"{int(df['quantity_full'].sum()):,}".replace(",", " ") + " шт.")
c3.metric("В пути к клиенту", f"{int(df.get('in_way_to_client', 0).sum()):,}".replace(",", " ") + " шт.")

# Pie chart by warehouse
st.markdown("### Распределение по складам")
wh = fetch_dataframe(STOCKS_BY_WH_QUERY)
if not wh.empty:
    wh_agg = wh.groupby("warehouse_name")["quantity_full"].sum().reset_index()
    wh_agg = wh_agg.sort_values("quantity_full", ascending=False)
    fig = px.pie(wh_agg, names="warehouse_name", values="quantity_full",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# Table
st.markdown("### Детализация")
cols_map = {"supplier_article": "Артикул", "subject": "Категория", "warehouse_name": "Склад",
            "quantity": "На складе", "quantity_full": "Полный остаток",
            "in_way_to_client": "В пути к клиенту", "in_way_from_client": "В пути от клиента",
            "price": "Цена", "discount": "Скидка, %"}
show = [c for c in cols_map if c in df.columns]
st.dataframe(df[show].rename(columns=cols_map), use_container_width=True, hide_index=True)

st.download_button("📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   "stocks_report.csv", "text/csv")
