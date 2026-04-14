"""Когортный анализ — артикулы группируются по месяцу первой продажи."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from marts import fetch_dataframe
from styles import (
    inject_global_styles, fmt_number, PLOTLY_LAYOUT, PLOTLY_COLORS,
    export_buttons, plotly_defaults, render_sortable_table,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("👥 Когортный анализ артикулов")
st.caption(
    "Артикулы группируются по месяцу, когда у них впервые были продажи. "
    "Показывает, как сохраняются продажи когорты на 1, 2, 3-м месяце жизни."
)

# ── Controls ────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    months_back = st.selectbox(
        "Глубина истории",
        [6, 9, 12, 18, 24],
        index=2,
        format_func=lambda x: f"{x} мес.",
    )
with c2:
    metric_choice = st.selectbox(
        "Метрика",
        ["units", "revenue"],
        format_func=lambda x: {"units": "Количество (шт)", "revenue": "Выручка (₽)"}[x],
    )
with c3:
    view_mode = st.selectbox(
        "Вид",
        ["retention", "absolute"],
        format_func=lambda x: {"retention": "Retention (% от 1-го месяца)", "absolute": "Абсолютные значения"}[x],
    )

# ── Data load ───────────────────────────────────────────────
from datetime import date
from dateutil.relativedelta import relativedelta
d_to = date.today()
d_from = d_to - relativedelta(months=months_back)
params = {"d_from": str(d_from), "d_to": str(d_to)}

# Use sales_daily for cohorts
SALES_QUERY = """
SELECT sales_date, nm_id, supplier_article, subject, brand,
       sales_count, net_revenue
FROM mart.sales_daily
WHERE sales_date BETWEEN :d_from AND :d_to
ORDER BY sales_date;
"""
df = fetch_dataframe(SALES_QUERY, params)
if df.empty:
    st.info("Нет данных о продажах за выбранный период")
    st.stop()

df["sales_date"] = pd.to_datetime(df["sales_date"])
df["sales_count"] = pd.to_numeric(df["sales_count"], errors="coerce").fillna(0)
df["revenue"] = pd.to_numeric(df["net_revenue"], errors="coerce").fillna(0)
df["month"] = df["sales_date"].dt.to_period("M")

# First sales month per article
first_sale = df.groupby("nm_id")["sales_date"].min().dt.to_period("M")
first_sale.name = "cohort"
df = df.merge(first_sale, left_on="nm_id", right_index=True)

# Cohort age in months
df["cohort_age"] = (df["month"] - df["cohort"]).apply(lambda x: x.n)

# Aggregate by cohort × age
if metric_choice == "units":
    agg = df.groupby(["cohort", "cohort_age"])["sales_count"].sum().unstack(fill_value=0)
else:
    agg = df.groupby(["cohort", "cohort_age"])["revenue"].sum().unstack(fill_value=0)

# Cohort size (articles in each cohort)
cohort_sizes = df.groupby("cohort")["nm_id"].nunique()

# Sort by cohort date
agg = agg.sort_index()

# Normalize if retention view
display = agg.copy()
if view_mode == "retention":
    first_col = display.iloc[:, 0].replace(0, np.nan)
    display = display.div(first_col, axis=0) * 100
    display = display.round(1)

# ── KPIs ────────────────────────────────────────────────────
n_cohorts = len(cohort_sizes)
n_articles = df["nm_id"].nunique()
oldest_cohort = cohort_sizes.index.min()
newest_cohort = cohort_sizes.index.max()

# Average month-2 retention for cohorts old enough
if 1 in agg.columns and view_mode == "retention" and len(display) > 1:
    m1_vals = display.iloc[:-1, 1].dropna()
    avg_m1_retention = float(m1_vals.mean()) if len(m1_vals) else 0
else:
    avg_m1_retention = 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Когорт", str(n_cohorts))
k2.metric("Артикулов", f"{n_articles:,}".replace(",", " "))
k3.metric("Первая когорта", str(oldest_cohort))
if view_mode == "retention":
    k4.metric("Ср. retention M+1", f"{avg_m1_retention:.1f}%")
else:
    total = float(agg.values.sum())
    k4.metric("Всего", fmt_number(total))

# ── Heatmap ─────────────────────────────────────────────────
st.markdown("### Тепловая карта когорт")

if display.empty:
    st.info("Недостаточно данных для построения тепловой карты")
else:
    z = display.values
    x_labels = [f"M+{c}" for c in display.columns]
    y_labels = [str(idx) for idx in display.index]

    # Text annotations
    if view_mode == "retention":
        text = [[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in z]
        hover = "Когорта %{y}<br>Возраст %{x}<br>Retention: %{z:.1f}%<extra></extra>"
        colorbar_title = "%"
    else:
        text = [[fmt_number(v) if v else "" for v in row] for row in z]
        hover = "Когорта %{y}<br>Возраст %{x}<br>Значение: %{z:,.0f}<extra></extra>"
        colorbar_title = "Значение"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#0f172a"),
        colorscale=[
            [0.0, "#f8fafc"],
            [0.2, "#dbeafe"],
            [0.5, "#93c5fd"],
            [0.8, "#3b82f6"],
            [1.0, "#1d4ed8"],
        ],
        hovertemplate=hover,
        colorbar=dict(title=colorbar_title, thickness=12, len=0.8),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Возраст когорты (месяцев с первой продажи)",
        yaxis_title="Когорта (месяц первой продажи)",
        yaxis=dict(autorange="reversed"),
        height=max(350, 40 * len(display)),
        margin=dict(l=10, r=10, t=30, b=40),
    )
    plotly_defaults(fig)
    st.plotly_chart(fig, width="stretch")

# ── Cohort size table ────────────────────────────────────────
st.markdown("### Размер когорт")
st.caption("Количество артикулов, впервые появившихся в продаже в данном месяце.")

cohort_rev = df.groupby("cohort")["revenue"].sum()
cohort_units = df.groupby("cohort")["sales_count"].sum()
cs_df = pd.DataFrame({
    "Когорта": [str(c) for c in cohort_sizes.index],
    "Артикулов": cohort_sizes.values,
    "Юнитов (все время)": cohort_units.reindex(cohort_sizes.index).values,
    "Выручка (все время)": cohort_rev.reindex(cohort_sizes.index).values,
})

hdr = (
    "<tr><th>#</th><th>Когорта</th><th>Артикулов</th>"
    "<th>Юнитов (все время)</th><th>Выручка (все время)</th></tr>"
)
rows = ""
for i, r in cs_df.iterrows():
    rows += (
        f'<tr><td class="ctr" style="color:#94a3b8">{i + 1}</td>'
        f'<td><b>{r["Когорта"]}</b></td>'
        f'<td class="num">{int(r["Артикулов"])}</td>'
        f'<td class="num">{fmt_number(r["Юнитов (все время)"])}</td>'
        f'<td class="num">{fmt_number(r["Выручка (все время)"])}</td>'
        f'</tr>'
    )
render_sortable_table("cohort", hdr, rows, height=400)

# ── Export ──────────────────────────────────────────────────
export_df = display.reset_index().rename(columns={"cohort": "Когорта"})
export_df.columns = ["Когорта"] + [f"M+{c}" for c in display.columns]
export_buttons(export_df, "cohorts", sheet_name="Cohorts")
