"""Еженедельный отчёт — выручка, прибыль, маржа по неделям."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marts import fetch_dataframe, WEEKLY_QUERY, default_date_range
from styles import inject_global_styles, fmt_number, table_css, PLOTLY_LAYOUT
from auth import check_auth, logout

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📅 Еженедельный отчёт")

# ── Filters: dates ───────────────────────────────────────────
d_def = default_date_range()
_fc1, _fc2 = st.columns(2)
with _fc1:
    d_from = st.date_input("Дата начала", value=d_def[0])
with _fc2:
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(WEEKLY_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Helpers ──────────────────────────────────────────────────

def _delta(curr, prev):
    if prev == 0 or pd.isna(prev):
        return "", ""
    d = (curr - prev) / abs(prev) * 100
    cls = "up" if d > 0 else "dn"
    sign = "+" if d > 0 else ""
    return f'{sign}{d:.1f}%', cls


def _week_label(row):
    ws = pd.to_datetime(row["week_start"]).strftime("%d.%m")
    we = pd.to_datetime(row["week_end"]).strftime("%d.%m")
    return f'W{row["year_week"]}  ({ws}–{we})'


# ── KPI cards ────────────────────────────────────────────────
total_orders = int(df["orders_count"].sum())
total_sales = int(df["sales_count"].sum())
total_revenue = df["net_revenue"].sum()
total_profit = df["profit_amount"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Заказы", f"{total_orders:,}".replace(",", " "))
c2.metric("Продажи", f"{total_sales:,}".replace(",", " "))
c3.metric("Выручка", f"{total_revenue:,.0f} ₽".replace(",", " "))
c4.metric("Прибыль", f"{total_profit:,.0f} ₽".replace(",", " "))

# ── Sort ascending for charts & delta calc ───────────────────
df = df.sort_values("year_week", ascending=True).reset_index(drop=True)
df["margin_pct"] = (
    df["profit_amount"] / df["net_revenue"].replace(0, pd.NA) * 100
).fillna(0).round(1)

# ── Chart 1: Revenue + Profit bars, Margin line ─────────────
st.markdown("### Выручка, прибыль и маржа по неделям")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=df["year_week"].astype(str), y=df["net_revenue"],
           name="Выручка", marker_color="#3b82f6", opacity=0.85),
    secondary_y=False,
)
fig.add_trace(
    go.Bar(x=df["year_week"].astype(str), y=df["profit_amount"],
           name="Прибыль", marker_color="#1e40af", opacity=0.85),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=df["year_week"].astype(str), y=df["margin_pct"],
               name="Маржа %", mode="lines+markers",
               line=dict(color="#f59e0b", width=2),
               marker=dict(size=6)),
    secondary_y=True,
)
fig.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group",
)
fig.update_yaxes(title_text="Сумма, ₽", secondary_y=False)
fig.update_yaxes(title_text="Маржа, %", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# ── HTML table with weekly deltas ────────────────────────────
TABLE_CSS = table_css("wk") + (
    '<style>'
    '.wk .delta{font-size:10px;padding:2px 5px;border-radius:4px;display:inline-block}'
    '.wk .delta.up{background:#dcfce7;color:#16a34a}'
    '.wk .delta.dn{background:#fee2e2;color:#dc2626}'
    '</style>'
)

st.markdown("### Детализация по неделям")

header = (
    "<tr>"
    "<th>Неделя</th>"
    "<th>Заказы</th><th>Δ%</th>"
    "<th>Продажи</th><th>Δ%</th>"
    "<th>Возвраты</th>"
    "<th>Выручка</th><th>Δ%</th>"
    "<th>Себестоимость</th>"
    "<th>Комиссия</th>"
    "<th>Прибыль</th><th>Δ%</th>"
    "<th>Маржа%</th>"
    "</tr>"
)

rows_html = []
for i, row in df.iterrows():
    prev = df.iloc[i - 1] if i > 0 else None

    lbl = _week_label(row)
    margin = row["margin_pct"]

    d_orders, c_orders = _delta(row["orders_count"], prev["orders_count"]) if prev is not None else ("", "")
    d_sales, c_sales = _delta(row["sales_count"], prev["sales_count"]) if prev is not None else ("", "")
    d_rev, c_rev = _delta(row["net_revenue"], prev["net_revenue"]) if prev is not None else ("", "")
    d_prof, c_prof = _delta(row["profit_amount"], prev["profit_amount"]) if prev is not None else ("", "")

    def _badge(val, cls):
        if not val:
            return '<td class="ctr">—</td>'
        return f'<td class="ctr"><span class="delta {cls}">{val}</span></td>'

    rows_html.append(
        f"<tr>"
        f'<td style="font-weight:600">{lbl}</td>'
        f'<td class="num">{fmt_number(row["orders_count"])}</td>{_badge(d_orders, c_orders)}'
        f'<td class="num">{fmt_number(row["sales_count"])}</td>{_badge(d_sales, c_sales)}'
        f'<td class="num">{fmt_number(row["returns_count"])}</td>'
        f'<td class="num">{fmt_number(row["net_revenue"])}</td>{_badge(d_rev, c_rev)}'
        f'<td class="num">{fmt_number(row["cost_amount"])}</td>'
        f'<td class="num">{fmt_number(row["commission_amount"])}</td>'
        f'<td class="num">{fmt_number(row["profit_amount"])}</td>{_badge(d_prof, c_prof)}'
        f'<td class="ctr">{margin:.1f}%</td>'
        f"</tr>"
    )

# Totals footer
t_orders = int(df["orders_count"].sum())
t_sales = int(df["sales_count"].sum())
t_returns = int(df["returns_count"].sum())
t_revenue = df["net_revenue"].sum()
t_cost = df["cost_amount"].sum()
t_comm = df["commission_amount"].sum()
t_profit = df["profit_amount"].sum()
t_margin = (t_profit / t_revenue * 100) if t_revenue else 0

footer = (
    "<tr>"
    f'<td>Итого</td>'
    f'<td class="num">{fmt_number(t_orders)}</td><td></td>'
    f'<td class="num">{fmt_number(t_sales)}</td><td></td>'
    f'<td class="num">{fmt_number(t_returns)}</td>'
    f'<td class="num">{fmt_number(t_revenue)}</td><td></td>'
    f'<td class="num">{fmt_number(t_cost)}</td>'
    f'<td class="num">{fmt_number(t_comm)}</td>'
    f'<td class="num">{fmt_number(t_profit)}</td><td></td>'
    f'<td class="ctr">{t_margin:.1f}%</td>'
    "</tr>"
)

table_html = (
    TABLE_CSS
    + '<div class="wk-wrap"><table class="wk">'
    + f"<thead>{header}</thead>"
    + "<tbody>" + "\n".join(reversed(rows_html)) + "</tbody>"
    + f"<tfoot>{footer}</tfoot>"
    + "</table></div>"
)

st.markdown(table_html, unsafe_allow_html=True)

# ── Chart 2: Sales + Returns stacked bar ─────────────────────
st.markdown("### Продажи и возвраты по неделям")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=df["year_week"].astype(str), y=df["sales_count"],
    name="Продажи", marker_color="#2563eb",
))
fig2.add_trace(go.Bar(
    x=df["year_week"].astype(str), y=df["returns_count"],
    name="Возвраты", marker_color="#dc2626",
))
fig2.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    xaxis_title="Неделя", yaxis_title="Количество",
)
st.plotly_chart(fig2, use_container_width=True)

# ── CSV download ─────────────────────────────────────────────
csv_data = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 Скачать CSV",
    csv_data,
    "weekly_report.csv",
    "text/csv",
)
