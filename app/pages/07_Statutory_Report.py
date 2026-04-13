"""Отчёт за период — помесячная сводка с трендами и HTML-таблицей."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marts import fetch_dataframe, STATUTORY_QUERY
from styles import inject_global_styles, format_currency, fmt_number, fmt_pct_tbl, table_css
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📋 Отчёт за период")

# ── Helpers ──────────────────────────────────────────────────

_RU_MONTHS = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def _delta_badge(curr, prev):
    if prev == 0 or pd.isna(prev) or pd.isna(curr):
        return ""
    d = (curr - prev) / abs(prev) * 100
    cls = "up" if d > 0 else "dn"
    sign = "+" if d > 0 else ""
    return f'<span class="delta {cls}">{sign}{d:.1f}%</span>'

# ── Load data ────────────────────────────────────────────────

df = fetch_dataframe(STATUTORY_QUERY)

if df.empty:
    st.info("Нет данных")
    st.stop()

# Parse period_month to datetime for sorting and formatting
df["_month_dt"] = pd.to_datetime(df["period_month"])
df = df.sort_values("_month_dt", ascending=False)
df["_label"] = df["_month_dt"].apply(lambda dt: f"{_RU_MONTHS[dt.month]} {dt.year}")

# ── Latest month summary ─────────────────────────────────────

latest = df.iloc[0]
prev = df.iloc[1] if len(df) > 1 else None

st.markdown(f"### {latest['_label']}")

c1, c2, c3, c4, c5 = st.columns(5)

def _delta_str(curr, prev_val):
    if prev_val is None or prev_val == 0:
        return None
    d = (curr - prev_val) / abs(prev_val) * 100
    return f"{d:+.1f}%"

c1.metric("Выручка", format_currency(latest.get("net_revenue", 0)),
          _delta_str(latest.get("net_revenue", 0), prev.get("net_revenue") if prev is not None else None))
c2.metric("Прибыль", format_currency(latest.get("profit_amount", 0)),
          _delta_str(latest.get("profit_amount", 0), prev.get("profit_amount") if prev is not None else None))
c3.metric("Операц. прибыль", format_currency(latest.get("operating_profit_amount", 0)),
          _delta_str(latest.get("operating_profit_amount", 0), prev.get("operating_profit_amount") if prev is not None else None))
c4.metric("Продажи", f'{int(latest.get("sales_count", 0)):,}'.replace(",", " ") + " шт.")
c5.metric("Возвраты", f'{int(latest.get("returns_count", 0)):,}'.replace(",", " ") + " шт.")

# ── Monthly trend chart ──────────────────────────────────────

st.markdown("### Помесячная динамика")
chart_df = df.sort_values("_month_dt")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(
    x=chart_df["_label"], y=chart_df["net_revenue"],
    name="Выручка", marker_color="#3b82f6",
    text=chart_df["net_revenue"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
), secondary_y=False)
fig.add_trace(go.Bar(
    x=chart_df["_label"], y=chart_df["profit_amount"],
    name="Прибыль", marker_color="#22c55e",
    text=chart_df["profit_amount"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
), secondary_y=False)

# Margin % line
chart_df = chart_df.copy()
chart_df["margin_pct"] = np.where(
    chart_df["net_revenue"] > 0,
    (chart_df["operating_profit_amount"] / chart_df["net_revenue"] * 100).round(1),
    0,
)
fig.add_trace(go.Scatter(
    x=chart_df["_label"], y=chart_df["margin_pct"],
    name="% маржинальности",
    line=dict(color="#dc2626", width=2, dash="dot"),
    mode="lines+markers+text",
    text=chart_df["margin_pct"].apply(lambda v: f"{v:.1f}%"),
    textposition="top center", textfont=dict(size=10),
), secondary_y=True)

fig.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    hovermode="x unified", margin=dict(t=50),
)
fig.update_yaxes(title_text="Сумма, ₽", secondary_y=False)
fig.update_yaxes(title_text="Маржа, %", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

# ── HTML table ───────────────────────────────────────────────

st.markdown("### Детализация по месяцам")

TABLE_CSS = table_css("stat") + '<style>.stat .delta{font-size:10px;padding:2px 5px;border-radius:4px;display:inline-block}.stat .delta.up{background:#dcfce7;color:#16a34a}.stat .delta.dn{background:#fee2e2;color:#dc2626}</style>'

metric_cols = [
    ("orders_count", "Заказы"),
    ("sales_count", "Продажи"),
    ("returns_count", "Возвраты"),
    ("gross_revenue", "Валовая"),
    ("net_revenue", "Выручка"),
    ("commission_amount", "Комиссия"),
    ("cost_amount", "Себестоим."),
    ("profit_amount", "Прибыль"),
    ("operating_profit_amount", "Операц.\nприбыль"),
]

hdr = '<tr><th>Месяц</th>'
for _, label in metric_cols:
    hdr += f'<th>{label}</th><th>Δ</th>'
hdr += '<th>Маржа</th></tr>'

# Iterate from newest to oldest; use next row as "previous" for delta
rows_list = df.to_dict("records")
rows_html = ""
for i, r in enumerate(rows_list):
    prev_r = rows_list[i + 1] if i + 1 < len(rows_list) else None
    row_html = f'<tr><td style="font-weight:600">{r["_label"]}</td>'

    for col, _ in metric_cols:
        cv = float(r.get(col, 0))
        pv = float(prev_r.get(col, 0)) if prev_r else 0
        pcls = ""
        if col in ("profit_amount", "operating_profit_amount", "net_revenue"):
            pcls = " pos" if cv > 0 else (" neg" if cv < 0 else "")
        row_html += f'<td class="num{pcls}">{fmt_number(cv)}</td>'
        row_html += f'<td class="ctr">{_delta_badge(cv, pv)}</td>'

    # Margin %
    rev = float(r.get("net_revenue", 0))
    profit = float(r.get("operating_profit_amount", 0))
    margin = profit / rev * 100 if rev else 0
    mcls = "pos" if margin > 0 else ("neg" if margin < 0 else "")
    row_html += f'<td class="ctr {mcls}">{fmt_pct_tbl(margin)}</td>'
    row_html += '</tr>'
    rows_html += row_html

# Footer totals
ftr = '<tr><td><b>Итого</b></td>'
for col, _ in metric_cols:
    total = float(df[col].sum()) if col in df.columns else 0
    pcls = ""
    if col in ("profit_amount", "operating_profit_amount", "net_revenue"):
        pcls = " pos" if total > 0 else (" neg" if total < 0 else "")
    ftr += f'<td class="num{pcls}">{fmt_number(total)}</td><td></td>'
tot_rev = df["net_revenue"].sum() if "net_revenue" in df.columns else 0
tot_profit = df["operating_profit_amount"].sum() if "operating_profit_amount" in df.columns else 0
tot_margin = tot_profit / tot_rev * 100 if tot_rev else 0
tot_mcls = "pos" if tot_margin > 0 else ("neg" if tot_margin < 0 else "")
ftr += f'<td class="ctr {tot_mcls}">{fmt_pct_tbl(tot_margin)}</td></tr>'

html = (
    f'{TABLE_CSS}<div class="stat-wrap"><table class="stat">'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>'
)
st.markdown(html, unsafe_allow_html=True)

st.download_button(
    "📥 Скачать CSV", df.drop(columns=["_month_dt", "_label"]).to_csv(index=False).encode("utf-8-sig"),
    "statutory_report.csv", "text/csv",
)
