"""ABC-анализ — классификация артикулов по вкладу в выручку."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from marts import fetch_dataframe, ABC_QUERY, default_date_range
from styles import inject_global_styles, format_currency, fmt_number, fmt_pct_tbl, table_css
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🔤 ABC-анализ")

# ── Sidebar ──────────────────────────���───────────────────────
with st.sidebar:
    st.header("Фильтры")
    d_def = default_date_range()
    d_from = st.date_input("Дата начала", value=d_def[0])
    d_to = st.date_input("��ата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(ABC_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранн��й период")
    st.stop()

# ── Helpers ���─────────────────────────────────────────────────

ABC_COLORS = {"A": "#1e40af", "B": "#3b82f6", "C": "#93c5fd"}
ABC_BG = {"A": "#eff6ff", "B": "#f0f7ff", "C": "#f8fafc"}

# ── Summary KPIs ────────��────────────────────────────────────
summary = df.groupby("abc_category").agg(
    count=("nm_id", "count"),
    revenue=("total_revenue", "sum"),
).reset_index()

total_rev = df["total_revenue"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего артикулов", len(df))
for col_obj, cat in zip([c2, c3, c4], ["A", "B", "C"]):
    row = summary[summary["abc_category"] == cat]
    cnt = int(row["count"].values[0]) if len(row) else 0
    rev = float(row["revenue"].values[0]) if len(row) else 0
    pct = rev / total_rev * 100 if total_rev else 0
    col_obj.metric(
        f"Категория {cat} ({cnt} шт.)",
        format_currency(rev),
        f"{pct:.0f}% выручки",
    )

# ── Pareto chart ─────────────────────────────────────────────
st.markdown("### Кривая Парето")
top = df.head(30).copy()
fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(
    x=top["supplier_article"],
    y=top["total_revenue"],
    name="Выручка",
    marker_color=[ABC_COLORS.get(c, "#93c5fd") for c in top["abc_category"]],
    text=top["total_revenue"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
))
fig_pareto.add_trace(go.Scatter(
    x=top["supplier_article"],
    y=top["cumulative_share"],
    name="Нарастающий итог, %",
    yaxis="y2",
    line=dict(color="#dc2626", width=2, dash="dot"),
    mode="lines+markers",
))
fig_pareto.update_layout(
    yaxis=dict(title="Выручка, ₽"),
    yaxis2=dict(title="Нарастающий итог, %", overlaying="y", side="right", range=[0, 105]),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis_tickangle=-45, hovermode="x unified", legend_title="",
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    margin=dict(t=40),
)
st.plotly_chart(fig_pareto, use_container_width=True)

# ── Pie chart ────────────────────────────────────────────────
col_pie, col_bar = st.columns(2)
with col_pie:
    st.markdown("### Доля выручки")
    fig_pie = px.pie(summary, names="abc_category", values="revenue",
                     color="abc_category", color_discrete_map=ABC_COLORS)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    st.markdown("### Количество артикулов")
    fig_bar = px.bar(summary, x="abc_category", y="count",
                     color="abc_category", color_discrete_map=ABC_COLORS,
                     text="count")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, xaxis_title="", yaxis_title="",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Category filter ───────���──────────────────────────────────
with st.sidebar:
    sel_abc = st.multiselect("Показать категории", ["A", "B", "C"], default=["A", "B", "C"])
filtered = df[df["abc_category"].isin(sel_abc)]

# ── HTML table ─────────��─────────────────────────────────────
TABLE_CSS = table_css("abc") + '<style>.abc .badge{padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;display:inline-block}</style>'

st.markdown("### Детализация по артикулам")

hdr = (
    "<tr><th>#</th><th>Артикул</th><th>П��едмет</th><th>Бренд</th>"
    "<th>Класс</th><th>Выручка</th><th>Доля, %</th>"
    "<th>Нарастающий итог, %</th></tr>"
)

rows_html = ""
for idx, (_, r) in enumerate(filtered.iterrows(), 1):
    cat = r.get("abc_category", "C")
    color = ABC_COLORS.get(cat, "#93c5fd")
    bg = ABC_BG.get(cat, "#f8fafc")
    rev = float(r.get("total_revenue", 0))
    share = float(r.get("revenue_share", 0))
    cum = float(r.get("cumulative_share", 0))

    rows_html += (
        f"<tr>"
        f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
        f'<td style="font-weight:600">{r.get("supplier_article", "")}</td>'
        f'<td>{r.get("subject", "")}</td>'
        f'<td>{r.get("brand", "")}</td>'
        f'<td class="ctr"><span class="badge" style="background:{bg};color:{color};'
        f'border:1px solid {color}40">{cat}</span></td>'
        f'<td class="num">{fmt_number(rev)}</td>'
        f'<td class="ctr">{fmt_pct_tbl(share)}</td>'
        f'<td class="ctr">{fmt_pct_tbl(cum)}</td>'
        f"</tr>"
    )

# Footer
tot = filtered["total_revenue"].sum()
ftr = (
    '<tr><td></td><td><b>Итого</b></td><td></td><td></td><td></td>'
    f'<td class="num">{fmt_number(tot)}</td><td></td><td></td></tr>'
)

html = (
    f'{TABLE_CSS}<div class="abc-wrap"><table class="abc">'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>'
)
st.markdown(html, unsafe_allow_html=True)
st.caption(f"Показано {len(filtered)} из {len(df)} артикулов")

st.download_button(
    "📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
    "abc_analysis.csv", "text/csv",
)
