"""ABC-анализ — классификация артикулов по вкладу в выручку."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from marts import fetch_dataframe, ABC_QUERY, default_date_range
from styles import plotly_defaults, inject_global_styles, format_currency, fmt_number, fmt_pct_tbl, table_css, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🔤 ABC-анализ")

# ── Filters ──────────────────────────────────────────────────
d_from, d_to = date_filter_bar("abc", default_days=90)

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
    marker_opacity=0.85,
    text=top["total_revenue"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Выручка: %{y:,.0f} ₽<extra></extra>",
))
fig_pareto.add_trace(go.Scatter(
    x=top["supplier_article"],
    y=top["cumulative_share"],
    name="Нарастающий итог, %",
    yaxis="y2",
    line=dict(color=PLOTLY_COLORS["rose"], width=2.5, shape="spline"),
    mode="lines+markers",
    marker=dict(size=7, line=dict(width=1.5, color="white")),
    fill="tozeroy",
    fillcolor="rgba(244,63,94,0.08)",
    hovertemplate="Накоплено: %{y:.1f}%<extra></extra>",
))
fig_pareto.update_layout(
    **PLOTLY_LAYOUT,
    yaxis=dict(title="Выручка, ₽"),
    yaxis2=dict(title="Нарастающий итог, %", overlaying="y", side="right", range=[0, 105]),
    xaxis_tickangle=-45, legend_title="",
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    margin=dict(t=40),
    bargap=0.25,
)
plotly_defaults(fig_pareto)
st.plotly_chart(fig_pareto, use_container_width=True)

# ── Pie chart ────────────────────────────────────────────────
col_pie, col_bar = st.columns(2)
with col_pie:
    st.markdown("### Доля выручки")
    fig_pie = px.pie(summary, names="abc_category", values="revenue",
                     color="abc_category", color_discrete_map=ABC_COLORS,
                     hole=0.4)
    fig_pie.update_traces(
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="<b>Категория %{label}</b><br>Выручка: %{value:,.0f} ₽<br>Доля: %{percent}<extra></extra>",
        marker=dict(line=dict(color="white", width=2)),
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT, margin=dict(t=10, b=10))
    plotly_defaults(fig_pie)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    st.markdown("### Количество артикулов")
    fig_bar = px.bar(summary, x="abc_category", y="count",
                     color="abc_category", color_discrete_map=ABC_COLORS,
                     text="count")
    fig_bar.update_traces(
        textposition="outside",
        marker=dict(opacity=0.9, line=dict(width=0.5, color="#1e3a5f")),
        hovertemplate="<b>Категория %{x}</b><br>Артикулов: %{y}<extra></extra>",
    )
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False, xaxis_title="", yaxis_title="",
        margin=dict(t=10, b=10),
        bargap=0.25,
    )
    plotly_defaults(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Category filter ───────���──────────────────────────────────
_fcat, = st.columns(1)
with _fcat:
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
