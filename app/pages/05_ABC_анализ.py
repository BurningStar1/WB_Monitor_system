"""ABC-анализ — мульти-метрика с комбинированными группами AAA/AAB/..."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from marts import fetch_dataframe, ABC_QUERY, FIN_PROFIT_QUERY, ORDERS_DAILY_AMOUNT_QUERY, default_date_range
from styles import plotly_defaults, inject_global_styles, format_currency, fmt_number, fmt_pct_tbl, table_css, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, wb_link, render_table, export_buttons
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🔤 ABC-анализ")

with st.expander("ℹ️ Как читать отчёт", expanded=False):
    st.markdown(
        """
        **ABC-анализ** делит артикулы на 3 группы по выбранной метрике (правило Парето):
        - **A** — топ-группа, обеспечивает основной вклад (по умолчанию 0–80%)
        - **B** — средние артикулы (80–95%)
        - **C** — «длинный хвост» (95–100%)

        Если ниже выбрано **«Все 3 метрики»**, каждому артикулу присваивается
        тройной код *XYZ* (например, **AAB** — A по выручке, A по прибыли, B по продажам).
        Это помогает увидеть артикулы, которые дают выручку, но не прибыль.

        Границы A/B в процентах меняются слева — 70/90 делает классификацию строже.
        """
    )

# ── Filters ──────────────────────────────────────────────────
d_from, d_to = date_filter_bar("abc", default_days=90)

# ── Metric & boundary selectors ──────────────────────────────
_mc1, _mc2, _mc3, _mc4 = st.columns([2, 1, 1, 1])
with _mc1:
    metric_name = st.selectbox(
        "Основная метрика (для графика Парето)",
        ["Выручка", "Операционная прибыль", "Количество продаж"],
        index=0, key="abc_metric",
    )
with _mc2:
    pct_a = st.number_input("Граница A, %", min_value=1, max_value=99, value=80, key="abc_pct_a")
with _mc3:
    pct_b = st.number_input("Граница B, %", min_value=1, max_value=99, value=95, key="abc_pct_b")
with _mc4:
    st.caption(f"A: 0–{pct_a}%  \nB: {pct_a}–{pct_b}%  \nC: {pct_b}–100%")

params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Load data ───────────────────────────────────────────────
fin_df = fetch_dataframe(FIN_PROFIT_QUERY, params)
ord_df = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)

articles = pd.DataFrame()

if not fin_df.empty:
    fin_agg = (
        fin_df.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            revenue=("ppvz_for_pay", "sum"),
            profit=("profit", "sum"),
            sales_count=("sales_count", "sum"),
        )
        .reset_index()
    )
    articles = fin_agg

if not ord_df.empty:
    ord_agg = (
        ord_df.groupby("nm_id")  # join by nm_id only — supplier_article may differ across marts
        .agg(
            orders_count=("orders_count", "sum"),
            orders_amount=("orders_amount", "sum"),
        )
        .reset_index()
    )
    if articles.empty:
        ord_agg_full = (
            ord_df.groupby(["nm_id", "supplier_article"])
            .agg(
                subject=("subject", "first"),
                brand=("brand", "first"),
                orders_count=("orders_count", "sum"),
                orders_amount=("orders_amount", "sum"),
            )
            .reset_index()
        )
        ord_agg_full["revenue"] = ord_agg_full["orders_amount"]
        ord_agg_full["profit"] = 0
        ord_agg_full["sales_count"] = ord_agg_full["orders_count"]
        articles = ord_agg_full
    else:
        articles = articles.merge(ord_agg, on="nm_id", how="left")
        articles["orders_count"] = articles["orders_count"].fillna(0)
        articles["orders_amount"] = articles["orders_amount"].fillna(0)

if articles.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── ABC classification by ALL 3 metrics simultaneously ──────

def _abc_for_column(vals, pct_a, pct_b):
    """Classify articles into A/B/C by cumulative share of given values."""
    sorted_idx = vals.sort_values(ascending=False).index
    total = vals.clip(lower=0).sum()
    if total <= 0:
        return pd.Series("C", index=vals.index)
    result = pd.Series("C", index=vals.index)
    cumsum = 0
    for idx in sorted_idx:
        v = vals.loc[idx]
        if v <= 0:
            continue
        cumsum += v
        pct = cumsum / total * 100
        if pct <= pct_a:
            result.loc[idx] = "A"
        elif pct <= pct_b:
            result.loc[idx] = "B"
    return result

articles["abc_revenue"] = _abc_for_column(articles["revenue"], pct_a, pct_b)
articles["abc_profit"] = _abc_for_column(articles["profit"], pct_a, pct_b)
articles["abc_sales"] = _abc_for_column(articles["sales_count"], pct_a, pct_b)
articles["abc_combined"] = articles["abc_revenue"] + articles["abc_profit"] + articles["abc_sales"]

# ── XYZ classification (demand stability via CV of daily orders) ─
# X: CV ≤ 10% (stable), Y: 10% < CV ≤ 25% (variable), Z: CV > 25% (erratic)
if not ord_df.empty:
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    _daily = ord_df.groupby(["nm_id", "order_date"]).agg(
        orders_count=("orders_count", "sum"),
    ).reset_index()
    xyz = (
        _daily.groupby("nm_id")
        .agg(mean=("orders_count", "mean"), std=("orders_count", "std"), n=("orders_count", "count"))
        .reset_index()
    )
    xyz["cv"] = np.where(
        (xyz["mean"] > 0) & (xyz["n"] >= 4),
        xyz["std"].fillna(0) / xyz["mean"] * 100,
        np.nan,
    )
    xyz["xyz"] = np.where(
        xyz["cv"].isna(), "—",
        np.where(xyz["cv"] <= 10, "X", np.where(xyz["cv"] <= 25, "Y", "Z")),
    )
    articles = articles.merge(xyz[["nm_id", "cv", "xyz"]], on="nm_id", how="left")
    articles["xyz"] = articles["xyz"].fillna("—")
else:
    articles["cv"] = np.nan
    articles["xyz"] = "—"
articles["abc_xyz"] = articles["abc_revenue"] + articles["xyz"]

# Selected metric for Pareto chart display
if metric_name == "Выручка":
    articles["total_value"] = articles["revenue"]
    articles["abc_category"] = articles["abc_revenue"]
    value_label = "Выручка"
    value_fmt = lambda v: format_currency(v)
elif metric_name == "Операционная прибыль":
    articles["total_value"] = articles["profit"]
    articles["abc_category"] = articles["abc_profit"]
    value_label = "Прибыль"
    value_fmt = lambda v: format_currency(v)
else:
    articles["total_value"] = articles["sales_count"]
    articles["abc_category"] = articles["abc_sales"]
    value_label = "Продажи, шт"
    value_fmt = lambda v: fmt_number(v)

# Sort by selected metric and compute shares
articles = articles.sort_values("total_value", ascending=False).reset_index(drop=True)
total_val = articles["total_value"].sum()
if total_val > 0:
    articles["value_share"] = (articles["total_value"] / total_val * 100).round(2)
else:
    articles["value_share"] = 0
articles["cumulative_share"] = articles["value_share"].cumsum().round(2)

# ── Helpers ──────────────────────────────────────────────────

ABC_COLORS = {"A": "#1e40af", "B": "#3b82f6", "C": "#93c5fd"}
ABC_BG = {"A": "#eff6ff", "B": "#f0f7ff", "C": "#f8fafc"}

# ── Summary KPIs (by selected metric) ───────────────────────
summary = articles.groupby("abc_category").agg(
    count=("nm_id", "count"),
    value=("total_value", "sum"),
).reset_index()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего артикулов", len(articles))
for col_obj, cat in zip([c2, c3, c4], ["A", "B", "C"]):
    row = summary[summary["abc_category"] == cat]
    cnt = int(row["count"].values[0]) if len(row) else 0
    val = float(row["value"].values[0]) if len(row) else 0
    pct = val / total_val * 100 if total_val else 0
    col_obj.metric(
        f"Категория {cat} ({cnt} шт.)",
        value_fmt(val),
        f"{pct:.0f}% {value_label.lower()}",
    )

# ── Combined group distribution ──────────────────────────────
st.markdown("### Комбинированные ABC-группы")
st.caption("Классификация одновременно по трём показателям: **Выручка** × **Прибыль** × **Продажи**")

comb_summary = (
    articles.groupby("abc_combined")
    .agg(count=("nm_id", "count"), revenue=("revenue", "sum"), profit=("profit", "sum"))
    .reset_index()
    .sort_values("count", ascending=False)
)

# Color palette for combined groups
_COMB_PALETTE = {
    "AAA": "#15803d", "AAB": "#16a34a", "AAC": "#22c55e",
    "ABA": "#1d4ed8", "ABB": "#3b82f6", "ABC": "#60a5fa",
    "ACA": "#7c3aed", "ACB": "#8b5cf6", "ACC": "#a78bfa",
    "BAA": "#b45309", "BAB": "#d97706", "BAC": "#f59e0b",
    "BBA": "#9333ea", "BBB": "#a855f7", "BBC": "#c084fc",
    "BCA": "#be185d", "BCB": "#db2777", "BCC": "#ec4899",
    "CAA": "#dc2626", "CAB": "#ef4444", "CAC": "#f87171",
    "CBA": "#78716c", "CBB": "#a8a29e", "CBC": "#d6d3d1",
    "CCA": "#71717a", "CCB": "#a1a1aa", "CCC": "#d4d4d8",
}

n_show = min(len(comb_summary), 9)
cols_comb = st.columns(n_show) if n_show > 0 else []
for i, (_, crow) in enumerate(comb_summary.head(n_show).iterrows()):
    grp = crow["abc_combined"]
    color = _COMB_PALETTE.get(grp, "#64748b")
    with cols_comb[i]:
        st.markdown(
            f'<div style="background:{color}12;border:1.5px solid {color};border-radius:10px;'
            f'padding:8px 4px;text-align:center;margin-bottom:8px">'
            f'<div style="font-size:18px;font-weight:800;color:{color}">{grp}</div>'
            f'<div style="font-size:20px;font-weight:700;color:#1e293b">{int(crow["count"])}</div>'
            f'<div style="font-size:10px;color:#64748b">артикулов</div></div>',
            unsafe_allow_html=True,
        )

# ── XYZ stability summary ───────────────────────────────────
st.markdown("### XYZ — стабильность спроса")
st.caption("X — стабильный спрос (CV ≤ 10%), Y — переменный (10–25%), Z — эпизодический (CV > 25%)")
_xyz_summary = articles.groupby("xyz").agg(
    count=("nm_id", "count"),
    revenue=("revenue", "sum"),
).reset_index()
_XYZ_COLORS_KPI = {"X": "#16a34a", "Y": "#f59e0b", "Z": "#dc2626", "—": "#94a3b8"}
_xcols = st.columns(max(len(_xyz_summary), 1))
for i, (_, xr) in enumerate(_xyz_summary.iterrows()):
    label = xr["xyz"]
    color = _XYZ_COLORS_KPI.get(label, "#94a3b8")
    with _xcols[i]:
        st.markdown(
            f'<div style="background:{color}12;border:1.5px solid {color};border-radius:10px;'
            f'padding:10px 6px;text-align:center;margin-bottom:8px">'
            f'<div style="font-size:18px;font-weight:800;color:{color}">{label}</div>'
            f'<div style="font-size:22px;font-weight:700;color:#1e293b">{int(xr["count"])}</div>'
            f'<div style="font-size:11px;color:#64748b">артикулов · {fmt_number(xr["revenue"])} ₽</div></div>',
            unsafe_allow_html=True,
        )

# ABC × XYZ matrix
with st.expander("Матрица ABC × XYZ", expanded=False):
    piv = (
        articles.pivot_table(
            index="abc_revenue", columns="xyz", values="nm_id",
            aggfunc="count", fill_value=0,
        )
        .reindex(index=["A", "B", "C"])
        .fillna(0)
        .astype(int)
    )
    if not piv.empty:
        st.dataframe(piv, width="stretch")

# ── Pareto chart ─────────────────────────────────────────────
st.markdown(f"### Кривая Парето ({value_label})")
top = articles.head(30).copy()
fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(
    x=top["supplier_article"],
    y=top["total_value"],
    name=value_label,
    marker_color=[ABC_COLORS.get(c, "#93c5fd") for c in top["abc_category"]],
    marker_opacity=0.85,
    text=top["total_value"].apply(lambda v: f"{v / 1000:,.0f}к" if abs(v) >= 1000 else f"{v:,.0f}"),
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>" + value_label + ": %{y:,.0f}<extra></extra>",
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
# Add boundary lines
fig_pareto.add_hline(y=pct_a, line_dash="dash", line_color="#16a34a", line_width=1,
                     annotation_text=f"A ({pct_a}%)", annotation_position="right",
                     yref="y2")
fig_pareto.add_hline(y=pct_b, line_dash="dash", line_color="#f59e0b", line_width=1,
                     annotation_text=f"B ({pct_b}%)", annotation_position="right",
                     yref="y2")

fig_pareto.update_layout(
    **PLOTLY_LAYOUT,
    yaxis=dict(title=value_label),
    yaxis2=dict(title="Нарастающий итог, %", overlaying="y", side="right", range=[0, 105]),
    xaxis_tickangle=-45, legend_title="",
    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
    margin=dict(t=40),
    bargap=0.25,
)
plotly_defaults(fig_pareto)
st.plotly_chart(fig_pareto, width="stretch")

# ── Pie chart ────────────────────────────────────────────────
col_pie, col_bar = st.columns(2)
with col_pie:
    st.markdown(f"### Доля {value_label.lower()}")
    fig_pie = px.pie(summary, names="abc_category", values="value",
                     color="abc_category", color_discrete_map=ABC_COLORS,
                     hole=0.4)
    fig_pie.update_traces(
        textinfo="label+percent",
        textfont_size=13,
        hovertemplate="<b>Категория %{label}</b><br>" + value_label + ": %{value:,.0f}<br>Доля: %{percent}<extra></extra>",
        marker=dict(line=dict(color="white", width=2)),
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT, margin=dict(t=10, b=10))
    plotly_defaults(fig_pie)
    st.plotly_chart(fig_pie, width="stretch")

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
    st.plotly_chart(fig_bar, width="stretch")

# ── Category filter ──────────────────────────────────────────
_fcat1, _fcat2 = st.columns(2)
with _fcat1:
    sel_abc = st.multiselect("Фильтр по основной метрике", ["A", "B", "C"], default=["A", "B", "C"])
with _fcat2:
    all_groups = sorted(articles["abc_combined"].unique())
    sel_groups = st.multiselect("Фильтр по комб. группе", all_groups, default=all_groups)

filtered = articles[
    articles["abc_category"].isin(sel_abc) &
    articles["abc_combined"].isin(sel_groups)
]

# ── HTML table with multi-dimensional ABC ───────────────────
TABLE_CSS = table_css("abc") + '<style>.abc .badge{padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;display:inline-block}.abc .comb-badge{padding:3px 10px;border-radius:8px;font-size:12px;font-weight:800;display:inline-block;letter-spacing:.5px}</style>'

st.markdown("### Детализация по артикулам")

hdr = (
    "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
    "<th>Выручка</th><th>ABC<br>Выр.</th>"
    "<th>Прибыль</th><th>ABC<br>Приб.</th>"
    "<th>Продажи</th><th>ABC<br>Прод.</th>"
    "<th>Группа</th><th>XYZ<br>(CV%)</th></tr>"
)
_XYZ_COLORS = {"X": "#16a34a", "Y": "#f59e0b", "Z": "#dc2626", "—": "#94a3b8"}


def _badge(cat):
    color = ABC_COLORS.get(cat, "#93c5fd")
    bg = ABC_BG.get(cat, "#f8fafc")
    return f'<span class="badge" style="background:{bg};color:{color};border:1px solid {color}40">{cat}</span>'


rows_html = ""
for idx, (_, r) in enumerate(filtered.iterrows(), 1):
    abc_r = r.get("abc_revenue", "C")
    abc_p = r.get("abc_profit", "C")
    abc_s = r.get("abc_sales", "C")
    combined = r.get("abc_combined", "CCC")
    comb_color = _COMB_PALETTE.get(combined, "#64748b")

    rev = float(r.get("revenue", 0))
    prof = float(r.get("profit", 0))
    scnt = float(r.get("sales_count", 0))
    prof_cls = "pos" if prof > 0 else ("neg" if prof < 0 else "")

    xyz = r.get("xyz", "—")
    cv = r.get("cv", np.nan)
    xyz_col = _XYZ_COLORS.get(xyz, "#94a3b8")
    cv_txt = f" ({cv:.0f}%)" if pd.notna(cv) else ""

    rows_html += (
        f"<tr>"
        f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
        f'<td style="font-weight:600">{wb_link(r.get("nm_id", 0), r.get("supplier_article", ""))}</td>'
        f'<td>{r.get("subject", "")}</td>'
        f'<td>{r.get("brand", "")}</td>'
        f'<td class="num">{fmt_number(rev)}</td>'
        f'<td class="ctr">{_badge(abc_r)}</td>'
        f'<td class="num {prof_cls}">{fmt_number(prof)}</td>'
        f'<td class="ctr">{_badge(abc_p)}</td>'
        f'<td class="num">{fmt_number(scnt)}</td>'
        f'<td class="ctr">{_badge(abc_s)}</td>'
        f'<td class="ctr"><span class="comb-badge" style="background:{comb_color}15;'
        f'color:{comb_color};border:1.5px solid {comb_color}">{combined}</span></td>'
        f'<td class="ctr"><span class="badge" style="background:{xyz_col}15;'
        f'color:{xyz_col};border:1px solid {xyz_col}55">{xyz}{cv_txt}</span></td>'
        f"</tr>"
    )

# Footer
tot_rev = filtered["revenue"].sum()
tot_prof = filtered["profit"].sum()
tot_sales = filtered["sales_count"].sum()
tot_pcls = "pos" if tot_prof > 0 else ("neg" if tot_prof < 0 else "")

ftr = (
    '<tr><td></td><td><b>Итого</b></td><td></td><td></td>'
    f'<td class="num"><b>{fmt_number(tot_rev)}</b></td><td></td>'
    f'<td class="num {tot_pcls}"><b>{fmt_number(tot_prof)}</b></td><td></td>'
    f'<td class="num"><b>{fmt_number(tot_sales)}</b></td><td></td>'
    f'<td></td><td></td></tr>'
)

html = (
    f'{TABLE_CSS}<div class="abc-wrap"><table class="abc" data-sortable>'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>{SORT_JS}'
)
render_table(html)
st.caption(f"Показано {len(filtered)} из {len(articles)} артикулов")

export_buttons(articles, "abc_analysis", sheet_name="ABC")
