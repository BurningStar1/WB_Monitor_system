"""Еженедельный отчёт — выручка, прибыль, маржа по неделям."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from marts import fetch_dataframe, FIN_WEEKLY_QUERY, default_date_range
from styles import plotly_defaults, inject_global_styles, fmt_number, table_css, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, render_table, export_buttons
from auth import check_auth, logout

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📅 Еженедельный отчёт")

# ── Filters: dates ───────────────────────────────────────────
d_from, d_to = date_filter_bar("weekly", default_days=90)

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(FIN_WEEKLY_QUERY, params)

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


with st.expander("ℹ️ Как считаем", expanded=False):
    st.markdown(
        """
        **Источник** — финансовые отчёты WB (`mart.finance_daily`, сгруппированы по ISO-неделям).

        **Показатели** (в стиле отчётов Raskка):
        - *Реализация до СПП* = `sales_amount − returns_amount` (цена товара без скидки постоянного покупателя)
        - *Реализация после СПП* = `retail_amount` (цена после СПП)
        - *К перечислению* = `ppvz_for_pay`
        - *Услуги WB* = комиссия + логистика + хранение + штрафы + приёмка + эквайринг + удержания − доп. выплаты
        - *Прибыль* = К перечислению − услуги WB − себестоимость − налог
        - *Маржа %* = прибыль / **Реализация до СПП** (Raskка-совместимо)
        """
    )

# ── KPI cards ────────────────────────────────────────────────
total_sales = int(df["sales_count"].sum())
total_returns = int(df["returns_count"].sum())
total_realization = df["realization_pre_spp"].sum()     # Реализация до СПП (Raskка)
total_payout = df["ppvz_for_pay"].sum()                  # К перечислению
total_profit = df["profit"].sum()

total_margin = round(total_profit / total_realization * 100, 1) if total_realization else 0
return_rate = round(total_returns / total_sales * 100, 1) if total_sales else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Продажи", f"{total_sales:,}".replace(",", " "))
c2.metric("Возвраты", f"{total_returns:,}".replace(",", " "), f"{return_rate}%")
c3.metric("Реализация до СПП", f"{total_realization:,.0f} ₽".replace(",", " "))
c4.metric("К перечислению", f"{total_payout:,.0f} ₽".replace(",", " "))
c5.metric("Прибыль", f"{total_profit:,.0f} ₽".replace(",", " "))
c6.metric("Маржа", f"{total_margin:.1f}%")

# ── Sort ascending for charts & delta calc ───────────────────
df = df.sort_values("year_week", ascending=True).reset_index(drop=True)
# Маржа считается от реализации ДО СПП (Raskка-совместимо).
df["margin_pct"] = (
    df["profit"] / df["realization_pre_spp"].replace(0, pd.NA) * 100
).fillna(0).round(1)

# ── Chart 1: Realization + Profit bars, Margin line ─────────
st.markdown("### Реализация, прибыль и маржа по неделям")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(
        x=df["year_week"].astype(str), y=df["realization_pre_spp"],
        name="Реализация до СПП",
        marker=dict(
            color=PLOTLY_COLORS["blue"],
            line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5),
        ),
        opacity=0.88,
        hovertemplate="Реализация до СПП: %{y:,.0f} ₽<extra></extra>",
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Bar(
        x=df["year_week"].astype(str), y=df["profit"],
        name="Прибыль",
        marker=dict(
            color=PLOTLY_COLORS["blue_dark"],
            line=dict(color="#172554", width=0.5),
        ),
        opacity=0.88,
        hovertemplate="Прибыль: %{y:,.0f} ₽<extra></extra>",
    ),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=df["year_week"].astype(str), y=df["margin_pct"],
        name="Маржа %", mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["amber"], width=2.5, shape="spline"),
        marker=dict(
            size=7, color=PLOTLY_COLORS["amber"],
            line=dict(color="white", width=1.5),
        ),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.08)",
        hovertemplate="Маржа: %{y:.1f}%<extra></extra>",
    ),
    secondary_y=True,
)
fig.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group",
    bargap=0.25,
    bargroupgap=0.1,
    height=420,
)
fig.update_yaxes(
    title_text="Сумма, ₽", secondary_y=False, tickformat=",",
)
fig.update_yaxes(
    title_text="Маржа, %", secondary_y=True,
    tickfont=dict(color=PLOTLY_COLORS["amber"]),
    title_font=dict(color=PLOTLY_COLORS["amber"]),
)

plotly_defaults(fig)
st.plotly_chart(fig, width="stretch")

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
    "<th>Продажи</th><th>Δ%</th>"
    "<th>Возвраты</th><th>%&nbsp;возвр.</th>"
    "<th>Реализ.&nbsp;до&nbsp;СПП</th><th>Δ%</th>"
    "<th>К&nbsp;перечисл.</th>"
    "<th>Логистика</th>"
    "<th>Хранение</th>"
    "<th>Комиссия</th>"
    "<th>Себестоимость</th>"
    "<th>Прибыль</th><th>Δ%</th>"
    "<th>Маржа%</th>"
    "</tr>"
)

rows_html = []
for i, row in df.iterrows():
    prev = df.iloc[i - 1] if i > 0 else None

    lbl = _week_label(row)
    margin = row["margin_pct"]

    d_sales, c_sales = _delta(row["sales_count"], prev["sales_count"]) if prev is not None else ("", "")
    d_rev, c_rev = _delta(row["realization_pre_spp"], prev["realization_pre_spp"]) if prev is not None else ("", "")
    d_prof, c_prof = _delta(row["profit"], prev["profit"]) if prev is not None else ("", "")

    def _badge(val, cls):
        if not val:
            return '<td class="ctr">—</td>'
        return f'<td class="ctr"><span class="delta {cls}">{val}</span></td>'

    _ret_pct = round(row["returns_count"] / row["sales_count"] * 100, 1) if row["sales_count"] else 0

    rows_html.append(
        f"<tr>"
        f'<td style="font-weight:600">{lbl}</td>'
        f'<td class="num">{fmt_number(row["sales_count"])}</td>{_badge(d_sales, c_sales)}'
        f'<td class="num">{fmt_number(row["returns_count"])}</td>'
        f'<td class="ctr">{_ret_pct:.1f}%</td>'
        f'<td class="num">{fmt_number(row["realization_pre_spp"])}</td>{_badge(d_rev, c_rev)}'
        f'<td class="num">{fmt_number(row["ppvz_for_pay"])}</td>'
        f'<td class="num">{fmt_number(row["logistics"])}</td>'
        f'<td class="num">{fmt_number(row["storage"])}</td>'
        f'<td class="num">{fmt_number(row["commission"])}</td>'
        f'<td class="num">{fmt_number(row["cost_amount"])}</td>'
        f'<td class="num">{fmt_number(row["profit"])}</td>{_badge(d_prof, c_prof)}'
        f'<td class="ctr">{margin:.1f}%</td>'
        f"</tr>"
    )

# Totals footer
t_sales = int(df["sales_count"].sum())
t_returns = int(df["returns_count"].sum())
t_realization = df["realization_pre_spp"].sum()
t_payout = df["ppvz_for_pay"].sum()
t_logistics = df["logistics"].sum()
t_storage = df["storage"].sum()
t_comm = df["commission"].sum()
t_cost = df["cost_amount"].sum()
t_profit = df["profit"].sum()
t_margin = (t_profit / t_realization * 100) if t_realization else 0

footer = (
    "<tr>"
    f'<td>Итого</td>'
    f'<td class="num">{fmt_number(t_sales)}</td><td></td>'
    f'<td class="num">{fmt_number(t_returns)}</td>'
    f'<td class="ctr">{round(t_returns / t_sales * 100, 1) if t_sales else 0:.1f}%</td>'
    f'<td class="num">{fmt_number(t_realization)}</td><td></td>'
    f'<td class="num">{fmt_number(t_payout)}</td>'
    f'<td class="num">{fmt_number(t_logistics)}</td>'
    f'<td class="num">{fmt_number(t_storage)}</td>'
    f'<td class="num">{fmt_number(t_comm)}</td>'
    f'<td class="num">{fmt_number(t_cost)}</td>'
    f'<td class="num">{fmt_number(t_profit)}</td><td></td>'
    f'<td class="ctr">{t_margin:.1f}%</td>'
    "</tr>"
)

table_html = (
    TABLE_CSS
    + '<div class="wk-wrap"><table class="wk" data-sortable>'
    + f"<thead>{header}</thead>"
    + "<tbody>" + "\n".join(reversed(rows_html)) + "</tbody>"
    + f"<tfoot>{footer}</tfoot>"
    + f"</table></div>{SORT_JS}"
)

render_table(table_html)

# ── Chart 2: Sales + Returns stacked bar ─────────────────────
st.markdown("### Продажи и возвраты по неделям")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=df["year_week"].astype(str), y=df["sales_count"],
    name="Продажи",
    marker=dict(
        color=PLOTLY_COLORS["blue"],
        line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5),
    ),
    hovertemplate="Продажи: %{y:,.0f} шт.<extra></extra>",
))
fig2.add_trace(go.Bar(
    x=df["year_week"].astype(str), y=df["returns_count"],
    name="Возвраты",
    marker=dict(
        color=PLOTLY_COLORS["rose"],
        line=dict(color="#be123c", width=0.5),
    ),
    hovertemplate="Возвраты: %{y:,.0f} шт.<extra></extra>",
))
fig2.update_layout(
    **PLOTLY_LAYOUT,
    barmode="stack",
    xaxis_title="Неделя", yaxis_title="Количество",
    bargap=0.25,
    height=400,
)
plotly_defaults(fig2)
st.plotly_chart(fig2, width="stretch")

# ── Export ────────────────────────────────────────────────────
export_buttons(df, "weekly_report", sheet_name="Weekly")
