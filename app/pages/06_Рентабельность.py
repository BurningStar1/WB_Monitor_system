"""Отчёт о прибыли — детализация с водопадной диаграммой и HTML-таблицей."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from marts import fetch_dataframe, FIN_PROFIT_QUERY, default_date_range
from styles import plotly_defaults, inject_global_styles, format_currency, format_pct, fmt_number, fmt_pct_tbl, table_css, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, wb_link, render_table, paginate, export_buttons
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("💰 Отчёт о прибыли")

with st.expander("ℹ️ Как считаем прибыль и маржу", expanded=False):
    st.markdown(
        """
        **Источник данных** — финансовые отчёты WB (`mart.finance_daily`),
        те же, что использует Raskка и отчёты о реализации.

        **Показатели (Raskка-совместимые):**
        - *Реализация до СПП* = `sales_amount − returns_amount` — цена товара
          без скидки постоянного покупателя (**база для маржи**)
        - *Реализация после СПП* = `retail_amount` — цена с учётом СПП
        - *К перечислению* = `ppvz_for_pay` — сумма, которую WB платит селлеру
        - *Услуги WB* = комиссия + логистика + хранение + штрафы + приёмка
          + эквайринг + удержания − доплаты
        - *Операционная прибыль* = К перечислению + Доплаты − Логистика
          − Хранение − Штрафы − Приёмка − Эквайринг − Удержания
          − Себестоимость − Налог
        - *Маржа %* = Прибыль / **Реализация до СПП** × 100

        Почему маржа считается от **до СПП**: это цена, по которой мы реально
        продаём товар. СПП — скидка WB за лояльность, в себестоимость она
        не входит, но на выручку давит. Измерять маржу в терминах ppvz_for_pay
        искусственно её завышает (числитель тот же, знаменатель меньше).

        **Цветовые индикаторы в таблице:**
        - 🟢 *Зелёный* — прибыль положительна (артикул зарабатывает)
        - 🔴 *Красный* — прибыль отрицательна (артикул теряет деньги)
        """
    )


# ── Filters: dates ───────────────────────────────────────────
d_from, d_to = date_filter_bar("profit", default_days=30)

params = {"d_from": str(d_from), "d_to": str(d_to)}
df = fetch_dataframe(FIN_PROFIT_QUERY, params)

if df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# Entity filters
categories = sorted(df["subject"].dropna().unique())
brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
_fe1, _fe2 = st.columns(2)
with _fe1:
    sel_cat = st.multiselect("Категория", categories, default=[])
with _fe2:
    sel_brands = st.multiselect("Бренд", brands, default=[])
if sel_cat:
    df = df[df["subject"].isin(sel_cat)]
if sel_brands:
    df = df[df["brand"].isin(sel_brands)]

if df.empty:
    st.warning("Нет данных по выбранным фильтрам. Измените параметры.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────

# Реализация до СПП — основа для маржи (Raskка-совместимо)
df["realization_pre_spp"] = df["sales_amount"] - df["returns_amount"]

total_realization = float(df["realization_pre_spp"].sum())   # до СПП
total_retail = float(df["retail_amount"].sum()) if "retail_amount" in df.columns else 0
total_rev = float(df["ppvz_for_pay"].sum())                   # к перечислению
total_cost = float(df["cost_amount"].sum())
total_comm = float(df["commission_amount"].sum())
total_logistics = float(df["logistics_amount"].sum())
total_storage = float(df["storage_amount"].sum())
total_penalty = float(df["penalty_amount"].sum())
total_wb_fees = float(df["total_wb_fees"].sum())
total_tax = float(df["tax_amount"].sum())
total_profit = float(df["profit"].sum())
margin = total_profit / total_realization * 100 if total_realization else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Реализация до СПП", format_currency(total_realization))
c2.metric("К перечислению", format_currency(total_rev))
c3.metric("Услуги WB", format_currency(total_wb_fees))
c4.metric("Себестоимость", format_currency(total_cost))
c5.metric("Прибыль", format_currency(total_profit))
c6.metric("Маржа (от до СПП)", format_pct(margin))

# ── Waterfall chart ──────────────────────────────────────────

st.markdown("### Структура финансового результата")

total_acceptance = float(df["acceptance_amount"].sum()) if "acceptance_amount" in df.columns else 0
total_acquiring = float(df["acquiring_amount"].sum()) if "acquiring_amount" in df.columns else 0
total_deduction = float(df["deduction_amount"].sum()) if "deduction_amount" in df.columns else 0
total_additional = float(df["additional_payment_amount"].sum()) if "additional_payment_amount" in df.columns else 0
# СПП = Реализация до СПП − Реализация после СПП (скидка постоянного покупателя)
total_spp = max(total_realization - total_retail, 0) if total_retail else 0
# Спред "после СПП → к перечислению" = комиссия WB, уже учтённая в retail_amount
_post_spp_to_payout = max(total_retail - total_rev, 0) if total_retail else total_comm

_wf_labels = ["Реализация до СПП", "СПП", "Комиссия WB",
              "Логистика", "Хранение",
              "Штрафы", "Приёмка", "Эквайринг", "Удержания",
              "Допл. за доставку", "Себестоимость", "Налоги", "Прибыль"]
_wf_values = [total_realization, -total_spp, -_post_spp_to_payout,
              -total_logistics, -total_storage,
              -total_penalty, -total_acceptance, -total_acquiring, -total_deduction,
              total_additional, -total_cost, -total_tax, total_profit]
_wf_texts = [fmt_number(abs(v)) for v in _wf_values]

fig_wf = go.Figure(go.Waterfall(
    x=_wf_labels,
    y=_wf_values,
    measure=["absolute", "relative", "relative",
             "relative", "relative", "relative", "relative",
             "relative", "relative", "relative", "relative",
             "relative", "total"],
    connector=dict(line=dict(color="#cbd5e1", width=1, dash="dash")),
    increasing_marker=dict(color=PLOTLY_COLORS["blue"],
                           line=dict(color="white", width=1.5)),
    decreasing_marker=dict(color=PLOTLY_COLORS["rose"],
                           line=dict(color="white", width=1.5)),
    totals_marker=dict(color=PLOTLY_COLORS["blue_dark"],
                       line=dict(color="white", width=1.5)),
    text=_wf_texts,
    textposition="outside",
    textfont=dict(size=11, color="#334155", family="Inter, system-ui, sans-serif"),
    hovertemplate="<b>%{x}</b><br>%{y:,.0f} \u20bd<extra></extra>",
))
fig_wf.update_layout(
    **PLOTLY_LAYOUT,
    yaxis_title="Сумма, \u20bd", showlegend=False,
    margin=dict(l=10, r=10, t=30, b=10),
    bargap=0.25,
)
plotly_defaults(fig_wf)
st.plotly_chart(fig_wf, width="stretch")

# ── Daily profit trend ───────────────────────────────────────

st.markdown("### Динамика прибыли по дням")
daily = df.groupby("report_date").agg(
    ppvz_for_pay=("ppvz_for_pay", "sum"),
    profit=("profit", "sum"),
    cost_amount=("cost_amount", "sum"),
).reset_index().sort_values("report_date")

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(
    x=daily["report_date"], y=daily["ppvz_for_pay"],
    name="\u041a \u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u044e",
    marker_color="rgba(59,130,246,0.25)",
    hovertemplate="<b>%{x|%d.%m}</b><br>\u041a \u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b.: %{y:,.0f} \u20bd<extra></extra>",
))
fig_trend.add_trace(go.Bar(
    x=daily["report_date"], y=daily["profit"],
    name="\u041f\u0440\u0438\u0431\u044b\u043b\u044c",
    marker=dict(color=PLOTLY_COLORS["green"], opacity=0.75),
    hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u0438\u0431\u044b\u043b\u044c: %{y:,.0f} \u20bd<extra></extra>",
))
fig_trend.add_trace(go.Scatter(
    x=daily["report_date"], y=daily["profit"],
    name="\u0422\u0440\u0435\u043d\u0434 \u043f\u0440\u0438\u0431\u044b\u043b\u0438",
    mode="lines+markers",
    line=dict(color=PLOTLY_COLORS["purple"], width=2.5, shape="spline"),
    marker=dict(color=PLOTLY_COLORS["purple"], size=7,
                line=dict(color="white", width=1.5)),
    fill="tozeroy",
    fillcolor="rgba(139,92,246,0.10)",
    hovertemplate="<b>%{x|%d.%m}</b><br>\u0422\u0440\u0435\u043d\u0434: %{y:,.0f} \u20bd<extra></extra>",
))
fig_trend.update_layout(
    **PLOTLY_LAYOUT,
    barmode="overlay",
    bargap=0.25,
    xaxis_title="",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    margin=dict(l=10, r=10, t=40, b=10),
)
plotly_defaults(fig_trend)
st.plotly_chart(fig_trend, width="stretch")

# ── Profit by article (aggregated) ──────────────────────────

st.markdown("### Прибыль по артикулам")

_agg_dict = {
    "subject": ("subject", "first"),
    "brand": ("brand", "first"),
    "sales_count": ("sales_count", "sum"),
    "returns_count": ("returns_count", "sum"),
    "sales_amount": ("sales_amount", "sum"),
    "returns_amount": ("returns_amount", "sum"),
    "realization_pre_spp": ("realization_pre_spp", "sum"),
    "ppvz_for_pay": ("ppvz_for_pay", "sum"),
    "commission_amount": ("commission_amount", "sum"),
    "logistics_amount": ("logistics_amount", "sum"),
    "storage_amount": ("storage_amount", "sum"),
    "penalty_amount": ("penalty_amount", "sum"),
    "cost_amount": ("cost_amount", "sum"),
    "profit": ("profit", "sum"),
}
art = df.groupby(["nm_id", "supplier_article"]).agg(**_agg_dict).reset_index()
# Маржа считается от Реализации до СПП (Raskка-совместимо)
art["margin_pct"] = np.where(
    art["realization_pre_spp"] > 0,
    (art["profit"] / art["realization_pre_spp"] * 100).round(1),
    0,
)
art = art.sort_values("profit", ascending=False).reset_index(drop=True)

TABLE_CSS = table_css("prf")

hdr = (
    "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
    "<th>Продажи</th><th>Возвраты</th><th>Реализ. до&nbsp;СПП</th>"
    "<th>К&nbsp;перечисл.</th>"
    "<th>Логистика</th><th>Хранение</th><th>Штрафы</th>"
    "<th>Себест.</th><th>Прибыль</th><th>Маржа</th></tr>"
)

display, _start, _end, _total = paginate(art, "prf_art", default_size=50)
rows_html = ""
for _i, (_, r) in enumerate(display.iterrows()):
    idx = _start + _i + 1
    profit = float(r["profit"])
    pcls = "pos" if profit > 0 else ("neg" if profit < 0 else "")
    m = float(r["margin_pct"])
    mcls = "pos" if m > 0 else ("neg" if m < 0 else "")

    rows_html += (
        f"<tr>"
        f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
        f'<td style="font-weight:600">{wb_link(r["nm_id"], r["supplier_article"])}</td>'
        f'<td>{r["subject"]}</td>'
        f'<td>{r.get("brand", "")}</td>'
        f'<td class="num">{int(r["sales_count"])}</td>'
        f'<td class="num">{int(r["returns_count"])}</td>'
        f'<td class="num">{fmt_number(r["realization_pre_spp"])}</td>'
        f'<td class="num">{fmt_number(r["ppvz_for_pay"])}</td>'
        f'<td class="num">{fmt_number(r["logistics_amount"])}</td>'
        f'<td class="num">{fmt_number(r["storage_amount"])}</td>'
        f'<td class="num">{fmt_number(r["penalty_amount"])}</td>'
        f'<td class="num">{fmt_number(r["cost_amount"])}</td>'
        f'<td class="num {pcls}">{fmt_number(profit)}</td>'
        f'<td class="ctr {mcls}">{fmt_pct_tbl(m)}</td>'
        f"</tr>"
    )

# Footer
ftr_profit = art["profit"].sum()
ftr_realization = art["realization_pre_spp"].sum()
ftr_cls = "pos" if ftr_profit > 0 else ("neg" if ftr_profit < 0 else "")
ftr_margin = ftr_profit / ftr_realization * 100 if ftr_realization else 0
ftr_mcls = "pos" if ftr_margin > 0 else ("neg" if ftr_margin < 0 else "")
ftr = (
    f'<tr><td></td><td><b>Итого</b></td><td></td><td></td>'
    f'<td class="num">{int(art["sales_count"].sum())}</td>'
    f'<td class="num">{int(art["returns_count"].sum())}</td>'
    f'<td class="num">{fmt_number(ftr_realization)}</td>'
    f'<td class="num">{fmt_number(art["ppvz_for_pay"].sum())}</td>'
    f'<td class="num">{fmt_number(art["logistics_amount"].sum())}</td>'
    f'<td class="num">{fmt_number(art["storage_amount"].sum())}</td>'
    f'<td class="num">{fmt_number(art["penalty_amount"].sum())}</td>'
    f'<td class="num">{fmt_number(art["cost_amount"].sum())}</td>'
    f'<td class="num {ftr_cls}">{fmt_number(ftr_profit)}</td>'
    f'<td class="ctr {ftr_mcls}">{fmt_pct_tbl(ftr_margin)}</td>'
    f'</tr>'
)

html = (
    f'{TABLE_CSS}<div class="prf-wrap"><table class="prf" data-sortable>'
    f'<thead>{hdr}</thead><tbody>{rows_html}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>{SORT_JS}'
)
render_table(html)

export_buttons(df, "profit_report", sheet_name="Profit")
