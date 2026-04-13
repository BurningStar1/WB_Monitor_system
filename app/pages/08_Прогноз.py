"""Прогноз — динамика заказов/прибыли и прогнозные метрики по артикулам."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from marts import fetch_dataframe, FORECAST_DAILY_QUERY, FORECAST_ARTICLE_QUERY
from styles import plotly_defaults,  inject_global_styles, fmt_number, fmt_pct_tbl, PLOTLY_LAYOUT, PLOTLY_COLORS
from auth import check_auth, logout

# ── Page setup ────────────────────────────────────────────────

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("\U0001f4c8 Прогноз")

# ── Filters ──────────────────────────────────────────
_fc1, _fc2, _fc3 = st.columns(3)
with _fc1:
    d_to = st.date_input("Дата окончания", value=date.today())
with _fc2:
    d_from = st.date_input("Дата начала", value=d_to - timedelta(days=64))
with _fc3:
    _horizon_options = {"7 дней": 7, "14 дней": 14, "30 дней": 30}
    _horizon_label = st.selectbox(
        "Горизонт прогноза", list(_horizon_options.keys()), index=0,
    )
    forecast_horizon = _horizon_options[_horizon_label]

params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Helpers ───────────────────────────────────────────────────




# ── Shared CSS for HTML tables ────────────────────────────────

TABLE_CSS = """
<style>
.art-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin-bottom:1rem;border:1px solid #e2e8f0}
.art-t{border-collapse:collapse;width:max-content;min-width:100%;
  font-size:12px;font-family:Inter,system-ui,sans-serif;background:#fff;color:#1e293b}
.art-t thead th{background:#f1f5f9;position:sticky;top:0;z-index:3;
  padding:6px 6px;border-bottom:2px solid #cbd5e1;border-right:1px solid #e2e8f0;
  font-weight:600;font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.3px;
  text-align:center;white-space:nowrap;vertical-align:bottom}
.art-t thead th:last-child{border-right:none}
.art-t td{border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  padding:4px 6px;white-space:nowrap;vertical-align:middle}
.art-t td:last-child{border-right:none}
.art-t tbody tr:nth-child(even){background:#fafbfc}
.art-t tbody tr:hover{background:#eef2ff}
.art-t .num{text-align:right}.art-t .ctr{text-align:center}
.art-t .rn{color:#94a3b8;font-size:11px;text-align:center;min-width:24px}
.art-t .pos{color:#16a34a;font-weight:700}.art-t .neg{color:#dc2626;font-weight:700}
.art-t .stock-ok{color:#16a34a;font-weight:700}
.art-t .stock-warn{color:#ca8a04;font-weight:700}
.art-t .stock-crit{color:#dc2626;font-weight:700}
.art-t tfoot td{background:#f1f5f9;font-weight:700;font-size:12px;
  border-top:2px solid #cbd5e1;padding:7px 8px;color:#1e293b}
</style>
"""

# ── Tabs ──────────────────────────────────────────────────────

tab_daily, tab_articles = st.tabs(["По дням", "По артикулам"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — По дням
# ══════════════════════════════════════════════════════════════

with tab_daily:
    df = fetch_dataframe(FORECAST_DAILY_QUERY, params)

    if df.empty:
        st.info("Нет данных за выбранный период")
        st.stop()

    df["order_date"] = pd.to_datetime(df["order_date"])

    # Ensure numeric columns
    _num_cols = [
        "orders_count", "orders_amount", "sales_count", "net_revenue",
        "gross_revenue", "commission_amount", "cost_amount", "profit_amount",
        "ma_orders_7d", "ma_orders_14d", "ma_revenue_7d", "ma_profit_7d",
    ]
    for c in _num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    n_days = len(df)

    # ── Forecast projection (linear extrapolation on last 14 MA7 points) ──

    last_date = df["order_date"].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_horizon,
        freq="D",
    )

    # -- Orders forecast --
    ma_orders_vals = df["ma_orders_7d"].dropna()
    _fit_n = min(14, len(ma_orders_vals))
    if _fit_n >= 2:
        _y_ord = ma_orders_vals.iloc[-_fit_n:].values.astype(float)
        _x_ord = np.arange(_fit_n)
        _coef_ord = np.polyfit(_x_ord, _y_ord, 1)
        _fc_x_ord = np.arange(_fit_n, _fit_n + forecast_horizon)
        forecast_orders_vals = np.polyval(_coef_ord, _fc_x_ord)
        forecast_orders_vals = np.clip(forecast_orders_vals, 0, None)
    else:
        forecast_orders_vals = np.full(forecast_horizon, 0.0)

    # -- Profit forecast --
    ma_profit_vals = df["ma_profit_7d"].dropna()
    _fit_n_p = min(14, len(ma_profit_vals))
    if _fit_n_p >= 2:
        _y_prf = ma_profit_vals.iloc[-_fit_n_p:].values.astype(float)
        _x_prf = np.arange(_fit_n_p)
        _coef_prf = np.polyfit(_x_prf, _y_prf, 1)
        _fc_x_prf = np.arange(_fit_n_p, _fit_n_p + forecast_horizon)
        forecast_profit_vals = np.polyval(_coef_prf, _fc_x_prf)
    else:
        forecast_profit_vals = np.full(forecast_horizon, 0.0)

    # Prepend last historical point for visual continuity
    fc_dates_full = pd.Index([last_date]).append(forecast_dates)
    fc_orders_full = np.concatenate([[float(ma_orders_vals.iloc[-1])], forecast_orders_vals])
    fc_profit_full = np.concatenate([[float(ma_profit_vals.iloc[-1])], forecast_profit_vals])

    # ── Summary KPIs ──────────────────────────────────────────

    total_orders = int(df["orders_count"].sum())
    avg_orders_day = df["orders_count"].mean()
    forecast_orders_sum = int(round(forecast_orders_vals.sum()))
    forecast_profit_sum = int(round(forecast_profit_vals.sum()))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Заказы за период", fmt_number(total_orders) or "0")
    k2.metric("Ср. заказов/день", f"{avg_orders_day:.1f}")
    k3.metric(f"Прогноз заказов {forecast_horizon}д", fmt_number(forecast_orders_sum) or "0")
    k4.metric(f"Прогноз прибыли {forecast_horizon}д", f"{fmt_number(forecast_profit_sum)} \u20bd" if forecast_profit_sum else "0 \u20bd")

    # ── Chart 1: Orders + moving averages ─────────────────────

    st.markdown("### Динамика заказов")

    fig_orders = go.Figure()

    fig_orders.add_trace(go.Bar(
        x=df["order_date"], y=df["orders_count"],
        name="\u0417\u0430\u043a\u0430\u0437\u044b, \u0448\u0442",
        marker_color="rgba(59,130,246,0.20)",
        hovertemplate="<b>%{x|%d.%m}</b><br>\u0417\u0430\u043a\u0430\u0437\u044b: %{y:.0f} \u0448\u0442<extra></extra>",
    ))

    fig_orders.add_trace(go.Scatter(
        x=df["order_date"], y=df["ma_orders_7d"],
        name="MA 7\u0434",
        mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["blue"], width=2.5, shape="spline"),
        marker=dict(color=PLOTLY_COLORS["blue"], size=7,
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%d.%m}</b><br>MA 7\u0434: %{y:.1f} \u0448\u0442<extra></extra>",
    ))

    fig_orders.add_trace(go.Scatter(
        x=df["order_date"], y=df["ma_orders_14d"],
        name="MA 14\u0434",
        mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["amber"], width=2, dash="dash", shape="spline"),
        marker=dict(color=PLOTLY_COLORS["amber"], size=6,
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.06)",
        hovertemplate="<b>%{x|%d.%m}</b><br>MA 14\u0434: %{y:.1f} \u0448\u0442<extra></extra>",
    ))

    # ── Forecast projection trace (orders) ────────────────────
    fig_orders.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_orders_full,
        name="\u041f\u0440\u043e\u0433\u043d\u043e\u0437",
        mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["blue_light"], width=2.5, dash="dash", shape="spline"),
        marker=dict(color=PLOTLY_COLORS["blue_light"], size=6,
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(147,197,253,0.10)",
        hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u043e\u0433\u043d\u043e\u0437: %{y:.1f} \u0448\u0442<extra></extra>",
    ))

    # Vertical dashed line marking forecast start
    fig_orders.add_vline(
        x=last_date,
        line_width=1.5, line_dash="dash", line_color=PLOTLY_COLORS["slate"],
        annotation_text="\u041d\u0430\u0447\u0430\u043b\u043e \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=PLOTLY_COLORS["slate"],
    )

    fig_orders.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="\u0417\u0430\u043a\u0430\u0437\u044b, \u0448\u0442",
        xaxis_title="",
        barmode="overlay",
        bargap=0.25,
    )

    plotly_defaults(fig_orders)
    st.plotly_chart(fig_orders, use_container_width=True)

    # ── Chart 2: Profit + MA ──────────────────────────────────

    st.markdown("### Динамика прибыли")

    # Separate positive/negative for coloring (green / rose)
    profit_colors = [
        PLOTLY_COLORS["green"] if v >= 0 else PLOTLY_COLORS["rose"]
        for v in df["profit_amount"]
    ]

    fig_profit = go.Figure()

    fig_profit.add_trace(go.Bar(
        x=df["order_date"], y=df["profit_amount"],
        name="\u041f\u0440\u0438\u0431\u044b\u043b\u044c",
        marker=dict(color=profit_colors,
                    line=dict(color="white", width=0.5)),
        hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u0438\u0431\u044b\u043b\u044c: %{y:,.0f} \u20bd<extra></extra>",
    ))

    fig_profit.add_trace(go.Scatter(
        x=df["order_date"], y=df["ma_profit_7d"],
        name="MA \u043f\u0440\u0438\u0431\u044b\u043b\u0438 7\u0434",
        mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["purple"], width=2.5, shape="spline"),
        marker=dict(color=PLOTLY_COLORS["purple"], size=7,
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.08)",
        hovertemplate="<b>%{x|%d.%m}</b><br>MA 7\u0434: %{y:,.0f} \u20bd<extra></extra>",
    ))

    # ── Forecast projection trace (profit) ────────────────────
    fig_profit.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_profit_full,
        name="\u041f\u0440\u043e\u0433\u043d\u043e\u0437",
        mode="lines+markers",
        line=dict(color="#c4b5fd", width=2.5, dash="dash", shape="spline"),
        marker=dict(color="#c4b5fd", size=6,
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(196,181,253,0.10)",
        hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u043e\u0433\u043d\u043e\u0437: %{y:,.0f} \u20bd<extra></extra>",
    ))

    # Vertical dashed line marking forecast start
    fig_profit.add_vline(
        x=last_date,
        line_width=1.5, line_dash="dash", line_color=PLOTLY_COLORS["slate"],
        annotation_text="\u041d\u0430\u0447\u0430\u043b\u043e \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=PLOTLY_COLORS["slate"],
    )

    fig_profit.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="\u041f\u0440\u0438\u0431\u044b\u043b\u044c, \u20bd",
        xaxis_title="",
        barmode="overlay",
        bargap=0.25,
    )

    plotly_defaults(fig_profit)
    st.plotly_chart(fig_profit, use_container_width=True)

    # ── Daily HTML table ──────────────────────────────────────

    st.markdown("### Детализация по дням")

    # Pagination
    c_pg1, c_pg2, c_pg3 = st.columns([1, 1, 4])
    with c_pg1:
        page_size_d = st.selectbox(
            "Строк", [15, 30, 50, 100], index=0,
            key="pg_size_daily", label_visibility="collapsed",
        )
    total_rows_d = len(df)
    total_pages_d = max((total_rows_d - 1) // page_size_d + 1, 1)
    with c_pg2:
        page_d = st.number_input(
            "Стр.", min_value=1, max_value=total_pages_d, value=1,
            key="pg_num_daily", label_visibility="collapsed",
        )
    s_d = (page_d - 1) * page_size_d
    e_d = min(s_d + page_size_d, total_rows_d)

    display_d = df.iloc[s_d:e_d]

    # Header
    hdr = (
        "<tr>"
        "<th>#</th><th>Дата</th><th>Заказы шт</th><th>Заказы \u20bd</th>"
        "<th>Продажи шт</th><th>Выручка</th><th>Комиссия</th>"
        "<th>Себестоимость</th><th>Прибыль</th>"
        "<th>MA 7д<br>(заказы)</th><th>MA 14д<br>(заказы)</th><th>Маржа%</th>"
        "</tr>"
    )

    rows = ""
    for idx, (_, row) in enumerate(display_d.iterrows(), start=s_d + 1):
        d_str = row["order_date"].strftime("%d.%m.%Y")
        profit = float(row["profit_amount"])
        revenue = float(row["net_revenue"])
        margin = (profit / revenue * 100) if revenue else 0
        pcls = "pos" if profit > 0 else ("neg" if profit < 0 else "")
        mcls = "pos" if margin > 0 else ("neg" if margin < 0 else "")

        tr = "<tr>"
        tr += f'<td class="rn">{idx}</td>'
        tr += f'<td class="ctr">{d_str}</td>'
        tr += f'<td class="num">{fmt_number(row["orders_count"])}</td>'
        tr += f'<td class="num">{fmt_number(row["orders_amount"])}</td>'
        tr += f'<td class="num">{fmt_number(row["sales_count"])}</td>'
        tr += f'<td class="num">{fmt_number(revenue)}</td>'
        tr += f'<td class="num">{fmt_number(row["commission_amount"])}</td>'
        tr += f'<td class="num">{fmt_number(row["cost_amount"])}</td>'
        tr += f'<td class="num {pcls}">{fmt_number(profit)}</td>'
        tr += f'<td class="num">{row["ma_orders_7d"]:.1f}</td>'
        tr += f'<td class="num">{row["ma_orders_14d"]:.1f}</td>'
        tr += f'<td class="ctr {mcls}">{fmt_pct_tbl(margin)}</td>'
        tr += "</tr>"
        rows += tr

    # Totals
    t_orders = int(display_d["orders_count"].sum())
    t_orders_amt = display_d["orders_amount"].sum()
    t_sales = int(display_d["sales_count"].sum())
    t_rev = display_d["net_revenue"].sum()
    t_comm = display_d["commission_amount"].sum()
    t_cost = display_d["cost_amount"].sum()
    t_profit = display_d["profit_amount"].sum()
    t_margin = (t_profit / t_rev * 100) if t_rev else 0
    t_pcls = "pos" if t_profit > 0 else ("neg" if t_profit < 0 else "")
    t_mcls = "pos" if t_margin > 0 else ("neg" if t_margin < 0 else "")

    ftr = (
        "<tr>"
        '<td></td><td><b>Итого</b></td>'
        f'<td class="num">{fmt_number(t_orders)}</td>'
        f'<td class="num">{fmt_number(t_orders_amt)}</td>'
        f'<td class="num">{fmt_number(t_sales)}</td>'
        f'<td class="num">{fmt_number(t_rev)}</td>'
        f'<td class="num">{fmt_number(t_comm)}</td>'
        f'<td class="num">{fmt_number(t_cost)}</td>'
        f'<td class="num {t_pcls}">{fmt_number(t_profit)}</td>'
        '<td></td><td></td>'
        f'<td class="ctr {t_mcls}">{fmt_pct_tbl(t_margin)}</td>'
        "</tr>"
    )

    html_daily = (
        f'{TABLE_CSS}<div class="art-wrap"><table class="art-t">'
        f'<thead>{hdr}</thead><tbody>{rows}</tbody>'
        f'<tfoot>{ftr}</tfoot></table></div>'
    )

    st.markdown(html_daily, unsafe_allow_html=True)
    st.caption(f"Показано {s_d + 1}\u2013{e_d} из {total_rows_d}")

    # CSV export
    export_d = df.copy()
    export_d["order_date"] = export_d["order_date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "\U0001f4e5 Скачать CSV (по дням)",
        export_d.to_csv(index=False).encode("utf-8-sig"),
        "forecast_daily.csv",
        "text/csv",
        key="dl_daily",
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — По артикулам
# ══════════════════════════════════════════════════════════════

with tab_articles:
    adf = fetch_dataframe(FORECAST_ARTICLE_QUERY, params)

    if adf.empty:
        st.info("Нет данных за выбранный период")
        st.stop()

    # Ensure numeric
    _art_num = [
        "orders_count", "orders_amount", "days_with_orders", "sales_count",
        "net_revenue", "gross_revenue", "commission_amount", "cost_amount",
        "profit_amount", "avg_price_before_spp", "avg_price_after_spp",
        "avg_spp_pct", "current_stock", "buyout_pct", "avg_orders_per_day",
        "days_of_stock",
    ]
    for c in _art_num:
        if c in adf.columns:
            adf[c] = pd.to_numeric(adf[c], errors="coerce").fillna(0)

    # Margin %
    adf["margin_pct"] = np.where(
        adf["net_revenue"] > 0,
        (adf["profit_amount"] / adf["net_revenue"] * 100).round(0),
        0,
    ).astype(float)

    # Pagination
    st.markdown("### Прогноз по артикулам")

    c_p1, c_p2, c_p3 = st.columns([1, 1, 4])
    with c_p1:
        page_size_a = st.selectbox(
            "Строк", [15, 30, 50, 100], index=0,
            key="pg_size_art", label_visibility="collapsed",
        )
    total_rows_a = len(adf)
    total_pages_a = max((total_rows_a - 1) // page_size_a + 1, 1)
    with c_p2:
        page_a = st.number_input(
            "Стр.", min_value=1, max_value=total_pages_a, value=1,
            key="pg_num_art", label_visibility="collapsed",
        )
    s_a = (page_a - 1) * page_size_a
    e_a = min(s_a + page_size_a, total_rows_a)
    display_a = adf.iloc[s_a:e_a]

    # Header
    hdr_a = (
        "<tr>"
        "<th>#</th><th>Артикул</th><th>Предмет</th>"
        "<th>Заказы шт</th><th>Заказы \u20bd</th><th>Ср. заказов/<br>день</th>"
        "<th>Продажи</th><th>% выкупа</th>"
        "<th>Ср. цена<br>до СПП</th><th>СПП%</th>"
        "<th>Комиссия</th><th>Себест-ть</th><th>Прибыль</th><th>Маржа%</th>"
        "<th>Остаток</th><th>Дней<br>запаса</th>"
        "</tr>"
    )

    rows_a = ""
    for idx, (_, row) in enumerate(display_a.iterrows(), start=s_a + 1):
        profit = float(row["profit_amount"])
        margin = float(row["margin_pct"])
        pcls = "pos" if profit > 0 else ("neg" if profit < 0 else "")
        mcls = "pos" if margin > 0 else ("neg" if margin < 0 else "")

        dos = row["days_of_stock"]
        if pd.isna(dos) or dos == 0:
            dos_str = "\u2014"
            dos_cls = ""
        else:
            dos = int(dos)
            dos_str = str(dos)
            if dos > 30:
                dos_cls = "stock-ok"
            elif dos >= 15:
                dos_cls = "stock-warn"
            else:
                dos_cls = "stock-crit"

        tr = "<tr>"
        tr += f'<td class="rn">{idx}</td>'
        tr += f'<td>{row.get("supplier_article", "")}</td>'
        tr += f'<td>{row.get("subject", "")}</td>'
        tr += f'<td class="num">{fmt_number(row["orders_count"])}</td>'
        tr += f'<td class="num">{fmt_number(row["orders_amount"])}</td>'
        tr += f'<td class="ctr">{row["avg_orders_per_day"]:.1f}</td>'
        tr += f'<td class="num">{fmt_number(row["sales_count"])}</td>'
        tr += f'<td class="ctr">{fmt_pct_tbl(row["buyout_pct"])}</td>'
        tr += f'<td class="num">{fmt_number(row["avg_price_before_spp"])}</td>'
        tr += f'<td class="ctr">{fmt_pct_tbl(row["avg_spp_pct"])}</td>'
        tr += f'<td class="num">{fmt_number(row["commission_amount"])}</td>'
        tr += f'<td class="num">{fmt_number(row["cost_amount"])}</td>'
        tr += f'<td class="num {pcls}">{fmt_number(profit)}</td>'
        tr += f'<td class="ctr {mcls}">{fmt_pct_tbl(margin)}</td>'
        tr += f'<td class="num">{fmt_number(row["current_stock"])}</td>'
        tr += f'<td class="ctr {dos_cls}">{dos_str}</td>'
        tr += "</tr>"
        rows_a += tr

    # Totals
    ta_orders = int(display_a["orders_count"].sum())
    ta_orders_amt = display_a["orders_amount"].sum()
    ta_sales = int(display_a["sales_count"].sum())
    ta_comm = display_a["commission_amount"].sum()
    ta_cost = display_a["cost_amount"].sum()
    ta_profit = display_a["profit_amount"].sum()
    ta_rev = display_a["net_revenue"].sum()
    ta_margin = (ta_profit / ta_rev * 100) if ta_rev else 0
    ta_stock = int(display_a["current_stock"].sum())
    ta_pcls = "pos" if ta_profit > 0 else ("neg" if ta_profit < 0 else "")
    ta_mcls = "pos" if ta_margin > 0 else ("neg" if ta_margin < 0 else "")

    ftr_a = (
        "<tr>"
        '<td></td><td><b>Итого</b></td><td></td>'
        f'<td class="num">{fmt_number(ta_orders)}</td>'
        f'<td class="num">{fmt_number(ta_orders_amt)}</td>'
        '<td></td>'
        f'<td class="num">{fmt_number(ta_sales)}</td>'
        '<td></td><td></td><td></td>'
        f'<td class="num">{fmt_number(ta_comm)}</td>'
        f'<td class="num">{fmt_number(ta_cost)}</td>'
        f'<td class="num {ta_pcls}">{fmt_number(ta_profit)}</td>'
        f'<td class="ctr {ta_mcls}">{fmt_pct_tbl(ta_margin)}</td>'
        f'<td class="num">{fmt_number(ta_stock)}</td>'
        '<td></td>'
        "</tr>"
    )

    html_art = (
        f'{TABLE_CSS}<div class="art-wrap"><table class="art-t">'
        f'<thead>{hdr_a}</thead><tbody>{rows_a}</tbody>'
        f'<tfoot>{ftr_a}</tfoot></table></div>'
    )

    st.markdown(html_art, unsafe_allow_html=True)
    st.caption(f"Показано {s_a + 1}\u2013{e_a} из {total_rows_a}")

    # CSV export
    st.download_button(
        "\U0001f4e5 Скачать CSV (по артикулам)",
        adf.to_csv(index=False).encode("utf-8-sig"),
        "forecast_articles.csv",
        "text/csv",
        key="dl_art",
    )
