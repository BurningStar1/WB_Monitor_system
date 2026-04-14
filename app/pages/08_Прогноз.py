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
from styles import plotly_defaults, inject_global_styles, fmt_number, fmt_pct_tbl, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, wb_link, render_table
from auth import check_auth, logout

# ── Page setup ────────────────────────────────────────────────

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("\U0001f4c8 Прогноз")

with st.expander("ℹ️ Как работает прогноз", expanded=False):
    st.markdown(
        """
        **Метод:** линейная регрессия по **7-дневной скользящей средней** (MA-7),
        построенная по истории за выбранный период.

        **Интервал 95%:** `± 1.96·σ·√(1 + t/N)`, где
        `σ` — стандартное отклонение остатков линии тренда,
        `N` — число точек истории, `t` — шаг прогноза.
        Интервал **расширяется** с горизонтом — дальше во времени, тем выше неопределённость.

        **Горизонты:** 7 / 14 / 30 дней.
        Для прогнозов прибыли отрицательные значения не обрезаются —
        убытки видны на графике красным.

        **Ограничения:** метод не учитывает сезонность и внешние события (распродажи, акции).
        Для длинных горизонтов (>30 дней) точность падает.
        """
    )

# ── Filters ──────────────────────────────────────────
d_from, d_to = date_filter_bar("forecast", default_days=90)
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

    # ── Fill missing calendar days in the selected range with zeros ──
    # Без этого длинные диапазоны (например «весь год») отрисовываются
    # «зубцами» из-за дней без заказов, и MA 7д/14д получаются ломаными.
    full_range = pd.date_range(start=pd.Timestamp(d_from), end=pd.Timestamp(d_to), freq="D")
    df = (
        df.set_index("order_date")
          .reindex(full_range)
          .rename_axis("order_date")
          .reset_index()
    )
    # Fill zero-value columns for days with no orders
    _fill_zero = ["orders_count", "orders_amount", "sales_count", "net_revenue",
                  "gross_revenue", "commission_amount", "cost_amount", "profit_amount"]
    for c in _fill_zero:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Recompute MAs on the reindexed (gap-free) series so чарт без пробелов
    df["ma_orders_7d"] = df["orders_count"].rolling(7, min_periods=1).mean().round(2)
    df["ma_orders_14d"] = df["orders_count"].rolling(14, min_periods=1).mean().round(2)
    df["ma_revenue_7d"] = df["net_revenue"].rolling(7, min_periods=1).mean().round(2)
    df["ma_profit_7d"] = df["profit_amount"].rolling(7, min_periods=1).mean().round(2)

    n_days = len(df)

    # ── Forecast projection (linear extrapolation on last 14 MA7 points) ──

    last_date = df["order_date"].iloc[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=forecast_horizon,
        freq="D",
    )

    def _linear_forecast_with_ci(ma_vals: pd.Series, horizon: int, clip_nonneg: bool = False):
        """Fit a linear trend on the last ≤14 MA points and project `horizon` days.

        Returns ``(point_forecast, lower_ci, upper_ci)`` as numpy arrays of
        length ``horizon``. CI is ±1.96·σ where σ is the residual stddev of
        the fit (widening with distance). If there are too few points, returns
        zero arrays.
        """
        vals = ma_vals.dropna()
        fit_n = min(14, len(vals))
        if fit_n < 2:
            z = np.zeros(horizon)
            return z, z, z
        y = vals.iloc[-fit_n:].values.astype(float)
        x = np.arange(fit_n)
        coef = np.polyfit(x, y, 1)
        resid = y - np.polyval(coef, x)
        sigma = float(np.std(resid, ddof=1)) if fit_n > 1 else 0.0
        fc_x = np.arange(fit_n, fit_n + horizon)
        point = np.polyval(coef, fc_x)
        # Widen CI with distance (σ * sqrt(1 + step/fit_n) approximation)
        widen = np.sqrt(1.0 + np.arange(1, horizon + 1) / max(fit_n, 1))
        band = 1.96 * sigma * widen
        lower = point - band
        upper = point + band
        if clip_nonneg:
            point = np.clip(point, 0, None)
            lower = np.clip(lower, 0, None)
        return point, lower, upper

    # -- Orders forecast --
    ma_orders_vals = df["ma_orders_7d"].dropna()
    forecast_orders_vals, fc_ord_lo, fc_ord_hi = _linear_forecast_with_ci(
        ma_orders_vals, forecast_horizon, clip_nonneg=True
    )

    # -- Profit forecast --
    ma_profit_vals = df["ma_profit_7d"].dropna()
    forecast_profit_vals, fc_prf_lo, fc_prf_hi = _linear_forecast_with_ci(
        ma_profit_vals, forecast_horizon, clip_nonneg=False
    )

    # Prepend last historical point for visual continuity
    fc_dates_full = pd.DatetimeIndex([last_date]).union(forecast_dates)
    _last_orders_ma = float(ma_orders_vals.iloc[-1]) if len(ma_orders_vals) else 0
    _last_profit_ma = float(ma_profit_vals.iloc[-1]) if len(ma_profit_vals) else 0
    fc_orders_full = np.concatenate([[_last_orders_ma], forecast_orders_vals])
    fc_profit_full = np.concatenate([[_last_profit_ma], forecast_profit_vals])
    fc_ord_lo_full = np.concatenate([[_last_orders_ma], fc_ord_lo])
    fc_ord_hi_full = np.concatenate([[_last_orders_ma], fc_ord_hi])
    fc_prf_lo_full = np.concatenate([[_last_profit_ma], fc_prf_lo])
    fc_prf_hi_full = np.concatenate([[_last_profit_ma], fc_prf_hi])

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
    # Upper CI (invisible, defines band top)
    fig_orders.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_ord_hi_full,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # Lower CI (fills to upper)
    fig_orders.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_ord_lo_full,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(147,197,253,0.20)",
        name="95% интервал",
        hovertemplate="<b>%{x|%d.%m}</b><br>Нижн. граница: %{y:.1f}<extra></extra>",
    ))
    fig_orders.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_orders_full,
        name="\u041f\u0440\u043e\u0433\u043d\u043e\u0437",
        mode="lines+markers",
        line=dict(color=PLOTLY_COLORS["blue_light"], width=2.5, dash="dash", shape="spline"),
        marker=dict(color=PLOTLY_COLORS["blue_light"], size=6,
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u043e\u0433\u043d\u043e\u0437: %{y:.1f} \u0448\u0442<extra></extra>",
    ))

    # Vertical dashed line marking forecast start
    _vline_x = last_date.to_pydatetime()
    fig_orders.add_shape(
        type="line", x0=_vline_x, x1=_vline_x, y0=0, y1=1, yref="paper",
        line=dict(width=1.5, dash="dash", color=PLOTLY_COLORS["slate"]),
    )
    fig_orders.add_annotation(
        x=_vline_x, y=1, yref="paper",
        text="\u041d\u0430\u0447\u0430\u043b\u043e \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430",
        showarrow=False, font=dict(size=10, color=PLOTLY_COLORS["slate"]),
        yanchor="bottom",
    )

    fig_orders.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="\u0417\u0430\u043a\u0430\u0437\u044b, \u0448\u0442",
        xaxis_title="",
        barmode="overlay",
        bargap=0.25,
    )

    plotly_defaults(fig_orders)
    st.plotly_chart(fig_orders, width="stretch")

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

    # ── Forecast projection trace (profit) + 95% CI ───────────
    fig_profit.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_prf_hi_full,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig_profit.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_prf_lo_full,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(196,181,253,0.20)",
        name="95% интервал",
        hovertemplate="<b>%{x|%d.%m}</b><br>Нижн. граница: %{y:,.0f} ₽<extra></extra>",
    ))
    fig_profit.add_trace(go.Scatter(
        x=fc_dates_full, y=fc_profit_full,
        name="\u041f\u0440\u043e\u0433\u043d\u043e\u0437",
        mode="lines+markers",
        line=dict(color="#c4b5fd", width=2.5, dash="dash", shape="spline"),
        marker=dict(color="#c4b5fd", size=6,
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%d.%m}</b><br>\u041f\u0440\u043e\u0433\u043d\u043e\u0437: %{y:,.0f} \u20bd<extra></extra>",
    ))

    # Vertical dashed line marking forecast start
    fig_profit.add_shape(
        type="line", x0=_vline_x, x1=_vline_x, y0=0, y1=1, yref="paper",
        line=dict(width=1.5, dash="dash", color=PLOTLY_COLORS["slate"]),
    )
    fig_profit.add_annotation(
        x=_vline_x, y=1, yref="paper",
        text="\u041d\u0430\u0447\u0430\u043b\u043e \u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0430",
        showarrow=False, font=dict(size=10, color=PLOTLY_COLORS["slate"]),
        yanchor="bottom",
    )

    fig_profit.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="\u041f\u0440\u0438\u0431\u044b\u043b\u044c, \u20bd",
        xaxis_title="",
        barmode="overlay",
        bargap=0.25,
    )

    plotly_defaults(fig_profit)
    st.plotly_chart(fig_profit, width="stretch")

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
        f'{TABLE_CSS}<div class="art-wrap"><table class="art-t" data-sortable>'
        f'<thead>{hdr}</thead><tbody>{rows}</tbody>'
        f'<tfoot>{ftr}</tfoot></table></div>{SORT_JS}'
    )

    render_table(html_daily)
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
        tr += f'<td>{wb_link(row.get("nm_id", 0), row.get("supplier_article", ""))}</td>'
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
        f'{TABLE_CSS}<div class="art-wrap"><table class="art-t" data-sortable>'
        f'<thead>{hdr_a}</thead><tbody>{rows_a}</tbody>'
        f'<tfoot>{ftr_a}</tfoot></table></div>{SORT_JS}'
    )

    render_table(html_art)
    st.caption(f"Показано {s_a + 1}\u2013{e_a} из {total_rows_a}")

    # CSV export
    st.download_button(
        "\U0001f4e5 Скачать CSV (по артикулам)",
        adf.to_csv(index=False).encode("utf-8-sig"),
        "forecast_articles.csv",
        "text/csv",
        key="dl_art",
    )
