"""Конверсия рекламы — аналитика рекламных кампаний и ROI."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from marts import fetch_dataframe, ADS_DAILY_QUERY, FINANCE_DAILY_QUERY, ORDERS_DAILY_AMOUNT_QUERY
from styles import inject_global_styles, fmt_number, fmt_pct_tbl, table_css, PLOTLY_LAYOUT, PLOTLY_COLORS
from auth import check_auth, logout

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📢 Конверсия рекламы")

# ── Helpers ──────────────────────────────────────────────────


def _safe_div(a, b):
    if not b or b == 0:
        return 0.0
    return a / b

# ── CSS ──────────────────────────────────────────────────────

TABLE_CSS = table_css("ads") + '<style>.ads .warn{color:#ca8a04;font-weight:700}</style>'

# ── Date filters ─────────────────────────────────────────────
fcol1, fcol2 = st.columns(2)
with fcol1:
    d_to = st.date_input("Дата окончания", value=date.today())
with fcol2:
    d_from = st.date_input("Дата начала", value=d_to - timedelta(days=29))

params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Load data ────────────────────────────────────────────────

ads = fetch_dataframe(ADS_DAILY_QUERY, params)
sales = fetch_dataframe(FINANCE_DAILY_QUERY, params)
# Normalize column names for compatibility
if "report_date" in sales.columns:
    sales = sales.rename(columns={"report_date": "sales_date", "ppvz_for_pay": "net_revenue"})
orders = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)

if ads.empty:
    st.info("Нет данных по рекламе за выбранный период")
    st.stop()

# Ensure numeric types
for c in ["views_count", "clicks_count", "ctr", "cpc", "orders_from_ads", "spend_amount"]:
    if c in ads.columns:
        ads[c] = pd.to_numeric(ads[c], errors="coerce").fillna(0)

# ── Entity filters ───────────────────────────────────────────

ref = sales if not sales.empty else ads
brands = sorted(ref["brand"].dropna().unique()) if "brand" in ref.columns else []
subjects = sorted(ref["subject"].dropna().unique()) if "subject" in ref.columns else []
fcol3, fcol4 = st.columns(2)
with fcol3:
    sel_brands = st.multiselect("Бренд", brands, default=[])
with fcol4:
    sel_subjects = st.multiselect("Предмет", subjects, default=[])

# Apply filters to ads via nm_id list from sales
if sel_brands and not sales.empty:
    nm_ids = sales.loc[sales["brand"].isin(sel_brands), "nm_id"].unique()
    ads = ads[ads["nm_id"].isin(nm_ids)]
if sel_subjects and not sales.empty:
    nm_ids = sales.loc[sales["subject"].isin(sel_subjects), "nm_id"].unique()
    ads = ads[ads["nm_id"].isin(nm_ids)]

if ads.empty:
    st.warning("Нет данных по выбранным фильтрам")
    st.stop()

# ── KPI cards ────────────────────────────────────────────────

total_spend = ads["spend_amount"].sum()
total_views = ads["views_count"].sum()
total_clicks = ads["clicks_count"].sum()
avg_ctr = _safe_div(total_clicks, total_views) * 100

total_orders_amount = orders["orders_amount"].sum() if not orders.empty and "orders_amount" in orders.columns else 0
drr_total = _safe_div(total_spend, total_orders_amount) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Расход на рекламу", f"{total_spend:,.0f} ₽".replace(",", " "))
c2.metric("Показы", f"{int(total_views):,}".replace(",", " "))
c3.metric("Клики", f"{int(total_clicks):,}".replace(",", " "))
c4.metric("CTR средний", f"{avg_ctr:.2f}%")
c5.metric("ДРР %", f"{drr_total:.1f}%")

# ── Tabs ─────────────────────────────────────────────────────

tab_articles, tab_days = st.tabs(["По артикулам", "По дням"])

# ══════════════════════════════════════════════════════════════
#  TAB 1: By articles
# ══════════════════════════════════════════════════════════════

with tab_articles:
    art = (
        ads.groupby(["nm_id", "supplier_article"])
        .agg(
            views=("views_count", "sum"),
            clicks=("clicks_count", "sum"),
            orders_ads=("orders_from_ads", "sum"),
            spend=("spend_amount", "sum"),
        )
        .reset_index()
    )
    art["ctr"] = art.apply(lambda r: _safe_div(r["clicks"], r["views"]) * 100, axis=1)
    art["cpc"] = art.apply(lambda r: _safe_div(r["spend"], r["clicks"]), axis=1)
    art["cpo"] = art.apply(lambda r: _safe_div(r["spend"], r["orders_ads"]), axis=1)

    # Join with sales for subject, brand, revenue
    if not sales.empty:
        sales_agg = (
            sales.groupby("nm_id")
            .agg(
                subject=("subject", "first"),
                brand=("brand", "first"),
                orders_total=("orders_count", "sum"),
                revenue=("net_revenue", "sum"),
            )
            .reset_index()
        )
        art = art.merge(sales_agg, on="nm_id", how="left")
    else:
        art["subject"] = ""
        art["brand"] = ""
        art["orders_total"] = 0
        art["revenue"] = 0

    art["revenue"] = art["revenue"].fillna(0)
    art["drr"] = art.apply(lambda r: _safe_div(r["spend"], r["revenue"]) * 100, axis=1)
    art["roi"] = art.apply(lambda r: _safe_div(r["revenue"] - r["spend"], r["spend"]) * 100, axis=1)
    art = art.sort_values("spend", ascending=False)

    def _drr_cls(v):
        if v < 10:
            return "pos"
        if v <= 25:
            return "warn"
        return "neg"

    def _roi_cls(v):
        if v > 100:
            return "pos"
        if v >= 0:
            return "warn"
        return "neg"

    # Build HTML table
    hdr = (
        "<tr><th>#</th><th>Артикул</th><th>Предмет</th>"
        "<th>Показы</th><th>Клики</th><th>CTR%</th><th>CPC</th>"
        "<th>Заказы из рекл.</th><th>Расход</th><th>Выручка</th>"
        "<th>ДРР%</th><th>ROI%</th></tr>"
    )
    rows = ""
    for i, (_, r) in enumerate(art.iterrows(), 1):
        drr_c = _drr_cls(r["drr"]) if r["revenue"] > 0 else ""
        roi_c = _roi_cls(r["roi"]) if r["spend"] > 0 else ""
        rows += (
            f'<tr><td class="num">{i}</td>'
            f'<td>{r["supplier_article"]}</td>'
            f'<td>{r.get("subject", "")}</td>'
            f'<td class="num">{fmt_number(r["views"])}</td>'
            f'<td class="num">{fmt_number(r["clicks"])}</td>'
            f'<td class="ctr">{fmt_pct_tbl(r["ctr"])}</td>'
            f'<td class="num">{fmt_number(r["cpc"], 2)}</td>'
            f'<td class="num">{fmt_number(r["orders_ads"])}</td>'
            f'<td class="num">{fmt_number(r["spend"])}</td>'
            f'<td class="num">{fmt_number(r["revenue"])}</td>'
            f'<td class="num {drr_c}">{fmt_pct_tbl(r["drr"])}</td>'
            f'<td class="num {roi_c}">{fmt_pct_tbl(r["roi"])}</td></tr>'
        )

    # Footer totals
    t_views = art["views"].sum()
    t_clicks = art["clicks"].sum()
    t_ctr = _safe_div(t_clicks, t_views) * 100
    t_spend = art["spend"].sum()
    t_cpc = _safe_div(t_spend, t_clicks)
    t_orders_ads = art["orders_ads"].sum()
    t_revenue = art["revenue"].sum()
    t_drr = _safe_div(t_spend, t_revenue) * 100
    t_roi = _safe_div(t_revenue - t_spend, t_spend) * 100

    foot = (
        f'<tr><td></td><td><b>ИТОГО</b></td><td></td>'
        f'<td class="num">{fmt_number(t_views)}</td>'
        f'<td class="num">{fmt_number(t_clicks)}</td>'
        f'<td class="ctr">{fmt_pct_tbl(t_ctr)}</td>'
        f'<td class="num">{fmt_number(t_cpc, 2)}</td>'
        f'<td class="num">{fmt_number(t_orders_ads)}</td>'
        f'<td class="num">{fmt_number(t_spend)}</td>'
        f'<td class="num">{fmt_number(t_revenue)}</td>'
        f'<td class="num {_drr_cls(t_drr)}">{fmt_pct_tbl(t_drr)}</td>'
        f'<td class="num {_roi_cls(t_roi)}">{fmt_pct_tbl(t_roi)}</td></tr>'
    )

    html = (
        f'{TABLE_CSS}<div class="ads-wrap"><table class="ads">'
        f'<thead>{hdr}</thead><tbody>{rows}</tbody>'
        f'<tfoot>{foot}</tfoot></table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2: By days
# ══════════════════════════════════════════════════════════════

with tab_days:
    daily = (
        ads.groupby("ads_date")
        .agg(
            views=("views_count", "sum"),
            clicks=("clicks_count", "sum"),
            orders_ads=("orders_from_ads", "sum"),
            spend=("spend_amount", "sum"),
        )
        .reset_index()
        .sort_values("ads_date")
    )
    daily["ctr"] = daily.apply(lambda r: _safe_div(r["clicks"], r["views"]) * 100, axis=1)
    daily["cpc"] = daily.apply(lambda r: _safe_div(r["spend"], r["clicks"]), axis=1)

    # Plotly chart: spend bars + CTR line on secondary axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=daily["ads_date"], y=daily["spend"],
            name="Расход", marker_color=PLOTLY_COLORS["blue"],
            opacity=0.75,
            marker=dict(line=dict(width=0.5, color=PLOTLY_COLORS["blue_dark"])),
            hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Расход: %{y:,.0f} ₽<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["ads_date"], y=daily["ctr"],
            name="CTR %", mode="lines+markers",
            line=dict(color=PLOTLY_COLORS["amber"], width=2.5, shape="spline"),
            marker=dict(size=7, line=dict(width=1.5, color="white")),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
            hovertemplate="<b>%{x|%d.%m.%Y}</b><br>CTR: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.25,
    )
    fig.update_xaxes(dtick="D1", tickformat="%d.%m", gridcolor="#f1f5f9")
    fig.update_yaxes(title_text="Расход, ₽", secondary_y=False, gridcolor="#f1f5f9")
    fig.update_yaxes(
        title_text="CTR %", secondary_y=True, gridcolor="#f1f5f9",
        title_font=dict(color=PLOTLY_COLORS["amber"]),
        tickfont=dict(color=PLOTLY_COLORS["amber"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily HTML table
    daily_sorted = daily.sort_values("ads_date", ascending=False)
    hdr_d = (
        "<tr><th>Дата</th><th>Показы</th><th>Клики</th>"
        "<th>CTR%</th><th>CPC</th><th>Заказы</th><th>Расход</th></tr>"
    )
    rows_d = ""
    for _, r in daily_sorted.iterrows():
        dt = pd.to_datetime(r["ads_date"]).strftime("%d.%m.%Y")
        rows_d += (
            f'<tr><td>{dt}</td>'
            f'<td class="num">{fmt_number(r["views"])}</td>'
            f'<td class="num">{fmt_number(r["clicks"])}</td>'
            f'<td class="ctr">{fmt_pct_tbl(r["ctr"])}</td>'
            f'<td class="num">{fmt_number(r["cpc"], 2)}</td>'
            f'<td class="num">{fmt_number(r["orders_ads"])}</td>'
            f'<td class="num">{fmt_number(r["spend"])}</td></tr>'
        )

    html_d = (
        f'{TABLE_CSS}<div class="ads-wrap"><table class="ads">'
        f'<thead>{hdr_d}</thead><tbody>{rows_d}</tbody></table></div>'
    )
    st.markdown(html_d, unsafe_allow_html=True)

# ── CSV download ─────────────────────────────────────────────

csv_data = art.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 Скачать отчёт CSV",
    csv_data,
    "ad_conversion.csv",
    "text/csv",
)
