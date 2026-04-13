"""Отчёт по артикулам — детальная таблица в стиле аналитического сервиса."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from marts import (
    fetch_dataframe,
    DASHBOARD_DETAIL_QUERY,
    FINANCE_DAILY_QUERY,
    ORDERS_DAILY_AMOUNT_QUERY,
    ADS_DAILY_QUERY,
    STOCKS_QUERY,
    STOCKS_HISTORY_QUERY,
)
from styles import inject_global_styles

# ── Helpers ───────────────────────────────────────────────────

def _wb_photo_url(nm_id: int) -> str:
    """Thumbnail URL for a WB product by nm_id."""
    vol = nm_id // 100000
    part = nm_id // 1000
    _RANGES = [
        (143, 1), (287, 2), (431, 3), (719, 4), (1007, 5),
        (1061, 6), (1115, 7), (1169, 8), (1313, 9), (1601, 10),
        (1655, 11), (1919, 12), (2045, 13), (2189, 14),
    ]
    basket = None
    for limit, num in _RANGES:
        if vol <= limit:
            basket = f"{num:02d}"
            break
    if basket is None:
        basket = f"{15 + (vol - 2190) // 216:02d}"
    return f"https://basket-{basket}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/images/c246x328/1.webp"


# ── Page setup ────────────────────────────────────────────────

inject_global_styles()
st.title("📦 Отчёт по артикулам")

# ── Sidebar filters ──────────────────────────────────────────

_DAYS_MAP = {"7 дней": 7, "14 дней": 14, "30 дней": 30, "90 дней": 90}

with st.sidebar:
    st.header("Фильтры")
    d_to = st.date_input("Дата окончания", value=date.today())
    period = st.selectbox("Период", list(_DAYS_MAP.keys()), index=0)
    d_from = d_to - timedelta(days=_DAYS_MAP[period] - 1)
    st.caption(f"{d_from.strftime('%d.%m.%Y')} — {d_to.strftime('%d.%m.%Y')}")

params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Load data ─────────────────────────────────────────────────

ord_df = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)
sales_df = fetch_dataframe(DASHBOARD_DETAIL_QUERY, params)
fin_df = fetch_dataframe(FINANCE_DAILY_QUERY, params)
ads_df = fetch_dataframe(ADS_DAILY_QUERY, params)
stocks_df = fetch_dataframe(STOCKS_QUERY, {})
stocks_hist = fetch_dataframe(STOCKS_HISTORY_QUERY, {})

if sales_df.empty and ord_df.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Entity filters ────────────────────────────────────────────

_src = sales_df if not sales_df.empty else ord_df
brands = sorted(_src["brand"].dropna().unique()) if "brand" in _src.columns else []
subjects = sorted(_src["subject"].dropna().unique()) if "subject" in _src.columns else []
articles = sorted(_src["supplier_article"].dropna().unique()) if "supplier_article" in _src.columns else []

with st.sidebar:
    sel_brands = st.multiselect("Бренд", brands)
    sel_subjects = st.multiselect("Предмет", subjects)
    sel_articles = st.multiselect("Артикул поставщика", articles)


def _filt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if sel_brands and "brand" in df.columns:
        df = df[df["brand"].isin(sel_brands)]
    if sel_subjects and "subject" in df.columns:
        df = df[df["subject"].isin(sel_subjects)]
    if sel_articles and "supplier_article" in df.columns:
        df = df[df["supplier_article"].isin(sel_articles)]
    return df


ord_df = _filt(ord_df)
sales_df = _filt(sales_df)
fin_df = _filt(fin_df)
ads_df = _filt(ads_df)

# ── 1. Sales aggregation per article ─────────────────────────

if not sales_df.empty:
    sales_agg = (
        sales_df.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            orders_count=("orders_count", "sum"),
            sales_count=("sales_count", "sum"),
            returns_count=("returns_count", "sum"),
            net_revenue=("net_revenue", "sum"),
            profit_amount=("profit_amount", "sum"),
            cost_amount=("cost_amount", "sum"),
            avg_price_before_spp=("avg_price_before_spp", "mean"),
            avg_price_after_spp=("avg_price_after_spp", "mean"),
        )
        .reset_index()
    )
else:
    sales_agg = pd.DataFrame(columns=["nm_id", "supplier_article"])

# ── 2. Orders aggregation + daily pivot ───────────────────────

daily_date_cols: list[str] = []

if not ord_df.empty:
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    n_days = max((ord_df["order_date"].max() - ord_df["order_date"].min()).days + 1, 1)

    ord_agg = (
        ord_df.groupby(["nm_id", "supplier_article"])
        .agg(
            orders_amount=("orders_amount", "sum"),
            _total_cnt=("orders_count", "sum"),
        )
        .reset_index()
    )
    ord_agg["orders_speed"] = (ord_agg["_total_cnt"] / n_days).round(1)

    # Daily pivot — last 7 dates
    last_dates = sorted(ord_df["order_date"].unique())[-7:]
    daily_slice = ord_df[ord_df["order_date"].isin(last_dates)].copy()

    pivot = daily_slice.pivot_table(
        index=["nm_id", "supplier_article"],
        columns="order_date",
        values="orders_count",
        aggfunc="sum",
        fill_value=0,
    )
    daily_date_cols = [d.strftime("%d.%m") for d in sorted(pivot.columns)]
    pivot.columns = daily_date_cols
    pivot = pivot.reset_index()

    # Orders dynamics list (for BarChartColumn)
    dyn = (
        daily_slice.groupby(["nm_id", "supplier_article", "order_date"])["orders_count"]
        .sum()
        .reset_index()
        .sort_values("order_date")
        .groupby(["nm_id", "supplier_article"])["orders_count"]
        .apply(list)
        .reset_index()
        .rename(columns={"orders_count": "orders_dynamics"})
    )
else:
    ord_agg = pd.DataFrame(columns=["nm_id", "supplier_article"])
    pivot = pd.DataFrame(columns=["nm_id", "supplier_article"])
    dyn = pd.DataFrame(columns=["nm_id", "supplier_article"])

# ── 3. Finance aggregation ────────────────────────────────────

has_finance = not fin_df.empty
if has_finance:
    fin_agg = (
        fin_df.groupby(["nm_id", "supplier_article"])
        .agg(
            fin_sales_amt=("sales_amount", "sum"),
            fin_returns_amt=("returns_amount", "sum"),
            fin_commission=("commission_amount", "sum"),
            fin_logistics=("logistics_amount", "sum"),
            fin_storage=("storage_amount", "sum"),
            fin_penalty=("penalty_amount", "sum"),
            fin_acceptance=("acceptance_amount", "sum"),
            fin_deduction=("deduction_amount", "sum"),
        )
        .reset_index()
    )
else:
    fin_agg = pd.DataFrame(columns=["nm_id", "supplier_article"])

# ── 4. Ads aggregation ────────────────────────────────────────

if not ads_df.empty:
    ads_agg = (
        ads_df.groupby("nm_id")
        .agg(ads_spend=("spend_amount", "sum"))
        .reset_index()
    )
else:
    ads_agg = pd.DataFrame(columns=["nm_id"])

# ── 5. Current stocks ─────────────────────────────────────────

if not stocks_df.empty:
    stk_agg = (
        stocks_df.groupby("nm_id")
        .agg(stock_qty=("quantity_full", "sum"))
        .reset_index()
    )
else:
    stk_agg = pd.DataFrame(columns=["nm_id"])

# ── 6. Stock history sparklines ────────────────────────────────

if not stocks_hist.empty:
    stk_spark = (
        stocks_hist.groupby(["nm_id", "snapshot_date"])["stock_qty"]
        .sum()
        .reset_index()
        .sort_values("snapshot_date")
        .groupby("nm_id")["stock_qty"]
        .apply(list)
        .reset_index()
        .rename(columns={"stock_qty": "stock_history"})
    )
else:
    stk_spark = pd.DataFrame(columns=["nm_id"])

# ── Merge all sources ─────────────────────────────────────────

r = sales_agg.copy() if not sales_agg.empty else ord_agg[["nm_id", "supplier_article"]].copy()

for extra in [ord_agg, pivot, dyn, fin_agg, ads_agg, stk_agg, stk_spark]:
    if extra.empty or len(extra.columns) <= 1:
        continue
    keys = [k for k in ["nm_id", "supplier_article"] if k in extra.columns and k in r.columns]
    if not keys:
        continue
    r = r.merge(extra, on=keys, how="left")

# Handle list columns before fillna
for lc in ("stock_history", "orders_dynamics"):
    if lc in r.columns:
        r[lc] = r[lc].apply(lambda x: x if isinstance(x, list) else [])

r = r.fillna(0)

# ── Ensure all columns exist ──────────────────────────────────

_ensure = [
    "orders_amount", "orders_speed", "orders_count", "sales_count",
    "returns_count", "net_revenue", "profit_amount", "cost_amount",
    "avg_price_before_spp", "avg_price_after_spp", "stock_qty",
    "fin_sales_amt", "fin_returns_amt", "fin_commission", "fin_logistics",
    "fin_storage", "fin_penalty", "fin_acceptance", "fin_deduction",
    "ads_spend",
]
for c in _ensure:
    if c not in r.columns:
        r[c] = 0

# ── Calculated columns ────────────────────────────────────────

# Photo
r["photo"] = r["nm_id"].apply(lambda x: _wb_photo_url(int(x)) if x else "")

# Buyout %
r["buyout_pct"] = np.where(
    r["orders_count"] > 0,
    (r["sales_count"] / r["orders_count"] * 100).round(0),
    0,
).astype(int)

# SPP %
r["spp_pct"] = np.where(
    r["avg_price_before_spp"] > 0,
    ((1 - r["avg_price_after_spp"] / r["avg_price_before_spp"]) * 100).round(1),
    0,
)

# Ads share %
r["ads_share_pct"] = np.where(
    r["net_revenue"] > 0,
    (r["ads_spend"] / r["net_revenue"] * 100).round(1),
    0,
)

# Other services = storage + penalty + acceptance + (deduction - ads_spend)
r["other_services"] = (
    r["fin_storage"] + r["fin_penalty"] + r["fin_acceptance"]
    + (r["fin_deduction"] - r["ads_spend"]).clip(lower=0)
).round(0)

# Profit per article
if has_finance:
    fin_real = r["fin_sales_amt"] - r["fin_returns_amt"]
    fin_svc = (
        r["fin_commission"] + r["fin_logistics"] + r["fin_storage"]
        + r["fin_penalty"] + r["fin_acceptance"] + r["fin_deduction"]
    )
    r["article_profit"] = (fin_real - fin_svc - r["cost_amount"]).round(0)
else:
    r["article_profit"] = r["profit_amount"].round(0)

# Margin %
r["margin_pct"] = np.where(
    r["net_revenue"] > 0,
    (r["article_profit"] / r["net_revenue"] * 100).round(0),
    0,
).astype(int)

# Round display columns
for c in ["avg_price_before_spp", "avg_price_after_spp", "orders_amount",
           "fin_commission", "fin_logistics", "ads_spend", "other_services"]:
    if c in r.columns:
        r[c] = r[c].round(0)

# Sort by orders_amount descending
sort_col = "orders_amount" if r["orders_amount"].sum() > 0 else "net_revenue"
r = r.sort_values(sort_col, ascending=False).reset_index(drop=True)

if r.empty:
    st.info("Нет данных")
    st.stop()

# ── SVG helpers ────────────────────────────────────────────────

def _sparkline(values, w=80, h=24, color="#7c3aed"):
    if not values or len(values) < 2:
        return ""
    mx = max(values) or 1
    mn = min(values)
    rng = mx - mn or 1
    pts = " ".join(
        f"{i / (len(values) - 1) * w:.1f},{h - (v - mn) / rng * (h - 4) - 2:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'
    )


def _barchart(values, w=70, h=22, color="#a78bfa"):
    if not values:
        return ""
    mx = max(values) or 1
    n = len(values)
    bw = w / n * 0.7
    gap = w / n
    bars = ""
    for i, v in enumerate(values):
        bh = max(v / mx * (h - 2), 1) if v > 0 else 0
        bars += f'<rect x="{i * gap:.1f}" y="{h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="1"/>'
    return f'<svg width="{w}" height="{h}" style="vertical-align:middle">{bars}</svg>'


def _fmt(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:,.0f}".replace(",", " ")


def _fmtp(v):
    if pd.isna(v) or v == 0:
        return "0%"
    return f"{v:.0f}%"


def _fmtp1(v):
    if pd.isna(v) or v == 0:
        return "0.0%"
    return f"{v:.1f}%"


# ── Notes ──────────────────────────────────────────────────────

st.caption(
    f"\\* значения по показателям рассчитаны как сумма за последние {_DAYS_MAP[period]} дней.  \n"
    f"\\*\\* средние значения рассчитаны по суммам за последние {_DAYS_MAP[period]} дней."
)

# ── Pagination ─────────────────────────────────────────────────

c_pg1, c_pg2, c_pg3 = st.columns([1, 1, 4])
with c_pg1:
    page_size = st.selectbox("Строк", [10, 25, 50, 100], index=0, label_visibility="collapsed")
total_rows = len(r)
total_pages = max((total_rows - 1) // page_size + 1, 1)
with c_pg2:
    page = st.number_input(
        "Стр.", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed"
    )
start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, total_rows)
display = r.iloc[start_idx:end_idx]

# ── Build HTML table ───────────────────────────────────────────

# Daily column totals for header
day_totals = {dc: int(r[dc].sum()) for dc in daily_date_cols if dc in r.columns}

TABLE_CSS = """
<style>
.art-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin-bottom:1rem;border:1px solid #e2e8f0}
.art-t{border-collapse:collapse;width:max-content;min-width:100%;
  font-size:12px;font-family:Inter,system-ui,sans-serif;background:#fff;color:#1e293b}

/* Header */
.art-t thead th{background:#f1f5f9;position:sticky;top:0;z-index:3;
  padding:6px 6px;border-bottom:2px solid #cbd5e1;border-right:1px solid #e2e8f0;
  font-weight:600;font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.3px;
  text-align:center;white-space:nowrap;vertical-align:bottom}
.art-t thead th:last-child{border-right:none}

/* Cells */
.art-t td{border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  padding:4px 6px;white-space:nowrap;vertical-align:middle}
.art-t td:last-child{border-right:none}

/* Zebra + hover */
.art-t tbody tr:nth-child(even){background:#fafbfc}
.art-t tbody tr:hover{background:#eef2ff}

/* Alignment helpers */
.art-t .num{text-align:right}.art-t .ctr{text-align:center}

/* Row number */
.art-t .rn{color:#94a3b8;font-size:11px;text-align:center;min-width:24px}

/* Daily order cells */
.art-t .day{text-align:center;min-width:30px;font-size:12px;font-weight:500;color:#64748b}
.art-t .day.hv{background:rgba(139,92,246,.12);color:#6d28d9;font-weight:700}

/* Profit colors */
.art-t .pos{color:#16a34a;font-weight:700}.art-t .neg{color:#dc2626;font-weight:700}

/* Small sub-labels */
.art-t .sub{font-size:9px;color:#94a3b8;margin-top:2px;letter-spacing:-.2px}

/* Photo */
.art-t .photo{width:34px;height:44px;object-fit:cover;border-radius:4px;
  box-shadow:0 1px 3px rgba(0,0,0,.1)}

/* Daily header */
.art-t th.day-h{font-size:10px;line-height:1.3;text-transform:none;letter-spacing:0;
  padding:5px 4px;min-width:36px}
.art-t th.day-h .dv{font-weight:800;font-size:13px;color:#1e293b;display:block}

/* Article card cell (photo + info combined) */
.art-t .art-card{display:flex;align-items:center;gap:6px;min-width:180px}
.art-t .art-info{display:flex;flex-direction:column;gap:0}
.art-t .art-name{font-weight:700;font-size:11px;color:#1e293b;max-width:140px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.art-t .art-subj{font-size:9px;color:#64748b}
.art-t .art-nm{font-size:8px;color:#94a3b8;font-family:monospace}

/* Summary / totals row */
.art-t tfoot td{background:#f1f5f9;font-weight:700;font-size:12px;
  border-top:2px solid #cbd5e1;padding:7px 8px;color:#1e293b}
</style>
"""

# ── Header row ─────────────────────────────────────────────────
hdr = "<tr>"
hdr += "<th>#</th><th>Артикул</th><th>Остаток</th>"
hdr += "<th>Ост.<br>динамика</th><th>Заказы, &#8381;</th><th>Заказы<br>в день</th>"
for dc in daily_date_cols:
    tot = day_totals.get(dc, "")
    hdr += f'<th class="day-h">{dc}<br><span class="dv">{tot}</span></th>'
hdr += "<th>Динамика</th><th>Выкуп</th>"
hdr += "<th>Цена<br>до СПП</th><th>Цена<br>после</th><th>СПП</th>"
if has_finance:
    hdr += "<th>Комиссия</th><th>Логистика</th>"
hdr += "<th>Реклама</th><th>ДРР</th>"
hdr += "<th>Прочие<br>услуги</th><th>Прибыль</th><th>Маржа</th>"
hdr += "</tr>"

# ── Data rows ──────────────────────────────────────────────────
rows = ""
for idx, (_, row) in enumerate(display.iterrows(), start=start_idx + 1):
    nm = int(row["nm_id"]) if row["nm_id"] else 0
    photo_url = _wb_photo_url(nm) if nm else ""
    stk_hist = row.get("stock_history", [])
    ord_dyn = row.get("orders_dynamics", [])
    profit = float(row.get("article_profit", 0))
    pcls = "pos" if profit > 0 else ("neg" if profit < 0 else "")
    margin = int(row.get("margin_pct", 0))
    mcls = "pos" if margin > 0 else ("neg" if margin < 0 else "")

    subj = row.get("subject", "")
    art = row.get("supplier_article", "")

    tr = "<tr>"
    tr += f'<td class="rn">{idx}</td>'
    tr += (
        f'<td><div class="art-card">'
        f'<img src="{photo_url}" class="photo" loading="lazy">'
        f'<div class="art-info">'
        f'<span class="art-name" title="{art}">{art}</span>'
        f'<span class="art-subj">{subj}</span>'
        f'<span class="art-nm">{nm}</span>'
        f'</div></div></td>'
    )
    tr += f'<td class="num">{int(row.get("stock_qty",0))}</td>'
    tr += f'<td class="ctr">{_sparkline(stk_hist if isinstance(stk_hist, list) else [])}</td>'
    tr += f'<td class="num">{_fmt(row.get("orders_amount", 0))}</td>'
    tr += f'<td class="ctr">{row.get("orders_speed", 0):.1f}</td>'
    for dc in daily_date_cols:
        v = int(row.get(dc, 0))
        cls = "day hv" if v > 0 else "day"
        tr += f'<td class="{cls}">{v if v > 0 else ""}</td>'
    tr += f'<td class="ctr">{_barchart(ord_dyn if isinstance(ord_dyn, list) else [])}</td>'
    tr += f'<td class="ctr">{_fmtp(row.get("buyout_pct", 0))}</td>'
    tr += f'<td class="num">{_fmt(row.get("avg_price_before_spp", 0))}</td>'
    tr += f'<td class="num">{_fmt(row.get("avg_price_after_spp", 0))}</td>'
    tr += f'<td class="ctr">{_fmtp1(row.get("spp_pct", 0))}</td>'
    if has_finance:
        tr += f'<td class="num">{_fmt(row.get("fin_commission", 0))}</td>'
        tr += f'<td class="num">{_fmt(row.get("fin_logistics", 0))}</td>'
    tr += f'<td class="num">{_fmt(row.get("ads_spend", 0))}</td>'
    tr += f'<td class="ctr">{_fmtp1(row.get("ads_share_pct", 0))}</td>'
    tr += f'<td class="num">{_fmt(row.get("other_services", 0))}</td>'
    tr += f'<td class="num {pcls}">{_fmt(profit)}</td>'
    tr += f'<td class="ctr {mcls}">{_fmtp(margin)}</td>'
    tr += "</tr>"
    rows += tr

# ── Totals row (footer) ────────────────────────────────────────
tot_orders_amt = display["orders_amount"].sum()
tot_stock = int(display["stock_qty"].sum())
tot_ads = display["ads_spend"].sum()
tot_profit = display["article_profit"].sum()
tot_revenue = display["net_revenue"].sum()
tot_pcls = "pos" if tot_profit > 0 else ("neg" if tot_profit < 0 else "")
tot_margin = int(round(tot_profit / tot_revenue * 100)) if tot_revenue else 0
tot_mcls = "pos" if tot_margin > 0 else ("neg" if tot_margin < 0 else "")

ftr = "<tr>"
ftr += '<td></td><td><b>Итого</b></td>'
ftr += f'<td class="num">{tot_stock}</td>'
ftr += '<td></td>'
ftr += f'<td class="num">{_fmt(tot_orders_amt)}</td>'
ftr += '<td></td>'
for dc in daily_date_cols:
    tot_d = int(display[dc].sum()) if dc in display.columns else 0
    ftr += f'<td class="ctr">{tot_d if tot_d else ""}</td>'
ftr += '<td></td><td></td><td></td><td></td><td></td>'
if has_finance:
    ftr += f'<td class="num">{_fmt(display["fin_commission"].sum())}</td>'
    ftr += f'<td class="num">{_fmt(display["fin_logistics"].sum())}</td>'
ftr += f'<td class="num">{_fmt(tot_ads)}</td>'
ftr += '<td></td><td></td>'
ftr += f'<td class="num {tot_pcls}">{_fmt(tot_profit)}</td>'
ftr += f'<td class="ctr {tot_mcls}">{_fmtp(tot_margin)}</td>'
ftr += "</tr>"

html = (
    f'{TABLE_CSS}<div class="art-wrap"><table class="art-t">'
    f'<thead>{hdr}</thead><tbody>{rows}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>'
)

st.markdown(html, unsafe_allow_html=True)

st.caption(f"Показано {start_idx + 1}–{end_idx} из {total_rows}")

# ── CSV export ─────────────────────────────────────────────────

export = r.drop(columns=["photo", "stock_history", "orders_dynamics"], errors="ignore")
st.download_button(
    "📥 Скачать CSV",
    export.to_csv(index=False).encode("utf-8-sig"),
    "article_report.csv",
    "text/csv",
)
