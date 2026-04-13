"""Неделя к неделе — сравнение двух смежных недель по артикулам."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from marts import fetch_dataframe, DASHBOARD_DETAIL_QUERY, ORDERS_DAILY_AMOUNT_QUERY
from styles import inject_global_styles

inject_global_styles()
st.title("🔄 Неделя к неделе")

# ── Helpers ───────────────────────────────────────────────────

def _fmt(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:,.0f}".replace(",", " ")

def _fmtp(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:.1f}%"

def _delta_cls(v):
    if pd.isna(v) or v == 0:
        return ""
    return "pos" if v > 0 else "neg"

def _delta_fmt(v):
    if pd.isna(v) or v == 0:
        return ""
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f}%"

# ── Week selector ─────────────────────────────────────────────

today = date.today()
# Calculate last 8 week boundaries (Monday-based)
weeks = []
d = today - timedelta(days=today.weekday())  # this Monday
for i in range(8):
    w_start = d - timedelta(weeks=i + 1)
    w_end = w_start + timedelta(days=6)
    label = f"{w_start.strftime('%d.%m')}–{w_end.strftime('%d.%m.%Y')}"
    weeks.append((label, w_start, w_end))

with st.sidebar:
    st.header("Сравнение")
    sel_curr = st.selectbox("Текущая неделя", range(len(weeks)),
                            format_func=lambda i: weeks[i][0], index=0)
    sel_prev = st.selectbox("Предыдущая неделя", range(len(weeks)),
                            format_func=lambda i: weeks[i][0], index=1)

curr_label, curr_start, curr_end = weeks[sel_curr]
prev_label, prev_start, prev_end = weeks[sel_prev]

# ── Load data ─────────────────────────────────────────────────

# Load both weeks of sales data
all_start = min(curr_start, prev_start)
all_end = max(curr_end, prev_end)
params = {"d_from": str(all_start), "d_to": str(all_end)}

sales = fetch_dataframe(DASHBOARD_DETAIL_QUERY, params)
orders = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)

if sales.empty and orders.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Aggregate by article × week ──────────────────────────────

def agg_week(df, d_from, d_to, date_col="sales_date"):
    if df.empty:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= pd.Timestamp(d_from)) & (df[date_col] <= pd.Timestamp(d_to))
    w = df[mask].copy()
    if w.empty:
        return pd.DataFrame()
    return (
        w.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            sales_count=("sales_count", "sum"),
            returns_count=("returns_count", "sum"),
            net_revenue=("net_revenue", "sum"),
            profit_amount=("profit_amount", "sum"),
            commission_amount=("commission_amount", "sum"),
            orders_count=("orders_count", "sum"),
            avg_price_before_spp=("avg_price_before_spp", "mean"),
        )
        .reset_index()
    )

def agg_orders_week(df, d_from, d_to):
    if df.empty:
        return pd.DataFrame()
    df["order_date"] = pd.to_datetime(df["order_date"])
    mask = (df["order_date"] >= pd.Timestamp(d_from)) & (df["order_date"] <= pd.Timestamp(d_to))
    w = df[mask].copy()
    if w.empty:
        return pd.DataFrame()
    return (
        w.groupby(["nm_id", "supplier_article"])
        .agg(orders_amount=("orders_amount", "sum"), ord_count=("orders_count", "sum"))
        .reset_index()
    )

curr_s = agg_week(sales, curr_start, curr_end)
prev_s = agg_week(sales, prev_start, prev_end)
curr_o = agg_orders_week(orders, curr_start, curr_end)
prev_o = agg_orders_week(orders, prev_start, prev_end)

# Merge sales + orders for each week
def merge_so(s, o):
    if s.empty and o.empty:
        return pd.DataFrame(columns=["nm_id", "supplier_article"])
    if s.empty:
        return o
    if o.empty:
        return s
    return s.merge(o, on=["nm_id", "supplier_article"], how="outer")

curr = merge_so(curr_s, curr_o)
prev = merge_so(prev_s, prev_o)

# Merge current and previous
if curr.empty and prev.empty:
    st.info("Нет данных для сравнения")
    st.stop()

keys = ["nm_id", "supplier_article"]
merged = curr.merge(prev, on=keys, how="outer", suffixes=("_curr", "_prev"))

# Fill meta from either side
for col in ["subject", "brand"]:
    c = f"{col}_curr"
    p = f"{col}_prev"
    if c in merged.columns and p in merged.columns:
        merged[col] = merged[c].fillna(merged[p])
        merged.drop(columns=[c, p], inplace=True)

# Fill NaN with 0 for numeric cols
num_cols = [c for c in merged.columns if c not in keys + ["subject", "brand"]]
merged[num_cols] = merged[num_cols].fillna(0)

# ── Calculate deltas ──────────────────────────────────────────

metric_pairs = [
    ("orders_amount", "Заказы ₽"),
    ("ord_count", "Заказы шт"),
    ("sales_count", "Продажи шт"),
    ("net_revenue", "Выручка"),
    ("profit_amount", "Прибыль"),
    ("commission_amount", "Комиссия"),
]

for col, _ in metric_pairs:
    cc = f"{col}_curr"
    pc = f"{col}_prev"
    if cc in merged.columns and pc in merged.columns:
        merged[f"{col}_delta"] = np.where(
            merged[pc] != 0,
            ((merged[cc] - merged[pc]) / merged[pc].abs() * 100).round(1),
            np.where(merged[cc] > 0, 100.0, 0.0),
        )
    else:
        merged[f"{col}_delta"] = 0

# Sort by current orders amount desc
sort_col = "orders_amount_curr" if "orders_amount_curr" in merged.columns else "net_revenue_curr"
merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)

# ── Summary KPIs ──────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
for mc, label, col_obj in [
    ("orders_amount", "Заказы ₽", c1),
    ("net_revenue", "Выручка", c2),
    ("profit_amount", "Прибыль", c3),
    ("sales_count", "Продажи шт", c4),
]:
    cc = f"{mc}_curr"
    pc = f"{mc}_prev"
    cv = merged[cc].sum() if cc in merged.columns else 0
    pv = merged[pc].sum() if pc in merged.columns else 0
    d = round((cv - pv) / pv * 100, 1) if pv != 0 else 0
    col_obj.metric(label, _fmt(cv), f"{d:+.1f}%")

st.divider()
st.caption(f"Текущая: **{curr_label}** vs Предыдущая: **{prev_label}**")

# ── HTML table ────────────────────────────────────────────────

TABLE_CSS = """
<style>
.wow-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin:1rem 0;border:1px solid #e2e8f0}
.wow{border-collapse:collapse;width:100%;font-size:11px;font-family:Inter,system-ui,sans-serif;
  background:#fff;color:#1e293b}
.wow th{background:#f1f5f9;padding:6px 8px;border-bottom:2px solid #cbd5e1;
  border-right:1px solid #e2e8f0;font-weight:600;font-size:10px;color:#475569;
  text-align:center;white-space:nowrap;vertical-align:bottom}
.wow td{padding:4px 6px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  white-space:nowrap;font-size:11px}
.wow tbody tr:nth-child(even){background:#fafbfc}
.wow tbody tr:hover{background:#eef2ff}
.wow .num{text-align:right}.wow .ctr{text-align:center}
.wow .pos{color:#16a34a;font-weight:700}.wow .neg{color:#dc2626;font-weight:700}
.wow .delta{font-size:10px;padding:2px 5px;border-radius:4px;display:inline-block}
.wow .delta.up{background:#dcfce7;color:#16a34a}
.wow .delta.dn{background:#fee2e2;color:#dc2626}
.wow .delta.eq{background:#f1f5f9;color:#64748b}
.wow .rn{color:#94a3b8;text-align:center}
</style>
"""

# Build header: Article | Metric Curr | Metric Prev | Δ% for each metric
hdr = '<tr><th>#</th><th>Артикул</th><th>Предмет</th>'
for _, label in metric_pairs:
    hdr += f'<th>{label}<br><small>тек.</small></th><th>{label}<br><small>пред.</small></th><th>Δ%</th>'
hdr += '</tr>'

rows_html = ""
for idx, (_, r) in enumerate(merged.head(100).iterrows(), 1):
    art = r.get("supplier_article", "")
    subj = r.get("subject", "")
    row = f'<tr><td class="rn">{idx}</td><td><b>{art}</b></td><td>{subj}</td>'
    for col, _ in metric_pairs:
        cc = f"{col}_curr"
        pc = f"{col}_prev"
        dc = f"{col}_delta"
        cv = float(r.get(cc, 0))
        pv = float(r.get(pc, 0))
        dv = float(r.get(dc, 0))
        dcls = "up" if dv > 0 else ("dn" if dv < 0 else "eq")
        row += f'<td class="num">{_fmt(cv)}</td>'
        row += f'<td class="num">{_fmt(pv)}</td>'
        row += f'<td class="ctr"><span class="delta {dcls}">{_delta_fmt(dv)}</span></td>'
    row += '</tr>'
    rows_html += row

html = f'{TABLE_CSS}<div class="wow-wrap"><table class="wow"><thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>'
st.markdown(html, unsafe_allow_html=True)
st.caption(f"Показано {min(100, len(merged))} из {len(merged)} артикулов")

st.download_button(
    "📥 Скачать CSV",
    merged.to_csv(index=False).encode("utf-8-sig"),
    "week_comparison.csv",
    "text/csv",
)
