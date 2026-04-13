import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from marts import fetch_dataframe, DASHBOARD_DETAIL_QUERY, FINANCE_DAILY_QUERY, ORDERS_DAILY_AMOUNT_QUERY, EXTRA_EXPENSES_QUERY, ADS_DAILY_QUERY, default_date_range
from styles import inject_global_styles, format_currency, format_pct

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()
st.title("📈 KPI-дашборд")

# ── Sidebar: date filters ────────────────────────────────────
with st.sidebar:
    st.header("Фильтры")
    d_def = default_date_range()
    d_from = st.date_input("Дата начала", value=d_def[0])
    d_to = st.date_input("Дата окончания", value=d_def[1])

params = {"d_from": str(d_from), "d_to": str(d_to)}
raw = fetch_dataframe(DASHBOARD_DETAIL_QUERY, params)

if raw.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Sidebar: entity filters ─────────────────────────────────
with st.sidebar:
    brands = sorted(raw["brand"].dropna().loc[raw["brand"] != ""].unique())
    sel_brands = st.multiselect("Бренд", brands, default=[])

    subjects = sorted(raw["subject"].dropna().unique())
    sel_subjects = st.multiselect("Предмет", subjects, default=[])

    articles = sorted(raw["supplier_article"].dropna().unique())
    sel_articles = st.multiselect("Артикул поставщика", articles, default=[])

# ── Apply filters ────────────────────────────────────────────
df = raw.copy()
if sel_brands:
    df = df[df["brand"].isin(sel_brands)]
if sel_subjects:
    df = df[df["subject"].isin(sel_subjects)]
if sel_articles:
    df = df[df["supplier_article"].isin(sel_articles)]

if df.empty:
    st.warning("Нет данных по выбранным фильтрам")
    st.stop()

# ══════════════════════════════════════════════════════════════
#  FINANCE DATA (from WB Financial Reports API)
# ══════════════════════════════════════════════════════════════
try:
    fin_raw = fetch_dataframe(FINANCE_DAILY_QUERY, params)
except Exception:
    fin_raw = pd.DataFrame()

has_finance = not fin_raw.empty

# Apply same entity filters to finance data
if has_finance:
    fin = fin_raw.copy()
    if sel_brands:
        fin = fin[fin["brand"].isin(sel_brands)]
    if sel_subjects:
        fin = fin[fin["subject"].isin(sel_subjects)]
    if sel_articles:
        fin = fin[fin["supplier_article"].isin(sel_articles)]
    has_finance = not fin.empty

# ══════════════════════════════════════════════════════════════
#  ORDERS DATA (from mart.orders_daily)
# ══════════════════════════════════════════════════════════════
try:
    ord_raw = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)
except Exception:
    ord_raw = pd.DataFrame()

has_orders = not ord_raw.empty
if has_orders:
    ord_df = ord_raw.copy()
    if sel_brands:
        ord_df = ord_df[ord_df["brand"].isin(sel_brands)]
    if sel_subjects:
        ord_df = ord_df[ord_df["subject"].isin(sel_subjects)]
    if sel_articles:
        ord_df = ord_df[ord_df["supplier_article"].isin(sel_articles)]
    has_orders = not ord_df.empty

# ══════════════════════════════════════════════════════════════
#  ADS DATA (from mart.ads_daily — WB Promotion API)
# ══════════════════════════════════════════════════════════════
try:
    ads_raw = fetch_dataframe(ADS_DAILY_QUERY, params)
except Exception:
    ads_raw = pd.DataFrame()

has_ads = not ads_raw.empty
if has_ads:
    ads_df = ads_raw.copy()
    if sel_articles:
        ads_df = ads_df[ads_df["supplier_article"].isin(sel_articles)]
    has_ads = not ads_df.empty
    ads_total_spend = float(ads_df["spend_amount"].sum()) if has_ads else 0.0
else:
    ads_total_spend = 0.0

# ══════════════════════════════════════════════════════════════
#  KPI SUMMARY
# ══════════════════════════════════════════════════════════════
gross_rev = float(df["gross_revenue"].sum())
net_rev = float(df["net_revenue"].sum())
commission = float(df["commission_amount"].sum())
cost = float(df["cost_amount"].sum())
orders = int(df["orders_count"].sum())
sales = int(df["sales_count"].sum())
returns = int(df["returns_count"].sum())

# Extra expenses — correct value from dict (mart value is inflated by LATERAL join)
try:
    extra = float(fetch_dataframe(EXTRA_EXPENSES_QUERY, params)["total"].iloc[0])
except Exception:
    extra = float(df["extra_expenses_amount"].sum())

# Finance breakdown (from WB Financial Reports API)
if has_finance:
    fin_sales_amt = float(fin["sales_amount"].sum())
    fin_returns_amt = float(fin["returns_amount"].sum())
    fin_realizacia = fin_sales_amt - fin_returns_amt
    fin_commission = float(fin["commission_amount"].sum())
    fin_logistics = float(fin["logistics_amount"].sum())
    fin_storage = float(fin["storage_amount"].sum())
    fin_penalty = float(fin["penalty_amount"].sum())
    fin_acceptance = float(fin["acceptance_amount"].sum())
    fin_deduction = float(fin["deduction_amount"].sum())
    fin_total_services = fin_commission + fin_logistics + fin_storage + fin_penalty + fin_acceptance + fin_deduction
    fin_payout = fin_realizacia - fin_total_services

    # Recalculate cost per-nm_id using finance NET sales (sales - returns)
    art_cost = df.groupby("nm_id").agg(
        _cost=("cost_amount", "sum"), _sales=("sales_count", "sum"))
    art_cost["unit_cost"] = art_cost["_cost"] / art_cost["_sales"].replace(0, 1)
    fin_by_art = fin.groupby("nm_id").agg(
        _sales=("sales_count", "sum"), _returns=("returns_count", "sum"))
    fin_by_art["net_sales"] = (fin_by_art["_sales"] - fin_by_art["_returns"]).clip(lower=0)
    matched = fin_by_art[["net_sales"]].join(art_cost["unit_cost"], how="left").fillna(0)
    cost = float((matched["net_sales"] * matched["unit_cost"]).sum())
else:
    fin_sales_amt = fin_returns_amt = fin_realizacia = 0.0
    fin_commission = fin_logistics = fin_storage = 0.0
    fin_penalty = fin_acceptance = fin_deduction = 0.0
    fin_total_services = commission
    fin_payout = 0.0

payout = fin_payout if has_finance else (net_rev - commission)

# Tax: 6% УСН applied to taxable profit
tax_rate = float(df["tax_amount"].sum()) / net_rev if net_rev > 0 else 0.06
# When has_finance: ads already inside deduction (part of payout), don't subtract again
if has_finance:
    tax = tax_rate * max(payout - cost, 0)
    op_profit = payout - cost - tax - extra
    margin_pct = (op_profit / fin_realizacia * 100) if fin_realizacia else 0
    roi_pct = (op_profit / cost * 100) if cost else 0
else:
    tax = tax_rate * max(payout - ads_total_spend - cost, 0)
    op_profit = payout - ads_total_spend - cost - tax - extra
    margin_pct = (op_profit / net_rev * 100) if net_rev else 0
    roi_pct = (op_profit / cost * 100) if cost else 0
avg_check = net_rev / sales if sales else 0

# ── Helpers ──────────────────────────────────────────────────
RUB = "&#8381;"
DOT = "&#9679;"

def _fmt(v):
    return f"{v:,.0f}".replace(",", " ")

def _row(color, label, value, pct):
    """One detail row: colored dot + label + value + grey percentage."""
    return (
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin:2px 0;'>"
        f"  <span><span style='color:{color}'>{DOT}</span> {label}</span>"
        f"  <span><b>{_fmt(value)}</b> <span style='color:#94a3b8'>{pct:.0f}%</span></span>"
        f"</div>"
    )

def _multi_bar(segments):
    """Multi-color progress bar. segments = [(pct, color), ...]"""
    parts = "".join(
        f"<div style='width:{p:.1f}%; background:{c};'></div>" for p, c in segments
    )
    return (
        f"<div style='background:#e2e8f0; border-radius:999px; height:7px;"
        f" margin:0.5rem 0; overflow:hidden; display:flex;'>{parts}</div>"
    )

import json as _json

def _pct_change(values):
    if not values or len(values) < 2:
        return 0
    prev, curr = values[-2], values[-1]
    return ((curr - prev) / prev * 100) if prev else 0

def _spark_card(idx, title, value, daily_values, daily_labels, color, date_str, expense=False):
    pct = _pct_change(daily_values)
    # For expense cards: growth is bad (red), decline is good (green)
    if expense:
        pct_color = "#ef4444" if pct >= 0 else "#22c55e"
    else:
        pct_color = "#22c55e" if pct >= 0 else "#ef4444"
    sign = "+" if pct >= 0 else ""
    data_json = _json.dumps([
        {"d": lbl, "v": round(v, 0)} for lbl, v in zip(daily_labels, daily_values)
    ]) if daily_values and daily_labels else "[]"
    return (
        f'<div style="background:white;border-radius:14px;padding:1rem 1.2rem;'
        f'box-shadow:0 4px 16px rgba(15,23,42,0.07);" class="spark-card" data-idx="{idx}"'
        f" data-points='{data_json}' data-color='{color}' data-title='{title}'>"
        f'<div style="font-size:0.95rem;color:#1e293b;font-weight:700;">{title}</div>'
        f'<div style="font-size:0.72rem;color:#94a3b8;">{date_str}</div>'
        f'<div style="font-size:1.7rem;font-weight:700;color:#0f172a;margin:0.25rem 0;white-space:nowrap;">'
        f'{_fmt(value)}</div>'
        f'<div style="font-size:0.72rem;color:{pct_color};font-weight:500;">'
        f'{sign}{pct:.0f}% динамика за день</div>'
        f'<div class="chart-area" style="position:relative;height:55px;margin-top:6px;">'
        f'<canvas id="canvas_{idx}" style="width:100%;height:55px;display:block;"></canvas>'
        f'<div class="tip" id="tip_{idx}" style="display:none;position:absolute;top:-8px;'
        f'background:#1e293b;color:white;font-size:0.68rem;padding:3px 8px;border-radius:6px;'
        f'white-space:nowrap;pointer-events:none;z-index:10;transform:translateX(-50%);"></div>'
        f'</div></div>'
    )

SPARK_JS = """
<script>
document.querySelectorAll('.spark-card').forEach(card => {
    const pts = JSON.parse(card.dataset.points);
    if (!pts.length) return;
    const color = card.dataset.color;
    const title = card.dataset.title;
    const canvas = card.querySelector('canvas');
    const tip = card.querySelector('.tip');
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth, H = 55;
    canvas.width = W * 2; canvas.height = H * 2;
    ctx.scale(2, 2);
    const vals = pts.map(p => p.v);
    const mx = Math.max(...vals), mn = Math.min(...vals);
    const rng = mx - mn || 1;
    const xs = [], ys = [];
    vals.forEach((v, i) => {
        xs.push(i / (vals.length - 1) * W);
        ys.push(H - (v - mn) / rng * H * 0.8 - H * 0.08);
    });
    // area
    ctx.beginPath();
    ctx.moveTo(0, H);
    xs.forEach((x, i) => ctx.lineTo(x, ys[i]));
    ctx.lineTo(W, H); ctx.closePath();
    ctx.fillStyle = color + '1a'; ctx.fill();
    // line
    ctx.beginPath();
    xs.forEach((x, i) => i === 0 ? ctx.moveTo(x, ys[i]) : ctx.lineTo(x, ys[i]));
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    // hover
    canvas.addEventListener('mousemove', e => {
        const rect = canvas.getBoundingClientRect();
        const mx2 = (e.clientX - rect.left);
        let closest = 0, minD = Infinity;
        xs.forEach((x, i) => { const d = Math.abs(mx2 - x); if (d < minD) { minD = d; closest = i; }});
        tip.style.display = 'block';
        tip.style.left = xs[closest] + 'px';
        tip.innerHTML = pts[closest].d + '<br><b>' + title + ': ' +
            pts[closest].v.toLocaleString('ru-RU', {maximumFractionDigits:0}) + '</b>';
    });
    canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
</script>
"""

# ── Derived percentages ──────────────────────────────────────
cost_pct = (cost / net_rev * 100) if net_rev else 0
tax_pct = (tax / net_rev * 100) if net_rev else 0
extra_pct = (extra / net_rev * 100) if net_rev else 0
total_costs = cost + tax + extra
total_costs_pct = (total_costs / net_rev * 100) if net_rev else 0

# Sales revenue (from non-return rows) and returns revenue
sales_amount = float(df.loc[df["sales_count"] > 0, "net_revenue"].sum())
returns_amount = abs(float(df.loc[df["returns_count"] > 0, "net_revenue"].sum()))

st.markdown("### Итоговые показатели за выбранный период")

# ── Card styles ──────────────────────────────────────────────
CARD = (
    "background:white; border-radius:14px; padding:1.2rem 1.4rem;"
    " box-shadow:0 4px 16px rgba(15,23,42,0.07); min-height:170px;"
)

# ── Build cards content ───────────────────────────────────────
if has_finance:
    # Use finance data for Реализация card too
    real_base = fin_realizacia if fin_realizacia else 1
    ads_pct = (ads_total_spend / real_base * 100)
    # Deduction already includes ads (WB Promotion) + reviews + other deductions.
    # Split for display: "Реклама" from ads API, rest into "Остальные"
    fin_deduction_ex_ads = max(fin_deduction - ads_total_spend, 0)
    fin_other = fin_storage + fin_penalty + fin_acceptance + fin_deduction_ex_ads
    # Total services already includes full deduction (and thus ads) — no extra add
    fin_total_with_ads = fin_total_services

    comm_pct = (fin_commission / real_base * 100)
    log_pct = (fin_logistics / real_base * 100)
    other_pct = (fin_other / real_base * 100)
    svc_total_pct = (fin_total_with_ads / real_base * 100)

    # Реализация card values (from finance)
    card_realizacia = fin_realizacia
    card_sales = fin_sales_amt
    card_returns = fin_returns_amt
    sales_pct = (fin_sales_amt / max(fin_sales_amt + fin_returns_amt, 1) * 100)
    returns_pct = (fin_returns_amt / max(fin_sales_amt + fin_returns_amt, 1) * 100)

    services_detail = (
        _row("#f59e0b", "Комиссия", fin_commission, comm_pct)
        + _row("#ef4444", "Логистика", fin_logistics, log_pct)
        + (_row("#8b5cf6", "Реклама", ads_total_spend, ads_pct) if ads_total_spend else "")
        + (_row("#10b981", "Остальные", fin_other, other_pct) if fin_other else "")
    )
    services_bar = _multi_bar([
        (comm_pct, "#f59e0b"),
        (log_pct, "#ef4444"),
        (ads_pct, "#8b5cf6"),
        (other_pct, "#10b981"),
    ])
    services_note = ""
    services_total = fin_total_with_ads
else:
    comm_pct = (commission / net_rev * 100) if net_rev else 0
    svc_total_pct = comm_pct
    card_realizacia = net_rev
    card_sales = sales_amount
    card_returns = returns_amount
    sales_pct = (sales / max(sales + returns, 1) * 100)
    returns_pct = (returns / max(sales + returns, 1) * 100)
    services_detail = _row("#f59e0b", "Комиссия", commission, comm_pct)
    services_bar = _multi_bar([(comm_pct, "#f59e0b")])
    services_note = (
        "<div style='font-size:0.7rem; color:#94a3b8; margin-top:4px;'>"
        "Логистика, хранение &#8212; загрузите фин. отчёт WB"
        "</div>"
    )
    services_total = commission

kpi_html = f"""
<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:1rem; margin-bottom:1rem;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">

  <!-- РЕАЛИЗАЦИЯ -->
  <div style="{CARD}">
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;">Реализация</div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {_fmt(card_realizacia)} {RUB}
    </div>
    {_multi_bar([
        (sales_pct, "#22c55e"),
        (returns_pct, "#dc2626"),
    ])}
    <div style="font-size:0.8rem; color:#475569; line-height:1.7;">
      {_row("#22c55e", "Продажи", card_sales, card_sales / max(card_realizacia, 1) * 100)}
      {_row("#dc2626", "Возвраты", card_returns, card_returns / max(card_realizacia, 1) * 100)}
    </div>
  </div>

  <!-- УСЛУГИ WB -->
  <div style="{CARD}">
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;">
      Услуги WB <span style="color:#94a3b8; font-size:0.78rem;">{svc_total_pct:.0f}%</span>
    </div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {_fmt(services_total)} {RUB}
    </div>
    {services_bar}
    <div style="font-size:0.8rem; color:#475569; line-height:1.7;">
      {services_detail}
    </div>
    {services_note}
  </div>

  <!-- НАЛОГИ И ЗАТРАТЫ -->
  <div style="{CARD}">
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;">
      Налоги и затраты <span style="color:#94a3b8; font-size:0.78rem;">{total_costs_pct:.0f}%</span>
    </div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {_fmt(total_costs)} {RUB}
    </div>
    {_multi_bar([
        (cost_pct, "#f97316"),
        (tax_pct, "#ef4444"),
        (extra_pct, "#a855f7"),
    ])}
    <div style="font-size:0.8rem; color:#475569; line-height:1.7;">
      {_row("#f97316", "Себестоимость", cost, cost_pct)}
      {_row("#ef4444", "Налог", tax, tax_pct)}
      {_row("#a855f7", "Доп. расходы", extra, extra_pct) if extra else ""}
    </div>
  </div>

  <!-- ОПЕРАЦИОННАЯ ПРИБЫЛЬ -->
  <div style="{CARD}">
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;">Операционная прибыль</div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {_fmt(op_profit)} {RUB}
    </div>
    <div style="font-size:0.85rem; color:#475569; line-height:2; margin-top:0.4rem;">
      <div style="display:flex; justify-content:space-between;">
        <span>Маржинальность</span> <b>{margin_pct:.1f}%</b>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Рентабельность</span> <b>{roi_pct:.1f}%</b>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Средний чек</span> <b>{_fmt(avg_check)} {RUB}</b>
      </div>
    </div>
  </div>

</div>
"""
st.html(kpi_html)

# ══════════════════════════════════════════════════════════════
#  SPARKLINE CARDS (5 mini-cards with daily charts)
# ══════════════════════════════════════════════════════════════
period_days = (d_to - d_from).days

def _resample(series, labels, period_days):
    """Resample daily data to weeks or months if period is long."""
    if period_days <= 60:
        return series, labels
    df_tmp = pd.DataFrame({"val": series, "dt": pd.to_datetime(labels)})
    if period_days > 365:
        grp = df_tmp.set_index("dt").resample("MS")["val"].sum()
        return grp.tolist(), [d.strftime("%b %Y") for d in grp.index]
    else:
        grp = df_tmp.set_index("dt").resample("W-MON")["val"].sum()
        return grp.tolist(), [d.strftime("%d.%m") for d in grp.index]

if has_orders:
    ord_by_day = ord_df.groupby("order_date")["orders_amount"].sum().sort_index()
    spark_orders_raw = ord_by_day.tolist()
    spark_orders_lbl = [str(d)[:10] for d in ord_by_day.index]
    total_orders_amt = float(ord_by_day.sum())
    spark_orders, spark_orders_lbl = _resample(spark_orders_raw, spark_orders_lbl, period_days)
else:
    spark_orders = spark_orders_lbl = []
    total_orders_amt = 0

if has_finance:
    fin_by_day = fin.groupby("report_date").agg(
        sales_amount=("sales_amount", "sum"),
        logistics_amount=("logistics_amount", "sum"),
        deduction_amount=("deduction_amount", "sum"),
        commission_amount=("commission_amount", "sum"),
        storage_amount=("storage_amount", "sum"),
        penalty_amount=("penalty_amount", "sum"),
        acceptance_amount=("acceptance_amount", "sum"),
    ).sort_index()
    fin_by_day["total_services"] = (
        fin_by_day["commission_amount"] + fin_by_day["logistics_amount"]
        + fin_by_day["storage_amount"] + fin_by_day["penalty_amount"]
        + fin_by_day["acceptance_amount"] + fin_by_day["deduction_amount"]
    )
    raw_lbl = [str(d)[:10] for d in fin_by_day.index]
    spark_sales, s_lbl = _resample(fin_by_day["sales_amount"].tolist(), raw_lbl, period_days)
    spark_logistics, l_lbl = _resample(fin_by_day["logistics_amount"].tolist(), raw_lbl, period_days)
    spark_services, sv_lbl = _resample(fin_by_day["total_services"].tolist(), raw_lbl, period_days)
else:
    spark_sales = spark_logistics = spark_services = []
    s_lbl = l_lbl = sv_lbl = []

# Ads sparkline — from mart.ads_daily (Promotion API)
if has_ads:
    ads_by_day = ads_df.groupby("ads_date")["spend_amount"].sum().sort_index()
    spark_ads_raw = ads_by_day.tolist()
    spark_ads_lbl_raw = [str(d)[:10] for d in ads_by_day.index]
    spark_ads, a_lbl = _resample(spark_ads_raw, spark_ads_lbl_raw, period_days)
else:
    spark_ads = a_lbl = []

end_fmt = f"{d_to.day:02d}.{d_to.month:02d}.{d_to.year}"
_font = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
spark_html = (
    f"<div style='{_font}'>"
    f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem;'>"
    + _spark_card(0, "Заказы", total_orders_amt, spark_orders, spark_orders_lbl, "#f97316", end_fmt)
    + _spark_card(1, "Продажи", fin_sales_amt, spark_sales, s_lbl, "#22c55e", end_fmt)
    + _spark_card(2, "Логистика", fin_logistics, spark_logistics, l_lbl, "#ef4444", end_fmt, expense=True)
    + "</div>"
    f"<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;'>"
    + _spark_card(3, "Реклама", ads_total_spend, spark_ads, a_lbl, "#8b5cf6", end_fmt, expense=True)
    + _spark_card(4, "Все услуги", fin_total_services, spark_services, sv_lbl, "#3b82f6", end_fmt, expense=True)
    + "</div></div>"
    + SPARK_JS
)
components.html(spark_html, height=400)

# ══════════════════════════════════════════════════════════════
#  MONTHLY CHARTS
# ══════════════════════════════════════════════════════════════
df["sales_date"] = pd.to_datetime(df["sales_date"])
df["month"] = df["sales_date"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby("month").agg(
    orders_count=("orders_count", "sum"),
    sales_count=("sales_count", "sum"),
    returns_count=("returns_count", "sum"),
    gross_revenue=("gross_revenue", "sum"),
    net_revenue=("net_revenue", "sum"),
    commission_amount=("commission_amount", "sum"),
    cost_amount=("cost_amount", "sum"),
    operating_profit_amount=("operating_profit_amount", "sum"),
).reset_index()
monthly["margin_pct"] = (
    monthly["operating_profit_amount"] / monthly["net_revenue"] * 100
).fillna(0).round(1)
monthly["avg_check"] = (
    monthly["net_revenue"] / monthly["sales_count"]
).fillna(0).round(0)
monthly["label"] = monthly["month"].dt.strftime("%b %Y")

# ── Chart 1: Orders + Sales + Avg Check ─────────────────────
st.markdown("### Заказы, продажи и средний чек по месяцам")
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(
    x=monthly["label"], y=monthly["orders_count"],
    name="Заказы", marker_color="#f97316",
    text=monthly["orders_count"], textposition="outside",
), secondary_y=False)
fig1.add_trace(go.Bar(
    x=monthly["label"], y=monthly["sales_count"],
    name="Продажи", marker_color="#3b82f6",
    text=monthly["sales_count"], textposition="outside",
), secondary_y=False)
fig1.add_trace(go.Scatter(
    x=monthly["label"], y=monthly["avg_check"],
    name="Средний чек", line=dict(color="#1e293b", width=3),
    mode="lines+markers+text",
    text=monthly["avg_check"].apply(lambda v: f"{v:,.0f}"),
    textposition="top center", textfont=dict(size=11),
), secondary_y=True)
fig1.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    hovermode="x unified", margin=dict(t=50),
)
fig1.update_yaxes(title_text="Количество", secondary_y=False)
fig1.update_yaxes(title_text="Средний чек, \u20bd", secondary_y=True)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Revenue + Operating Profit + Margin % ──────────
st.markdown("### Реализация, операционная прибыль и маржинальность")
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Bar(
    x=monthly["label"], y=monthly["net_revenue"],
    name="Реализация (нетто)", marker_color="#3b82f6",
    text=monthly["net_revenue"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
), secondary_y=False)
fig2.add_trace(go.Bar(
    x=monthly["label"], y=monthly["operating_profit_amount"],
    name="Операц. прибыль", marker_color="#f97316",
    text=monthly["operating_profit_amount"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
), secondary_y=False)
fig2.add_trace(go.Scatter(
    x=monthly["label"], y=monthly["margin_pct"],
    name="% маржинальности",
    line=dict(color="#1e293b", width=2, dash="dot"),
    mode="lines+markers+text",
    text=monthly["margin_pct"].apply(lambda v: f"{v:.1f}%"),
    textposition="top center", textfont=dict(size=11),
), secondary_y=True)
margin_max = max(monthly["margin_pct"].max() * 1.5, 10)
fig2.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    hovermode="x unified", margin=dict(t=50),
)
fig2.update_yaxes(title_text="Сумма, \u20bd", secondary_y=False)
fig2.update_yaxes(title_text="Маржинальность, %", secondary_y=True,
                  range=[0, margin_max])
st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  DAILY DYNAMICS (amounts in rubles)
# ══════════════════════════════════════════════════════════════
st.markdown("### Динамика показателей")

daily_rev = df.groupby("sales_date").agg(
    net_revenue=("net_revenue", "sum"),
    profit_amount=("profit_amount", "sum"),
).reset_index()

if has_orders:
    daily_ord = ord_df.copy()
    daily_ord["order_date"] = pd.to_datetime(daily_ord["order_date"])
    daily_ord = daily_ord.groupby("order_date")["orders_amount"].sum().reset_index()
    daily_ord = daily_ord.rename(columns={"order_date": "sales_date"})
    daily_dyn = daily_rev.merge(daily_ord, on="sales_date", how="outer").fillna(0)
else:
    daily_dyn = daily_rev.copy()
    daily_dyn["orders_amount"] = 0
daily_dyn = daily_dyn.sort_values("sales_date")

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=daily_dyn["sales_date"], y=daily_dyn["orders_amount"],
    name="Заказы", marker_color="#f97316",
))
fig3.add_trace(go.Bar(
    x=daily_dyn["sales_date"], y=daily_dyn["net_revenue"],
    name="Продажи", marker_color="#3b82f6",
))
fig3.add_trace(go.Bar(
    x=daily_dyn["sales_date"], y=daily_dyn["profit_amount"],
    name="Прибыль", marker_color="#22c55e",
))
fig3.update_layout(
    barmode="group",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Дата", yaxis_title="Сумма, \u20bd",
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    hovermode="x unified",
)
st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TOP-10 HORIZONTAL BAR CHARTS
# ══════════════════════════════════════════════════════════════
st.markdown("### Операционная прибыль: Топ-10")
tc1, tc2, tc3 = st.columns(3)

# ── Top-10 by brand ──
with tc1:
    st.markdown("**По брендам**")
    by_brand = (
        df[df["brand"].notna() & (df["brand"] != "")]
        .groupby("brand")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_b = px.bar(
        by_brand, x="operating_profit_amount", y="brand",
        orientation="h", color_discrete_sequence=["#3b82f6"],
        text=by_brand["operating_profit_amount"].apply(
            lambda v: f"{v:,.0f}".replace(",", " ")),
    )
    fig_b.update_traces(textposition="outside")
    fig_b.update_layout(
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_b, use_container_width=True)

# ── Top-10 by subject ──
with tc2:
    st.markdown("**По предметам**")
    by_subj = (
        df[df["subject"].notna()]
        .groupby("subject")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_s = px.bar(
        by_subj, x="operating_profit_amount", y="subject",
        orientation="h", color_discrete_sequence=["#22c55e"],
        text=by_subj["operating_profit_amount"].apply(
            lambda v: f"{v:,.0f}".replace(",", " ")),
    )
    fig_s.update_traces(textposition="outside")
    fig_s.update_layout(
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ── Top-10 by article ──
with tc3:
    st.markdown("**По артикулам**")
    by_art = (
        df[df["supplier_article"].notna()]
        .groupby("supplier_article")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_a = px.bar(
        by_art, x="operating_profit_amount", y="supplier_article",
        orientation="h", color_discrete_sequence=["#dc2626"],
        text=by_art["operating_profit_amount"].apply(
            lambda v: f"{v:,.0f}".replace(",", " ")),
    )
    fig_a.update_traces(textposition="outside")
    fig_a.update_layout(
        showlegend=False, height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_a, use_container_width=True)

# ── CSV download ─────────────────────────────────────────────
st.download_button(
    "📥 Скачать CSV", df.to_csv(index=False).encode("utf-8-sig"),
    "kpi_dashboard.csv", "text/csv",
)
