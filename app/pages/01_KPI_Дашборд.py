import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from marts import fetch_dataframe, DASHBOARD_DETAIL_QUERY, FINANCE_DAILY_QUERY, ORDERS_DAILY_AMOUNT_QUERY, EXTRA_EXPENSES_QUERY, ADS_DAILY_QUERY
from styles import plotly_defaults, inject_global_styles, fmt_number, date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, export_buttons
from auth import check_auth, logout

# ── Page setup ───────────────────────────────────────────────
inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📈 KPI-дашборд")

with st.expander("ℹ️ Как устроен дашборд", expanded=False):
    st.markdown(
        """
        **Блоки сверху вниз:**

        1. **KPI-карточки** (4 цветных блока) — ключевые суммы за период:
           - *Реализация* — выплата от WB по отчёту «ppvz_for_pay»
           - *Услуги WB* — логистика + хранение + штрафы + эквайринг + приёмка
           - *Налоги и затраты* — налог по ставке УСН + доп. расходы из справочника
           - *Операционная прибыль* — итоговый финансовый результат

        2. **Мини-тренды** — индикаторы с 3-колонкой числа и спарклайном
           (продажи / возвраты / выручка / прибыль / средний чек и т.д.).

        3. **Динамика** — временные ряды по дням с заливкой и hover-подписями.

        4. **Топ-5** — бренды / предметы / артикулы по прибыли.

        **Δ%** в карточках — сравнение с предыдущим периодом такой же длины.
        """
    )

# ── Filters: dates ───────────────────────────────────────────
d_from, d_to = date_filter_bar("kpi", default_days=30)

params = {"d_from": str(d_from), "d_to": str(d_to)}
raw = fetch_dataframe(DASHBOARD_DETAIL_QUERY, params)

if raw.empty:
    st.info("Нет данных за выбранный период")
    st.stop()

# ── Filters: entities ────────────────────────────────────────
brands = sorted(raw["brand"].dropna().loc[raw["brand"] != ""].unique())
subjects = sorted(raw["subject"].dropna().unique())
articles = sorted(raw["supplier_article"].dropna().unique())
_ec1, _ec2, _ec3 = st.columns(3)
with _ec1:
    sel_brands = st.multiselect("Бренд", brands, default=[])
with _ec2:
    sel_subjects = st.multiselect("Предмет", subjects, default=[])
with _ec3:
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
    fin_realizacia = fin_sales_amt - fin_returns_amt           # Реализация до СПП
    fin_retail_amt = float(fin["retail_amount"].sum())          # Реализация после СПП
    fin_commission = float(fin["commission_amount"].sum())
    fin_logistics = float(fin["logistics_amount"].sum())
    fin_storage = float(fin["storage_amount"].sum())
    fin_penalty = float(fin["penalty_amount"].sum())
    fin_acceptance = float(fin["acceptance_amount"].sum())
    fin_deduction = float(fin["deduction_amount"].sum())
    fin_acquiring = float(fin["acquiring_amount"].sum())
    fin_additional = float(fin["additional_payment_amount"].sum())
    fin_total_services = (
        fin_commission + fin_logistics + fin_storage
        + fin_penalty + fin_acceptance + fin_acquiring + fin_deduction
        - fin_additional
    )
    fin_payout = float(fin["ppvz_for_pay"].sum())               # К перечислению
    # Себестоимость — из SQL (стабильно по nm_id через LATERAL JOIN).
    cost = float(fin["cost_amount"].sum())
    # NB: ppvz_for_pay УЖЕ за минусом комиссии. Формула xlsx-методологии
    # RASK ОПИУ (совпадает с PNL_MONTHLY_QUERY):
    #   pre_tax = ppvz - logistics - storage - penalty - acceptance
    #           - deduction + additional - cost - extra_expenses
    # `deduction_amount` из WB Finance API — единый бакет (внутр. реклама
    # + отзывы + прочие удержания). `ads_spend` из Promotion API — часть
    # deduction_amount, НЕ вычитается повторно во избежание двойного счёта.
    # Эквайринг НЕ вычитается (xlsx-методология его не учитывает).
    services_no_commission = (
        fin_logistics + fin_storage + fin_penalty
        + fin_acceptance + fin_deduction - fin_additional
    )
    pre_tax_agg = (
        fin_payout - services_no_commission - cost - extra
    )
    # ── Налог ─────────────────────────────────────────────────
    # Считаем налог как GREATEST(pre_tax_agg, 0) × взвешенная ставка,
    # чтобы KPI ровно совпадал с PNL_MONTHLY (ОПИУ-пейдж).
    # Взвешенная ставка = SUM(tax_rate_pct × pre_tax_row_positive) /
    #                     SUM(pre_tax_row_positive)
    # Для однородного периода (март 2026: rate=6%) = 6%.
    if "tax_rate_pct" in fin.columns and "pre_tax_profit" in fin.columns:
        _pos = fin[fin["pre_tax_profit"] > 0]
        if not _pos.empty:
            _w = float(_pos["pre_tax_profit"].sum())
            _wrate = (
                float((_pos["tax_rate_pct"] * _pos["pre_tax_profit"]).sum()) / _w
                if _w > 0 else 0.0
            )
        else:
            _wrate = 0.0
    else:
        _wrate = 0.0
    tax = max(pre_tax_agg, 0) * _wrate / 100.0
    tax_rate_pct = _wrate
    op_profit = pre_tax_agg - tax
    # Маржинальность считаем от реализации ДО СПП (требование Расkка).
    margin_pct = (op_profit / fin_realizacia * 100) if fin_realizacia else 0
    roi_pct = (op_profit / cost * 100) if cost else 0
else:
    fin_sales_amt = fin_returns_amt = fin_realizacia = fin_retail_amt = 0.0
    fin_commission = fin_logistics = fin_storage = 0.0
    fin_penalty = fin_acceptance = fin_deduction = 0.0
    fin_acquiring = fin_additional = 0.0
    fin_total_services = commission
    fin_payout = net_rev - commission
    # Fallback: no finance_daily → только sales_daily. В этом случае мы
    # не знаем deduction_amount, поэтому используем ads_total_spend как
    # приближение (это устаревший путь; при заполненной finance_daily
    # ветка выше использует deduction_amount из WB Finance API).
    tax_rate = float(df["tax_amount"].sum()) / net_rev if net_rev > 0 else 0.06
    tax = tax_rate * max(fin_payout - ads_total_spend - cost, 0)
    op_profit = fin_payout - ads_total_spend - cost - tax - extra
    margin_pct = (op_profit / net_rev * 100) if net_rev else 0
    roi_pct = (op_profit / cost * 100) if cost else 0

payout = fin_payout
avg_check = net_rev / sales if sales else 0

# ── Helpers ──────────────────────────────────────────────────
RUB = "&#8381;"
DOT = "&#9679;"


def _row(color, label, value, pct):
    """One detail row: colored dot + label + value + grey percentage."""
    return (
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin:2px 0;'>"
        f"  <span><span style='color:{color}'>{DOT}</span> {label}</span>"
        f"  <span><b>{fmt_number(value)}</b> <span style='color:#94a3b8'>{pct:.0f}%</span></span>"
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

def _pct_change(values):
    if not values or len(values) < 2:
        return 0
    prev, curr = values[-2], values[-1]
    return ((curr - prev) / prev * 100) if prev else 0


_SPARK_W = 260
_SPARK_H = 55


def _smooth_path(pts, close_to=None):
    """Catmull-Rom → Cubic-Bezier path for smooth sparklines.

    ``close_to`` — optional ``(x, y)`` of a baseline point to close the area polygon.
    If given, prefixes with ``M{close_to}`` and appends ``L{first_pt}`` before the
    curves so the result can be used as an area fill.
    """
    if not pts:
        return ""
    if len(pts) == 1:
        x, y = pts[0]
        return f"M{x:.1f},{y:.1f}"

    # Tension: 0 = linear, 0.5 = classic Catmull-Rom. Keep low for sparklines.
    T = 0.22

    if close_to is not None:
        cx, cy = close_to
        d = f"M{cx:.1f},{cy:.1f} L{pts[0][0]:.1f},{pts[0][1]:.1f}"
    else:
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"

    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else p2
        cp1x = p1[0] + (p2[0] - p0[0]) * T
        cp1y = p1[1] + (p2[1] - p0[1]) * T
        cp2x = p2[0] - (p3[0] - p1[0]) * T
        cp2y = p2[1] - (p3[1] - p1[1]) * T
        d += (
            f" C{cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f}"
            f" {p2[0]:.1f},{p2[1]:.1f}"
        )
    return d


def _spark_svg(values, color, width=_SPARK_W, height=_SPARK_H):
    """Render an inline SVG sparkline (smooth area + curve). Hover handled by JS below."""
    if not values or len(values) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}"'
            f' preserveAspectRatio="none" style="display:block;"></svg>'
        )
    mx = max(values)
    mn = min(values)
    rng = (mx - mn) or 1
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = i / max(n - 1, 1) * width
        y = height - (v - mn) / rng * height * 0.82 - height * 0.08
        pts.append((x, y))
    line_path = _smooth_path(pts)
    # Area path: start at bottom-left, move to first point, curve through points,
    # then close down to bottom-right.
    area_path = (
        _smooth_path(pts, close_to=(0, height))
        + f" L{width:.1f},{height:.1f} Z"
    )
    # Marker + vertical guide — hidden until mouse enters, driven by SPARK_HOVER_JS.
    # preserveAspectRatio="xMidYMid meet" keeps curves proportional on any screen width.
    return (
        f'<svg class="spk-svg" viewBox="0 0 {width} {height}" width="100%" height="{height}"'
        f' preserveAspectRatio="none"'
        f' shape-rendering="geometricPrecision"'
        f' style="display:block;overflow:visible;cursor:crosshair;">'
        f'<path d="{area_path}" fill="{color}" fill-opacity="0.14"/>'
        f'<path d="{line_path}" fill="none" stroke="{color}"'
        f' stroke-width="2" stroke-linejoin="round" stroke-linecap="round"'
        f' vector-effect="non-scaling-stroke"/>'
        f'<line class="spk-guide" x1="0" y1="0" x2="0" y2="{height}" stroke="{color}"'
        f' stroke-width="1" stroke-dasharray="3,3" opacity="0"/>'
        f'<circle class="spk-dot" r="4" fill="white" stroke="{color}"'
        f' stroke-width="2" opacity="0"/>'
        f'</svg>'
    )


def _spark_card(idx, title, value, daily_values, daily_labels, color, date_str, expense=False):
    pct = _pct_change(daily_values)
    if expense:
        pct_color = "#ef4444" if pct >= 0 else "#22c55e"
    else:
        pct_color = "#22c55e" if pct >= 0 else "#ef4444"
    sign = "+" if pct >= 0 else ""
    svg = _spark_svg(daily_values or [], color)
    # Embed raw data as JSON in data-* attributes for JS hover handler.
    # Escape quotes so JSON survives inside double-quoted HTML attributes.
    import json as _json
    import html as _html
    data_vals = _html.escape(_json.dumps(daily_values or []), quote=True)
    data_lbls = _html.escape(_json.dumps(daily_labels or []), quote=True)
    return (
        f'<div class="spk-card" data-values="{data_vals}" data-labels="{data_lbls}"'
        f' data-color="{color}" data-title="{title}"'
        f' style="background:white;border-radius:14px;padding:1rem 1.2rem 1.1rem;'
        f'box-shadow:0 4px 16px rgba(15,23,42,0.07);position:relative;overflow:visible;">'
        f'<div style="font-size:0.95rem;color:#1e293b;font-weight:700;">{title}</div>'
        f'<div style="font-size:0.72rem;color:#94a3b8;">{date_str}</div>'
        f'<div style="font-size:1.7rem;font-weight:700;color:#0f172a;margin:0.25rem 0;white-space:nowrap;">'
        f'{fmt_number(value)}</div>'
        f'<div style="font-size:0.72rem;color:{pct_color};font-weight:500;">'
        f'{sign}{pct:.0f}% динамика за день</div>'
        f'<div class="spk-chart" style="margin-top:6px;position:relative;overflow:visible;">{svg}'
        f'<div class="spk-tip" style="position:absolute;pointer-events:none;'
        f'display:none;background:rgba(15,23,42,0.92);color:white;padding:4px 8px;'
        f'border-radius:6px;font-size:11px;white-space:nowrap;'
        f'box-shadow:0 4px 10px rgba(0,0,0,0.18);z-index:999;"></div>'
        f'</div>'
        f'</div>'
    )


# JS injected inside the iframe: on mousemove over a .spk-chart it snaps
# to the nearest data point, shows tooltip + circle marker + vertical guide.
SPARK_HOVER_JS = (
    "<script>\n"
    "(function(){\n"
    "  const W=" + str(_SPARK_W) + ", H=" + str(_SPARK_H) + ";\n"
    "  function fmtNum(v){return Math.abs(v)>=1000?v.toLocaleString('ru-RU').replace(/,/g,' '):(v||0).toString();}\n"
    "  document.querySelectorAll('.spk-card').forEach(card=>{\n"
    "    const values=JSON.parse(card.dataset.values||'[]');\n"
    "    const labels=JSON.parse(card.dataset.labels||'[]');\n"
    "    if(values.length<2) return;\n"
    "    const chart=card.querySelector('.spk-chart');\n"
    "    const svg=card.querySelector('.spk-svg');\n"
    "    const dot=card.querySelector('.spk-dot');\n"
    "    const guide=card.querySelector('.spk-guide');\n"
    "    const tip=card.querySelector('.spk-tip');\n"
    "    const mx=Math.max(...values), mn=Math.min(...values), rng=(mx-mn)||1;\n"
    "    function onMove(e){\n"
    "      const rect=svg.getBoundingClientRect();\n"
    "      const chartRect=chart.getBoundingClientRect();\n"
    "      const rel=(e.clientX-rect.left)/rect.width;\n"
    "      const idx=Math.max(0,Math.min(values.length-1,Math.round(rel*(values.length-1))));\n"
    "      const v=values[idx], lbl=labels[idx]||'';\n"
    "      const xVb=idx/(values.length-1)*W;\n"
    "      const yVb=H-(v-mn)/rng*H*0.82-H*0.08;\n"
    "      dot.setAttribute('cx',xVb); dot.setAttribute('cy',yVb); dot.setAttribute('opacity','1');\n"
    "      guide.setAttribute('x1',xVb); guide.setAttribute('x2',xVb); guide.setAttribute('opacity','0.6');\n"
    "      const pxX=(xVb/W)*rect.width + (rect.left-chartRect.left);\n"
    "      const pxY=(yVb/H)*rect.height + (rect.top-chartRect.top);\n"
    "      // Show tooltip + measure\n"
    "      tip.style.display='block';\n"
    "      tip.style.left='0px'; tip.style.top='0px'; tip.style.transform='none';\n"
    "      tip.innerHTML='<div style=\"opacity:0.75;font-size:10px\">'+lbl+'</div><div style=\"font-weight:600\">'+fmtNum(v)+'</div>';\n"
    "      const tw=tip.offsetWidth, th=tip.offsetHeight;\n"
    "      // Horizontal: center under point, clamp inside chart bounds (4px margin)\n"
    "      let tx=pxX - tw/2;\n"
    "      tx=Math.max(4, Math.min(chartRect.width-tw-4, tx));\n"
    "      // Vertical: prefer above point; if not enough room, place below\n"
    "      let ty=pxY - th - 8;\n"
    "      if(ty<2){ ty=pxY + 12; }\n"
    "      tip.style.left=tx+'px';\n"
    "      tip.style.top=ty+'px';\n"
    "    }\n"
    "    function onLeave(){ dot.setAttribute('opacity','0'); guide.setAttribute('opacity','0'); tip.style.display='none'; }\n"
    "    chart.addEventListener('mousemove',onMove);\n"
    "    chart.addEventListener('mouseleave',onLeave);\n"
    "  });\n"
    "})();\n"
    "</script>"
)

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
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;" title="Продажи − Возвраты по данным финансового отчёта WB (если подключён).">Реализация ⓘ</div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {fmt_number(card_realizacia)} {RUB}
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
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;" title="Комиссия, логистика, хранение, штрафы, приёмка, удержания (включая рекламу WB).">
      Услуги WB ⓘ <span style="color:#94a3b8; font-size:0.78rem;">{svc_total_pct:.0f}%</span>
    </div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {fmt_number(services_total)} {RUB}
    </div>
    {services_bar}
    <div style="font-size:0.8rem; color:#475569; line-height:1.7;">
      {services_detail}
    </div>
    {services_note}
  </div>

  <!-- НАЛОГИ И ЗАТРАТЫ -->
  <div style="{CARD}">
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;" title="Себестоимость товара + налог (УСН 6% от прибыли) + прочие фиксированные расходы из справочника.">
      Налоги и затраты ⓘ <span style="color:#94a3b8; font-size:0.78rem;">{total_costs_pct:.0f}%</span>
    </div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {fmt_number(total_costs)} {RUB}
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
    <div style="font-size:0.82rem; color:#64748b; font-weight:600;" title="Реализация − Услуги WB − Себестоимость − Налог − Доп. расходы. Маржинальность = прибыль / реализация.">Операционная прибыль ⓘ</div>
    <div style="font-size:1.8rem; font-weight:700; color:#0f172a; margin:0.2rem 0;">
      {fmt_number(op_profit)} {RUB}
    </div>
    <div style="font-size:0.85rem; color:#475569; line-height:2; margin-top:0.4rem;">
      <div style="display:flex; justify-content:space-between;">
        <span>Маржинальность</span> <b>{margin_pct:.1f}%</b>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Рентабельность</span> <b>{roi_pct:.1f}%</b>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Средний чек</span> <b>{fmt_number(avg_check)} {RUB}</b>
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
        acquiring_amount=("acquiring_amount", "sum"),
        deduction_amount=("deduction_amount", "sum"),
        commission_amount=("commission_amount", "sum"),
        storage_amount=("storage_amount", "sum"),
        penalty_amount=("penalty_amount", "sum"),
        acceptance_amount=("acceptance_amount", "sum"),
        additional_payment_amount=("additional_payment_amount", "sum"),
    ).sort_index()
    # «Все услуги» — полный набор удержаний WB по финансовому отчёту.
    # Используется только для sparkline; не участвует в расчёте прибыли.
    fin_by_day["total_services"] = (
        fin_by_day["commission_amount"] + fin_by_day["logistics_amount"]
        + fin_by_day["storage_amount"] + fin_by_day["penalty_amount"]
        + fin_by_day["acceptance_amount"]
        + fin_by_day["acquiring_amount"] + fin_by_day["deduction_amount"]
        - fin_by_day["additional_payment_amount"]
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
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>body{margin:0;padding:0;background:transparent;"
    f"{_font}"
    "color:#0f172a}</style></head><body>"
    f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem;'>"
    + _spark_card(0, "Заказы", total_orders_amt, spark_orders, spark_orders_lbl, "#f97316", end_fmt)
    + _spark_card(1, "Продажи", fin_sales_amt, spark_sales, s_lbl, "#22c55e", end_fmt)
    + _spark_card(2, "Логистика", fin_logistics, spark_logistics, l_lbl, "#ef4444", end_fmt, expense=True)
    + "</div>"
    f"<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;'>"
    + _spark_card(3, "Реклама", ads_total_spend, spark_ads, a_lbl, "#8b5cf6", end_fmt, expense=True)
    + _spark_card(4, "Все услуги", fin_total_services, spark_services, sv_lbl, "#3b82f6", end_fmt, expense=True)
    + "</div>"
    + SPARK_HOVER_JS
    + "</body></html>"
)
# Use st.iframe (replaces deprecated components.v1.html) — st.html/st.markdown
# strip <svg> via sanitizer, so we keep an isolated iframe for the spark cards.
st.iframe(spark_html, height=420)

# ══════════════════════════════════════════════════════════════
#  Finance-based article aggregation (single source of truth)
#  Uses pre-computed cost_amount / net_profit_amount from SQL so
#  totals are stable across any date filter (no df cost_amount dep).
# ══════════════════════════════════════════════════════════════
if has_finance:
    _fin_art = fin.groupby(["nm_id", "supplier_article"]).agg(
        ppvz_for_pay=("ppvz_for_pay", "sum"),
        logistics_amount=("logistics_amount", "sum"),
        storage_amount=("storage_amount", "sum"),
        penalty_amount=("penalty_amount", "sum"),
        acceptance_amount=("acceptance_amount", "sum"),
        acquiring_amount=("acquiring_amount", "sum"),
        deduction_amount=("deduction_amount", "sum"),
        additional_payment_amount=("additional_payment_amount", "sum"),
        commission_amount=("commission_amount", "sum"),
        sales_count=("sales_count", "sum"),
        returns_count=("returns_count", "sum"),
        sales_amount=("sales_amount", "sum"),
        returns_amount=("returns_amount", "sum"),
        retail_amount=("retail_amount", "sum"),
        cost_amount=("cost_amount", "sum"),
        tax_amount=("tax_amount", "sum"),
        gross_profit_amount=("gross_profit_amount", "sum"),
        net_profit_amount=("net_profit_amount", "sum"),
        subject=("subject", "first"),
        brand=("brand", "first"),
    ).reset_index()
    _fin_art["fin_op_profit"] = _fin_art["net_profit_amount"]
else:
    _fin_art = pd.DataFrame()

# ══════════════════════════════════════════════════════════════
#  MONTHLY CHARTS
#  Orders come from orders_daily (by order_date), everything else
#  (sales, revenue, profit) from finance_daily (by report_date).
#  Using single source per metric avoids date-basis mismatches
#  that previously made Feb profit differ between ranges.
# ══════════════════════════════════════════════════════════════
# Orders by month (from orders_daily, which is order_date based)
if has_orders:
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    ord_df["month"] = ord_df["order_date"].dt.to_period("M").dt.to_timestamp()
    orders_monthly = ord_df.groupby("month").agg(
        orders_count=("orders_count", "sum"),
        orders_amount=("orders_amount", "sum"),
    ).reset_index()
else:
    orders_monthly = pd.DataFrame({"month": [], "orders_count": [], "orders_amount": []})

if has_finance:
    fin["report_date"] = pd.to_datetime(fin["report_date"])
    fin["month"] = fin["report_date"].dt.to_period("M").dt.to_timestamp()
    fin_monthly = fin.groupby("month").agg(
        sales_count=("sales_count", "sum"),
        returns_count=("returns_count", "sum"),
        sales_amount=("sales_amount", "sum"),
        returns_amount=("returns_amount", "sum"),
        retail_amount=("retail_amount", "sum"),
        ppvz_for_pay=("ppvz_for_pay", "sum"),
        commission_amount=("commission_amount", "sum"),
        logistics_amount=("logistics_amount", "sum"),
        storage_amount=("storage_amount", "sum"),
        penalty_amount=("penalty_amount", "sum"),
        acceptance_amount=("acceptance_amount", "sum"),
        acquiring_amount=("acquiring_amount", "sum"),
        deduction_amount=("deduction_amount", "sum"),
        additional_payment_amount=("additional_payment_amount", "sum"),
        cost_amount=("cost_amount", "sum"),
        tax_amount=("tax_amount", "sum"),
        gross_profit_amount=("gross_profit_amount", "sum"),
        net_profit_amount=("net_profit_amount", "sum"),
    ).reset_index()
    # Раскка-style: Реализация до СПП = sales − returns (по цене со скидкой продавца).
    fin_monthly["net_revenue"] = fin_monthly["sales_amount"] - fin_monthly["returns_amount"]
    fin_monthly["operating_profit_amount"] = fin_monthly["net_profit_amount"]
    monthly = fin_monthly.merge(orders_monthly, on="month", how="outer").fillna(0)
else:
    # Fallback: sales_daily
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    df["month"] = df["sales_date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").agg(
        sales_count=("sales_count", "sum"),
        returns_count=("returns_count", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        net_revenue=("net_revenue", "sum"),
        commission_amount=("commission_amount", "sum"),
        cost_amount=("cost_amount", "sum"),
        operating_profit_amount=("operating_profit_amount", "sum"),
    ).reset_index()
    monthly = monthly.merge(orders_monthly, on="month", how="outer").fillna(0)

monthly = monthly.sort_values("month").reset_index(drop=True)
# Fallback для исторических месяцев без orders_daily: используем
# sales_count из финансового отчёта как приближение к "Заказам"
# (orders_daily ETL хранит только последние ~6 месяцев — для старых
# периодов отсутствие заказов на графике сбивало пользователей).
if "sales_count" in monthly.columns:
    _missing_orders = monthly["orders_count"].fillna(0) == 0
    _has_sales = monthly["sales_count"].fillna(0) > 0
    monthly.loc[_missing_orders & _has_sales, "orders_count"] = monthly.loc[
        _missing_orders & _has_sales, "sales_count"
    ]
    monthly["orders_estimated"] = (_missing_orders & _has_sales).astype(int)
else:
    monthly["orders_estimated"] = 0
monthly["margin_pct"] = (
    monthly["operating_profit_amount"] / monthly["net_revenue"].replace(0, 1) * 100
).fillna(0).round(1)
# Cap margin to plausible range (-200%..200%) — для очень старых
# периодов с неполным cost_reference маржа могла «улетать» в +1000%
# и портить шкалу графика.
monthly["margin_pct"] = monthly["margin_pct"].clip(lower=-200, upper=200)
monthly["avg_check"] = (
    monthly["net_revenue"] / monthly["sales_count"].replace(0, 1)
).fillna(0).round(0)
monthly["label"] = pd.to_datetime(monthly["month"]).dt.strftime("%b %Y")

# ── Chart 1: Orders + Sales + Avg Check ─────────────────────
st.markdown("### Заказы, продажи и средний чек по месяцам")
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(
    x=monthly["label"], y=monthly["orders_count"],
    name="Заказы",
    marker=dict(color=PLOTLY_COLORS["amber"], line=dict(color="#d97706", width=0.5)),
    opacity=0.88,
    text=monthly["orders_count"], textposition="outside",
    hovertemplate="Заказы: %{y:,.0f} шт.<extra></extra>",
), secondary_y=False)
fig1.add_trace(go.Bar(
    x=monthly["label"], y=monthly["sales_count"],
    name="Продажи",
    marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5)),
    opacity=0.88,
    text=monthly["sales_count"], textposition="outside",
    hovertemplate="Продажи: %{y:,.0f} шт.<extra></extra>",
), secondary_y=False)
fig1.add_trace(go.Scatter(
    x=monthly["label"], y=monthly["avg_check"],
    name="Средний чек",
    line=dict(color=PLOTLY_COLORS["purple"], width=2.5, shape="spline"),
    mode="lines+markers+text",
    marker=dict(size=7, color=PLOTLY_COLORS["purple"], line=dict(color="white", width=1.5)),
    fill="tozeroy", fillcolor="rgba(139,92,246,0.06)",
    text=monthly["avg_check"].apply(lambda v: f"{v:,.0f}"),
    textposition="top center", textfont=dict(size=11),
    hovertemplate="Средний чек: %{y:,.0f} ₽<extra></extra>",
), secondary_y=True)
fig1.update_layout(
    **PLOTLY_LAYOUT,  # title уже задан пустым внутри PLOTLY_LAYOUT
    barmode="group", bargap=0.25, bargroupgap=0.1,
    height=420, margin=dict(t=50),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)
fig1.update_yaxes(title_text="Количество", secondary_y=False, tickformat=",")
fig1.update_yaxes(title_text="Средний чек, \u20bd", secondary_y=True,
                  tickfont=dict(color=PLOTLY_COLORS["purple"]),
                  title_font=dict(color=PLOTLY_COLORS["purple"]))
plotly_defaults(fig1)
st.plotly_chart(fig1, width="stretch")

# ── Chart 2: Revenue + Operating Profit + Margin % ──────────
st.markdown("### Реализация, операционная прибыль и маржинальность")
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Bar(
    x=monthly["label"], y=monthly["net_revenue"],
    name="Реализация (нетто)",
    marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5)),
    opacity=0.88,
    text=monthly["net_revenue"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
    hovertemplate="Реализация: %{y:,.0f} ₽<extra></extra>",
), secondary_y=False)
fig2.add_trace(go.Bar(
    x=monthly["label"], y=monthly["operating_profit_amount"],
    name="Операц. прибыль",
    marker=dict(color=PLOTLY_COLORS["green"], line=dict(color=PLOTLY_COLORS["green_dark"], width=0.5)),
    opacity=0.88,
    text=monthly["operating_profit_amount"].apply(lambda v: f"{v / 1000:,.0f}к"),
    textposition="outside",
    hovertemplate="Прибыль: %{y:,.0f} ₽<extra></extra>",
), secondary_y=False)
fig2.add_trace(go.Scatter(
    x=monthly["label"], y=monthly["margin_pct"],
    name="% маржинальности",
    line=dict(color=PLOTLY_COLORS["amber"], width=2.5, shape="spline"),
    mode="lines+markers+text",
    marker=dict(size=7, color=PLOTLY_COLORS["amber"], line=dict(color="white", width=1.5)),
    fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
    text=monthly["margin_pct"].apply(lambda v: f"{v:.1f}%"),
    textposition="top center", textfont=dict(size=11),
    hovertemplate="Маржа: %{y:.1f}%<extra></extra>",
), secondary_y=True)
# Cap margin Y-axis to plausible range so historic outliers don't squash the chart.
_mx = float(monthly["margin_pct"].max() if len(monthly) else 0)
margin_max = min(max(_mx * 1.5, 10), 100)
fig2.update_layout(
    **PLOTLY_LAYOUT,  # title уже задан пустым внутри PLOTLY_LAYOUT
    barmode="group", bargap=0.25, bargroupgap=0.1,
    height=420, margin=dict(t=50),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
)
fig2.update_yaxes(title_text="Сумма, \u20bd", secondary_y=False, tickformat=",")
fig2.update_yaxes(title_text="Маржинальность, %", secondary_y=True,
                  range=[0, margin_max],
                  tickfont=dict(color=PLOTLY_COLORS["amber"]),
                  title_font=dict(color=PLOTLY_COLORS["amber"]))
plotly_defaults(fig2)
st.plotly_chart(fig2, width="stretch")

# ══════════════════════════════════════════════════════════════
#  DAILY DYNAMICS (amounts in rubles)
# ══════════════════════════════════════════════════════════════
st.markdown("### Динамика показателей")

if has_finance:
    # Дневная динамика строится на финансовом отчёте (report_date = rr_dt).
    fin_daily = fin.groupby("report_date").agg(
        net_revenue=("sales_amount", "sum"),            # до СПП, без возвратов
        returns_amount=("returns_amount", "sum"),
        profit_amount=("net_profit_amount", "sum"),
    ).reset_index()
    fin_daily["net_revenue"] = fin_daily["net_revenue"] - fin_daily["returns_amount"]
    fin_daily.drop(columns=["returns_amount"], inplace=True, errors="ignore")
    fin_daily = fin_daily.rename(columns={"report_date": "sales_date"})
    fin_daily["sales_date"] = pd.to_datetime(fin_daily["sales_date"])
    daily_rev = fin_daily
else:
    df["sales_date"] = pd.to_datetime(df["sales_date"])
    daily_rev = df.groupby("sales_date").agg(
        net_revenue=("net_revenue", "sum"),
        profit_amount=("profit_amount", "sum"),
    ).reset_index()

if has_orders:
    daily_ord = ord_df[["order_date", "orders_amount"]].copy()
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
    name="Заказы",
    marker=dict(color=PLOTLY_COLORS["amber"], line=dict(color="#d97706", width=0.5)),
    opacity=0.75,
    hovertemplate="%{x|%d.%m}<br>Заказы: %{y:,.0f} ₽<extra></extra>",
))
fig3.add_trace(go.Bar(
    x=daily_dyn["sales_date"], y=daily_dyn["net_revenue"],
    name="Продажи",
    marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5)),
    opacity=0.80,
    hovertemplate="%{x|%d.%m}<br>Продажи: %{y:,.0f} ₽<extra></extra>",
))
fig3.add_trace(go.Bar(
    x=daily_dyn["sales_date"], y=daily_dyn["profit_amount"],
    name="Прибыль",
    marker=dict(color=PLOTLY_COLORS["green"], line=dict(color=PLOTLY_COLORS["green_dark"], width=0.5)),
    opacity=0.80,
    hovertemplate="%{x|%d.%m}<br>Прибыль: %{y:,.0f} ₽<extra></extra>",
))
fig3.update_layout(
    **PLOTLY_LAYOUT,
    barmode="group", bargap=0.25, bargroupgap=0.1,
    height=400,
    xaxis_title="Дата", yaxis_title="Сумма, \u20bd",
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
)
plotly_defaults(fig3)
st.plotly_chart(fig3, width="stretch")

# ══════════════════════════════════════════════════════════════
#  TOP-10 HORIZONTAL BAR CHARTS
#  Источник — предагрегированный `_fin_art` (одна строка на nm_id),
#  чтобы суммирование по brand/subject/article НЕ умножалось на число дней.
# ══════════════════════════════════════════════════════════════
st.markdown("### Операционная прибыль: Топ-10")
tc1, tc2, tc3 = st.columns(3)

if has_finance and not _fin_art.empty:
    _top_src = _fin_art.rename(columns={"fin_op_profit": "operating_profit_amount"})
else:
    _top_src = df.groupby(["nm_id", "supplier_article", "brand", "subject"]).agg(
        operating_profit_amount=("operating_profit_amount", "sum")
    ).reset_index()


def _fmt_bar_text(series):
    return series.apply(
        lambda v: f"{v / 1000:,.0f}к".replace(",", " ") if abs(v) >= 1000 else f"{v:,.0f}"
    )


# ── Top-10 by brand ──
with tc1:
    st.markdown("**По брендам**")
    by_brand = (
        _top_src[_top_src["brand"].notna() & (_top_src["brand"] != "")]
        .groupby("brand")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_b = px.bar(
        by_brand, x="operating_profit_amount", y="brand",
        orientation="h", color_discrete_sequence=[PLOTLY_COLORS["blue"]],
        text=_fmt_bar_text(by_brand["operating_profit_amount"]),
    )
    fig_b.update_traces(
        textposition="outside",
        marker=dict(line=dict(color=PLOTLY_COLORS["blue_dark"], width=0.5)),
        hovertemplate="<b>%{y}</b><br>Прибыль: %{x:,.0f} ₽<extra></extra>",
    )
    fig_b.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False, height=380, bargap=0.25,
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    plotly_defaults(fig_b)
    st.plotly_chart(fig_b, width="stretch")

# ── Top-10 by subject ──
with tc2:
    st.markdown("**По предметам**")
    by_subj = (
        _top_src[_top_src["subject"].notna()]
        .groupby("subject")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_s = px.bar(
        by_subj, x="operating_profit_amount", y="subject",
        orientation="h", color_discrete_sequence=[PLOTLY_COLORS["green"]],
        text=_fmt_bar_text(by_subj["operating_profit_amount"]),
    )
    fig_s.update_traces(
        textposition="outside",
        marker=dict(line=dict(color=PLOTLY_COLORS["green_dark"], width=0.5)),
        hovertemplate="<b>%{y}</b><br>Прибыль: %{x:,.0f} ₽<extra></extra>",
    )
    fig_s.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False, height=380, bargap=0.25,
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    plotly_defaults(fig_s)
    st.plotly_chart(fig_s, width="stretch")

# ── Top-10 by article ──
with tc3:
    st.markdown("**По артикулам**")
    by_art = (
        _top_src[_top_src["supplier_article"].notna()]
        .groupby("supplier_article")["operating_profit_amount"]
        .sum().reset_index()
        .sort_values("operating_profit_amount", ascending=True)
        .tail(10)
    )
    fig_a = px.bar(
        by_art, x="operating_profit_amount", y="supplier_article",
        orientation="h", color_discrete_sequence=[PLOTLY_COLORS["purple"]],
        text=_fmt_bar_text(by_art["operating_profit_amount"]),
    )
    fig_a.update_traces(
        textposition="outside",
        marker=dict(line=dict(color="#6d28d9", width=0.5)),
        hovertemplate="<b>%{y}</b><br>Прибыль: %{x:,.0f} ₽<extra></extra>",
    )
    fig_a.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False, height=380, bargap=0.25,
        xaxis_title="", yaxis_title="",
        margin=dict(l=10, r=80, t=10, b=10),
    )
    plotly_defaults(fig_a)
    st.plotly_chart(fig_a, width="stretch")

# ── Export ────────────────────────────────────────────────────
export_buttons(df, "kpi_dashboard", sheet_name="KPI")
