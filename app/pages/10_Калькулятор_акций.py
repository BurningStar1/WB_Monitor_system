"""Калькулятор акций — оценка целесообразности участия в промо-акциях WB."""
import io
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from marts import fetch_dataframe, FIN_PROMO_BASELINE_QUERY
from styles import inject_global_styles, SORT_JS, wb_link, render_table
from auth import check_auth, logout

# MIME-type для xlsx
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _to_excel(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Сериализует DataFrame в xlsx-байты."""
    buf = io.BytesIO()
    out = df.copy()
    # Strip timezone info — openpyxl не поддерживает tz-aware datetimes
    for col in out.select_dtypes(include=["datetimetz"]).columns:
        out[col] = out[col].dt.tz_localize(None)
    out.to_excel(buf, index=False, engine="openpyxl", sheet_name=sheet_name)
    return buf.getvalue()

# ── Page setup ────────────────────────────────────────────────

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("\U0001f3f7 \u041a\u0430\u043b\u044c\u043a\u0443\u043b\u044f\u0442\u043e\u0440 \u0430\u043a\u0446\u0438\u0439")

with st.expander("ℹ️ Как рассчитывается выгода от акции", expanded=False):
    st.markdown(
        """
        **Цель:** оценить, выгодно ли снижать цену артикула ради участия в промо-акции WB.

        **Как считаем прибыль по акционной цене:**
        1. `Цена со скидкой покупателя (СПП)` = `Акционная цена × (1 − avg_SPP%)`
        2. `Комиссия WB` — пропорционально средней комиссии с единицы
        3. `Прибыль на единицу` = `Цена со СПП − Комиссия − Логистика − Себестоимость`
        4. `Break-even sales` — во сколько раз должны вырасти продажи,
           чтобы сохранить текущую дневную прибыль

        **Вердикты:**
        - 🟢 **Рекомендуется** — промо прибыльно, темп роста ≤1.5× достижим
        - 🟡 **Осторожно** — нужен рост продаж 1.5–3×
        - 🔴 **Не рекомендуется** — требуется рост >3×, маловероятно
        - ❌ **Не участвовать** — прибыль на единицу отрицательная

        **Совет:** сравнивайте **break-even** с реальной историей продаж артикула —
        если темпы в прошлых акциях были ≤2×, то порог 3× выглядит недостижимым.
        """
    )

# ── Verdict helpers ───────────────────────────────────────────

_VERDICT_STYLES = {
    "loss":    {"label": "\u041d\u0435 \u0443\u0447\u0430\u0441\u0442\u0432\u043e\u0432\u0430\u0442\u044c",  "color": "#dc2626", "bg": "#fef2f2", "border": "#fca5a5"},
    "good":    {"label": "\u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f",   "color": "#16a34a", "bg": "#f0fdf4", "border": "#86efac"},
    "caution": {"label": "\u041e\u0441\u0442\u043e\u0440\u043e\u0436\u043d\u043e",     "color": "#ca8a04", "bg": "#fefce8", "border": "#fde047"},
    "bad":     {"label": "\u041d\u0435 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f", "color": "#dc2626", "bg": "#fef2f2", "border": "#fca5a5"},
}


def _classify(promo_profit: float, tempo_ratio: float) -> str:
    if promo_profit <= 0:
        return "loss"
    if tempo_ratio <= 1.5:
        return "good"
    if tempo_ratio <= 3.0:
        return "caution"
    return "bad"


def _safe_float(val, default=0.0):
    """Safely convert to float, handling NaN/None."""
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except (ValueError, TypeError):
        return default


def _calc_promo(row: pd.Series, promo_price: float) -> dict:
    """Calculate all promo metrics for a single article (finance-based)."""
    avg_spp_pct = _safe_float(row.get("avg_spp_pct", 0))
    avg_price_after = _safe_float(row.get("avg_price_after_spp", 0))
    commission_per_unit = _safe_float(row.get("commission_per_unit", 0))
    logistics_per_unit = _safe_float(row.get("logistics_per_unit", 0))
    cost_per_unit = _safe_float(row.get("cost_per_unit", 0))
    profit_per_unit = _safe_float(row.get("profit_per_unit", 0))
    avg_orders_day = _safe_float(row.get("avg_orders_day", 0))
    buyout_pct = _safe_float(row.get("buyout_pct", 0))

    promo_price_after_spp = promo_price * (1 - avg_spp_pct / 100)

    # Proportional commission
    if avg_price_after > 0:
        promo_commission = promo_price_after_spp * (commission_per_unit / avg_price_after)
    else:
        promo_commission = 0

    promo_profit_per_unit = promo_price_after_spp - promo_commission - logistics_per_unit - cost_per_unit

    buyout_share = buyout_pct / 100 if buyout_pct > 0 else 0
    current_daily_profit = profit_per_unit * avg_orders_day * buyout_share

    if promo_profit_per_unit > 0 and current_daily_profit > 0:
        break_even_sales = current_daily_profit / promo_profit_per_unit
    else:
        break_even_sales = None

    current_actual_sales = avg_orders_day * buyout_share
    if break_even_sales is not None and current_actual_sales > 0:
        tempo_ratio = break_even_sales / current_actual_sales
    else:
        tempo_ratio = float("inf")

    verdict_key = _classify(promo_profit_per_unit, tempo_ratio)

    return {
        "promo_price": promo_price,
        "promo_price_after_spp": round(promo_price_after_spp, 0),
        "promo_commission": round(promo_commission, 2),
        "promo_profit_per_unit": round(promo_profit_per_unit, 2),
        "current_daily_profit": round(current_daily_profit, 2),
        "break_even_sales": round(break_even_sales, 2) if break_even_sales else None,
        "current_actual_sales": round(current_actual_sales, 2),
        "tempo_ratio": round(tempo_ratio, 2) if tempo_ratio != float("inf") else None,
        "verdict_key": verdict_key,
    }


def _fmt(v, suffix=""):
    """Format a numeric value for display."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "\u2014"
    return f"{v:,.0f}{suffix}".replace(",", " ")


def _fmt2(v, suffix=""):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "\u2014"
    return f"{v:,.2f}{suffix}".replace(",", " ")


# ── Custom CSS ────────────────────────────────────────────────

PROMO_CSS = """
<style>
/* Result card */
.promo-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
    border-top: 4px solid var(--accent);
    margin-bottom: 1rem;
}
.promo-card .card-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.promo-card .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}
.promo-card .metric-item {
    padding: 0.5rem 0.75rem;
    border-radius: 10px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.promo-card .metric-item .ml {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    font-weight: 600;
    margin-bottom: 2px;
}
.promo-card .metric-item .mv {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
}
.promo-card .metric-item .mv.green { color: #16a34a; }
.promo-card .metric-item .mv.red   { color: #dc2626; }
.promo-card .metric-item .mv.amber { color: #ca8a04; }

/* Verdict badge */
.verdict-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.3px;
    margin-top: 0.75rem;
}

/* Section divider inside card */
.promo-card .section-label {
    font-size: 12px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 1rem 0 0.5rem 0;
    padding-top: 0.75rem;
    border-top: 1px solid #e2e8f0;
}

/* Verdict description */
.verdict-desc {
    font-size: 13px;
    color: #475569;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* Batch table */
.art-wrap {
    overflow-x: auto;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(15, 23, 42, 0.08);
    margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
}
.art-t {
    border-collapse: collapse;
    width: max-content;
    min-width: 100%;
    font-size: 12px;
    font-family: Inter, system-ui, sans-serif;
    background: #fff;
    color: #1e293b;
}
.art-t thead th {
    background: #f1f5f9;
    position: sticky;
    top: 0;
    z-index: 3;
    padding: 8px 10px;
    border-bottom: 2px solid #cbd5e1;
    border-right: 1px solid #e2e8f0;
    font-weight: 600;
    font-size: 11px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    text-align: center;
    white-space: nowrap;
}
.art-t thead th:last-child { border-right: none; }
.art-t td {
    border-bottom: 1px solid #f1f5f9;
    border-right: 1px solid #f8fafc;
    padding: 6px 10px;
    white-space: nowrap;
    vertical-align: middle;
}
.art-t td:last-child { border-right: none; }
.art-t tbody tr:nth-child(even) { background: #fafbfc; }
.art-t tbody tr:hover { background: #eef2ff; }
.art-t .num { text-align: right; }
.art-t .ctr { text-align: center; }
.art-t .pos { color: #16a34a; font-weight: 700; }
.art-t .neg { color: #dc2626; font-weight: 700; }
.art-t .warn { color: #ca8a04; font-weight: 700; }
.art-t .badge-cell {
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    display: inline-block;
}
</style>
"""

st.markdown(PROMO_CSS, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────

df = fetch_dataframe(FIN_PROMO_BASELINE_QUERY, {})

if df.empty:
    st.info("\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u043f\u043e \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u0430\u043c \u0437\u0430 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 30 \u0434\u043d\u0435\u0439")
    st.stop()

# Build display label for selectbox
df["_label"] = df["supplier_article"].astype(str) + "  |  " + df["nm_id"].astype(str)

# ── Tabs ──────────────────────────────────────────────────────

tab_single, tab_batch, tab_excel, tab_manual = st.tabs([
    "\U0001f50d \u0420\u0430\u0441\u0447\u0451\u0442 \u043f\u043e \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u0443",
    "\U0001f4ca \u041c\u0430\u0441\u0441\u043e\u0432\u044b\u0439 \u0430\u043d\u0430\u043b\u0438\u0437",
    "\U0001f4c2 \u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0438\u0437 Excel",
    "\u270f\ufe0f \u0420\u0443\u0447\u043d\u043e\u0439 \u0432\u0432\u043e\u0434",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Single article calculator
# ══════════════════════════════════════════════════════════════

with tab_single:
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.markdown("### \u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b")

        selected_label = st.selectbox(
            "\u0410\u0440\u0442\u0438\u043a\u0443\u043b",
            df["_label"].tolist(),
            index=0,
            help="\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0430\u0440\u0442\u0438\u043a\u0443\u043b \u0434\u043b\u044f \u0440\u0430\u0441\u0447\u0451\u0442\u0430",
        )

        sel_row = df[df["_label"] == selected_label].iloc[0]

        # Show current price as reference
        cur_price = float(sel_row["avg_price_before_spp"])
        st.caption(f"\u0422\u0435\u043a\u0443\u0449\u0430\u044f \u0446\u0435\u043d\u0430 \u0434\u043e \u0421\u041f\u041f: **{_fmt(cur_price, ' \u20bd')}**")

        promo_price = st.number_input(
            "\u0426\u0435\u043d\u0430 \u043f\u043e \u0430\u043a\u0446\u0438\u0438 (\u0434\u043e \u0421\u041f\u041f), \u20bd",
            min_value=0.0,
            value=round(cur_price * 0.8, 0) if cur_price > 0 else 0.0,
            step=10.0,
            format="%.0f",
        )

        discount_pct = round((1 - promo_price / cur_price) * 100, 1) if cur_price > 0 else 0
        st.caption(f"\u0421\u043a\u0438\u0434\u043a\u0430: **{discount_pct:.1f}%**")

        calc_clicked = st.button("\U0001f4b0 \u0420\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u0442\u044c", width="stretch")

    with col_result:
        if calc_clicked and promo_price > 0:
            res = _calc_promo(sel_row, promo_price)
            vs = _VERDICT_STYLES[res["verdict_key"]]

            # Metric value color class
            profit_cls = "green" if res["promo_profit_per_unit"] > 0 else "red"
            tempo_cls = "green" if res["tempo_ratio"] and res["tempo_ratio"] <= 1.5 else (
                "amber" if res["tempo_ratio"] and res["tempo_ratio"] <= 3.0 else "red"
            )

            # Verdict description
            if res["verdict_key"] == "loss":
                verdict_desc = "\u041f\u0440\u043e\u0434\u0430\u0436\u0430 \u043f\u043e \u0430\u043a\u0446\u0438\u0438 \u043f\u0440\u0438\u043d\u0435\u0441\u0451\u0442 \u0443\u0431\u044b\u0442\u043e\u043a \u043d\u0430 \u043a\u0430\u0436\u0434\u043e\u0439 \u0435\u0434\u0438\u043d\u0438\u0446\u0435. \u0423\u0447\u0430\u0441\u0442\u0438\u0435 \u043d\u0435 \u0438\u043c\u0435\u0435\u0442 \u0441\u043c\u044b\u0441\u043b\u0430."
            elif res["verdict_key"] == "good":
                verdict_desc = (
                    f"\u0414\u043b\u044f \u043a\u043e\u043c\u043f\u0435\u043d\u0441\u0430\u0446\u0438\u0438 \u043d\u0443\u0436\u043d\u043e \u0443\u0432\u0435\u043b\u0438\u0447\u0438\u0442\u044c \u0442\u0435\u043c\u043f \u043f\u0440\u043e\u0434\u0430\u0436 \u0432\u0441\u0435\u0433\u043e \u0432 {_fmt2(res['tempo_ratio'])}x. "
                    "\u042d\u0442\u043e \u0440\u0435\u0430\u043b\u0438\u0441\u0442\u0438\u0447\u043d\u043e \u0434\u043b\u044f \u0431\u043e\u043b\u044c\u0448\u0438\u043d\u0441\u0442\u0432\u0430 \u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u0439."
                )
            elif res["verdict_key"] == "caution":
                verdict_desc = (
                    f"\u041d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c \u0440\u043e\u0441\u0442 \u0442\u0435\u043c\u043f\u0430 \u043f\u0440\u043e\u0434\u0430\u0436 \u0432 {_fmt2(res['tempo_ratio'])}x. "
                    "\u0423\u0447\u0430\u0441\u0442\u0432\u0443\u0439\u0442\u0435 \u0442\u043e\u043b\u044c\u043a\u043e \u0435\u0441\u043b\u0438 \u0443\u0432\u0435\u0440\u0435\u043d\u044b \u0432 \u0440\u043e\u0441\u0442\u0435 \u0441\u043f\u0440\u043e\u0441\u0430."
                )
            else:
                verdict_desc = (
                    f"\u041d\u0443\u0436\u0435\u043d \u0440\u043e\u0441\u0442 \u0442\u0435\u043c\u043f\u0430 \u043f\u0440\u043e\u0434\u0430\u0436 \u0432 {_fmt2(res['tempo_ratio'])}x. "
                    "\u042d\u0442\u043e \u043f\u0440\u0430\u043a\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u043d\u0435\u0434\u043e\u0441\u0442\u0438\u0436\u0438\u043c\u043e."
                )

            card_html = f"""
            <div class="promo-card" style="--accent: {vs['border']}">
                <div class="card-title">\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0440\u0430\u0441\u0447\u0451\u0442\u0430</div>

                <div class="section-label" style="border-top:none;margin-top:0;padding-top:0">\u0422\u0435\u043a\u0443\u0449\u0438\u0435 \u043f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="ml">\u0426\u0435\u043d\u0430 \u0434\u043e \u0421\u041f\u041f</div>
                        <div class="mv">{_fmt(cur_price)} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u0426\u0435\u043d\u0430 \u043f\u043e\u0441\u043b\u0435 \u0421\u041f\u041f</div>
                        <div class="mv">{_fmt(float(sel_row['avg_price_after_spp']))} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u041f\u0440\u0438\u0431\u044b\u043b\u044c / \u0435\u0434.</div>
                        <div class="mv">{_fmt2(float(sel_row['profit_per_unit']))} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u0414\u043d\u0435\u0432\u043d\u0430\u044f \u043f\u0440\u0438\u0431\u044b\u043b\u044c</div>
                        <div class="mv">{_fmt2(res['current_daily_profit'])} \u20bd</div>
                    </div>
                </div>

                <div class="section-label">\u041f\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u0438 \u043f\u043e \u0430\u043a\u0446\u0438\u0438 (\u0441\u043a\u0438\u0434\u043a\u0430 {discount_pct:.0f}%)</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="ml">\u0426\u0435\u043d\u0430 \u0434\u043e \u0421\u041f\u041f</div>
                        <div class="mv">{_fmt(promo_price)} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u0426\u0435\u043d\u0430 \u043f\u043e\u0441\u043b\u0435 \u0421\u041f\u041f</div>
                        <div class="mv">{_fmt(res['promo_price_after_spp'])} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u041f\u0440\u0438\u0431\u044b\u043b\u044c / \u0435\u0434.</div>
                        <div class="mv {profit_cls}">{_fmt2(res['promo_profit_per_unit'])} \u20bd</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u041a\u043e\u043c\u0438\u0441\u0441\u0438\u044f / \u0435\u0434.</div>
                        <div class="mv">{_fmt2(res['promo_commission'])} \u20bd</div>
                    </div>
                </div>

                <div class="section-label">\u0422\u043e\u0447\u043a\u0430 \u0431\u0435\u0437\u0443\u0431\u044b\u0442\u043e\u0447\u043d\u043e\u0441\u0442\u0438</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="ml">\u0422\u0435\u043a\u0443\u0449\u0438\u0435 \u043f\u0440\u043e\u0434\u0430\u0436\u0438 / \u0434\u0435\u043d\u044c</div>
                        <div class="mv">{_fmt2(res['current_actual_sales'])} \u0448\u0442.</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u041d\u0443\u0436\u043d\u043e \u043f\u0440\u043e\u0434\u0430\u0432\u0430\u0442\u044c / \u0434\u0435\u043d\u044c</div>
                        <div class="mv {tempo_cls}">{_fmt2(res['break_even_sales'])} \u0448\u0442.</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u0420\u043e\u0441\u0442 \u0442\u0435\u043c\u043f\u0430 \u043f\u0440\u043e\u0434\u0430\u0436</div>
                        <div class="mv {tempo_cls}">{_fmt2(res['tempo_ratio'])}x</div>
                    </div>
                    <div class="metric-item">
                        <div class="ml">\u0421\u0435\u0431\u0435\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c / \u0435\u0434.</div>
                        <div class="mv">{_fmt2(float(sel_row['cost_per_unit']))} \u20bd</div>
                    </div>
                </div>

                <div style="text-align:center;margin-top:1rem">
                    <span class="verdict-badge" style="background:{vs['bg']};color:{vs['color']};border:2px solid {vs['border']}">
                        {vs['label']}
                    </span>
                    <div class="verdict-desc">{verdict_desc}</div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

        elif calc_clicked and promo_price <= 0:
            st.warning("\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u0446\u0435\u043d\u0443 \u043f\u043e \u0430\u043a\u0446\u0438\u0438 \u0431\u043e\u043b\u044c\u0448\u0435 0")
        else:
            # Placeholder when nothing calculated yet
            st.markdown(
                '<div class="promo-card" style="--accent: #cbd5e1; text-align:center; padding:3rem 1.5rem;">'
                '<div style="font-size:40px;margin-bottom:0.5rem">\U0001f4ca</div>'
                '<div style="font-size:15px;color:#64748b">'
                '\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0430\u0440\u0442\u0438\u043a\u0443\u043b, \u0443\u043a\u0430\u0436\u0438\u0442\u0435 \u0446\u0435\u043d\u0443 \u043f\u043e \u0430\u043a\u0446\u0438\u0438<br>\u0438 \u043d\u0430\u0436\u043c\u0438\u0442\u0435 <b>\u0420\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u0442\u044c</b>'
                '</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# TAB 2 — Batch analysis
# ══════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("### \u041c\u0430\u0441\u0441\u043e\u0432\u044b\u0439 \u0440\u0430\u0441\u0447\u0451\u0442 \u0430\u043a\u0446\u0438\u0439")
    st.caption(
        "\u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u043f\u0440\u043e\u0446\u0435\u043d\u0442 \u0441\u043a\u0438\u0434\u043a\u0438 \u0434\u043b\u044f \u0432\u0441\u0435\u0445 \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u043e\u0432 \u0438\u043b\u0438 \u0432\u0432\u0435\u0434\u0438\u0442\u0435 \u0438\u043d\u0434\u0438\u0432\u0438\u0434\u0443\u0430\u043b\u044c\u043d\u044b\u0435 \u0446\u0435\u043d\u044b \u0432 \u0442\u0430\u0431\u043b\u0438\u0446\u0435."
    )

    col_disc, col_apply = st.columns([1, 1])
    with col_disc:
        global_discount = st.slider(
            "\u0421\u043a\u0438\u0434\u043a\u0430 \u043d\u0430 \u0432\u0441\u0435 \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u044b, %",
            min_value=0, max_value=80, value=20, step=5,
        )
    with col_apply:
        st.write("")
        st.write("")
        apply_batch = st.button("\U0001f680 \u0420\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u0442\u044c \u0434\u043b\u044f \u0432\u0441\u0435\u0445", width="stretch")

    if apply_batch:
        batch_rows = []
        for _, row in df.iterrows():
            cur_p = float(row["avg_price_before_spp"])
            promo_p = round(cur_p * (1 - global_discount / 100), 0)
            if promo_p <= 0:
                continue
            res = _calc_promo(row, promo_p)
            vs = _VERDICT_STYLES[res["verdict_key"]]

            batch_rows.append({
                "supplier_article": row["supplier_article"],
                "nm_id": int(row["nm_id"]),
                "subject": row["subject"],
                "avg_price_before_spp": cur_p,
                "promo_price": promo_p,
                "promo_price_after_spp": res["promo_price_after_spp"],
                "profit_per_unit": float(row["profit_per_unit"]),
                "promo_profit_per_unit": res["promo_profit_per_unit"],
                "current_actual_sales": res["current_actual_sales"],
                "break_even_sales": res["break_even_sales"],
                "tempo_ratio": res["tempo_ratio"],
                "verdict_key": res["verdict_key"],
                "verdict_label": vs["label"],
                "verdict_color": vs["color"],
                "verdict_bg": vs["bg"],
            })

        if not batch_rows:
            st.info("\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u0434\u043b\u044f \u0440\u0430\u0441\u0447\u0451\u0442\u0430")
            st.stop()

        # Summary KPIs
        bdf = pd.DataFrame(batch_rows)
        n_total = len(bdf)
        n_good = len(bdf[bdf["verdict_key"] == "good"])
        n_caution = len(bdf[bdf["verdict_key"] == "caution"])
        n_bad = len(bdf[bdf["verdict_key"].isin(["bad", "loss"])])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("\u0412\u0441\u0435\u0433\u043e \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u043e\u0432", n_total)
        k2.metric("\u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f", n_good)
        k3.metric("\u041e\u0441\u0442\u043e\u0440\u043e\u0436\u043d\u043e", n_caution)
        k4.metric("\u041d\u0435 \u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f", n_bad)

        # Build HTML table
        hdr = (
            "<tr>"
            "<th>#</th>"
            "<th>\u0410\u0440\u0442\u0438\u043a\u0443\u043b</th>"
            "<th>\u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f</th>"
            "<th>\u0426\u0435\u043d\u0430 \u0441\u0435\u0439\u0447\u0430\u0441</th>"
            "<th>\u0426\u0435\u043d\u0430 \u043f\u043e \u0430\u043a\u0446\u0438\u0438</th>"
            "<th>\u0426\u0435\u043d\u0430 \u043f\u043e\u0441\u043b\u0435 \u0421\u041f\u041f</th>"
            "<th>\u041f\u0440\u0438\u0431\u044b\u043b\u044c<br>\u0441\u0435\u0439\u0447\u0430\u0441</th>"
            "<th>\u041f\u0440\u0438\u0431\u044b\u043b\u044c<br>\u043f\u043e \u0430\u043a\u0446\u0438\u0438</th>"
            "<th>\u041f\u0440\u043e\u0434\u0430\u0436\u0438<br>\u0441\u0435\u0439\u0447\u0430\u0441</th>"
            "<th>\u041d\u0443\u0436\u043d\u043e<br>\u043f\u0440\u043e\u0434\u0430\u0432\u0430\u0442\u044c</th>"
            "<th>\u0420\u043e\u0441\u0442<br>\u0442\u0435\u043c\u043f\u0430</th>"
            "<th>\u0412\u0435\u0440\u0434\u0438\u043a\u0442</th>"
            "</tr>"
        )

        rows_html = ""
        for idx, brow in enumerate(batch_rows, 1):
            pcls = "pos" if brow["promo_profit_per_unit"] > 0 else "neg"
            tr = brow["tempo_ratio"]
            tcls = "pos" if tr and tr <= 1.5 else ("warn" if tr and tr <= 3.0 else "neg")

            rows_html += (
                f"<tr>"
                f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
                f'<td style="font-weight:600">{wb_link(brow["nm_id"], brow["supplier_article"])}<br>'
                f'<span style="font-size:10px;color:#94a3b8">{brow["nm_id"]}</span></td>'
                f'<td>{brow["subject"]}</td>'
                f'<td class="num">{_fmt(brow["avg_price_before_spp"])}</td>'
                f'<td class="num" style="font-weight:700">{_fmt(brow["promo_price"])}</td>'
                f'<td class="num">{_fmt(brow["promo_price_after_spp"])}</td>'
                f'<td class="num">{_fmt2(brow["profit_per_unit"])}</td>'
                f'<td class="num {pcls}">{_fmt2(brow["promo_profit_per_unit"])}</td>'
                f'<td class="ctr">{_fmt2(brow["current_actual_sales"])}</td>'
                f'<td class="ctr {tcls}">{_fmt2(brow["break_even_sales"])}</td>'
                f'<td class="ctr {tcls}">{_fmt2(tr)}x</td>'
                f'<td class="ctr"><span class="badge-cell" style="background:{brow["verdict_bg"]};'
                f'color:{brow["verdict_color"]};border:1px solid {brow["verdict_color"]}30">'
                f'{brow["verdict_label"]}</span></td>'
                f"</tr>"
            )

        table_html = (
            f'<div class="art-wrap"><table class="art-t" data-sortable>'
            f"<thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>{SORT_JS}"
        )
        render_table(table_html)

        # CSV export
        export_df = bdf[[
            "supplier_article", "nm_id", "subject",
            "avg_price_before_spp", "promo_price", "promo_price_after_spp",
            "profit_per_unit", "promo_profit_per_unit",
            "current_actual_sales", "break_even_sales", "tempo_ratio",
            "verdict_label",
        ]].rename(columns={
            "supplier_article": "\u0410\u0440\u0442\u0438\u043a\u0443\u043b",
            "nm_id": "nm_id",
            "subject": "\u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f",
            "avg_price_before_spp": "\u0426\u0435\u043d\u0430 \u0441\u0435\u0439\u0447\u0430\u0441",
            "promo_price": "\u0426\u0435\u043d\u0430 \u043f\u043e \u0430\u043a\u0446\u0438\u0438",
            "promo_price_after_spp": "\u0426\u0435\u043d\u0430 \u043f\u043e\u0441\u043b\u0435 \u0421\u041f\u041f",
            "profit_per_unit": "\u041f\u0440\u0438\u0431\u044b\u043b\u044c \u0441\u0435\u0439\u0447\u0430\u0441",
            "promo_profit_per_unit": "\u041f\u0440\u0438\u0431\u044b\u043b\u044c \u043f\u043e \u0430\u043a\u0446\u0438\u0438",
            "current_actual_sales": "\u041f\u0440\u043e\u0434\u0430\u0436\u0438 \u0441\u0435\u0439\u0447\u0430\u0441/\u0434\u0435\u043d\u044c",
            "break_even_sales": "\u041d\u0443\u0436\u043d\u043e \u043f\u0440\u043e\u0434\u0430\u0432\u0430\u0442\u044c/\u0434\u0435\u043d\u044c",
            "tempo_ratio": "\u0420\u043e\u0441\u0442 \u0442\u0435\u043c\u043f\u0430",
            "verdict_label": "\u0412\u0435\u0440\u0434\u0438\u043a\u0442",
        })

        st.download_button(
            "\U0001f4e5 \u0421\u043a\u0430\u0447\u0430\u0442\u044c Excel",
            _to_excel(export_df, sheet_name="Анализ"),
            "promo_analysis.xlsx",
            _XLSX_MIME,
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — Excel upload with individual promo prices
# ════════════════════════════════════════════════════════════���═

with tab_excel:
    st.markdown("### Загрузка цен из Excel")
    st.caption(
        "Загруз��те файл Excel/CSV с двумя колонками: **Артикул** (supplier_article) "
        "и **Цена по акции** (число). Система автоматически рассчитает все показатели."
    )

    # Template download
    template_df = pd.DataFrame({
        "Артикул": df["supplier_article"].head(5).tolist(),
        "Цена по акции": [0] * min(5, len(df)),
    })
    st.download_button(
        "\U0001f4cb Скачать шаблон Excel",
        _to_excel(template_df, sheet_name="Цены"),
        "promo_template.xlsx",
        _XLSX_MIME,
        key="template_dl",
    )

    uploaded = st.file_uploader(
        "Выберите файл Excel или CSV",
        type=["xlsx", "xls", "csv"],
        help="Файл должен содержать колонки 'Артикул' и 'Цена по акции'",
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                udf = pd.read_csv(uploaded)
            else:
                udf = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

        # Normalize column names
        col_map = {}
        for c in udf.columns:
            cl = str(c).strip().lower()
            if cl in ("артикул", "supplier_article", "артикул пос��авщика", "article"):
                col_map[c] = "supplier_article"
            elif cl in ("цена по акции", "цена акции", "promo_price", "цена", "price", "акционная цена"):
                col_map[c] = "promo_price"
        udf = udf.rename(columns=col_map)

        if "supplier_article" not in udf.columns or "promo_price" not in udf.columns:
            st.error(
                "Не найдены нужные колонки. Файл должен содержать:\n"
                "- **Артикул** (или supplier_article)\n"
                "- **Цена по акции** (или promo_price)"
            )
            st.stop()

        udf["supplier_article"] = udf["supplier_article"].astype(str).str.strip()
        udf["promo_price"] = pd.to_numeric(udf["promo_price"], errors="coerce").fillna(0)
        udf = udf[udf["promo_price"] > 0]

        if udf.empty:
            st.warning("Нет строк с ценой по акции > 0")
            st.stop()

        # Match with baseline data
        matched = udf.merge(df, on="supplier_article", how="inner")
        not_found = udf[~udf["supplier_article"].isin(df["supplier_article"])]

        if not not_found.empty:
            st.warning(f"Не найдено {len(not_found)} артикулов: {', '.join(not_found['supplier_article'].head(5).tolist())}")

        if matched.empty:
            st.error("Ни один артикул не найден в базе данных")
            st.stop()

        st.success(f"Найдено {len(matched)} артикулов для расчёта")

        # Calculate promo for each matched row
        excel_rows = []
        for _, row in matched.iterrows():
            promo_p = float(row["promo_price"])
            res = _calc_promo(row, promo_p)
            vs = _VERDICT_STYLES[res["verdict_key"]]
            cur_p = _safe_float(row.get("avg_price_before_spp", 0))
            discount = round((1 - promo_p / cur_p) * 100, 1) if cur_p > 0 else 0

            excel_rows.append({
                "supplier_article": row["supplier_article"],
                "nm_id": int(row["nm_id"]),
                "subject": row.get("subject", ""),
                "avg_price_before_spp": cur_p,
                "promo_price": promo_p,
                "discount_pct": discount,
                "promo_price_after_spp": res["promo_price_after_spp"],
                "profit_per_unit": _safe_float(row.get("profit_per_unit", 0)),
                "promo_profit_per_unit": res["promo_profit_per_unit"],
                "current_actual_sales": res["current_actual_sales"],
                "break_even_sales": res["break_even_sales"],
                "tempo_ratio": res["tempo_ratio"],
                "verdict_key": res["verdict_key"],
                "verdict_label": vs["label"],
                "verdict_color": vs["color"],
                "verdict_bg": vs["bg"],
            })

        edf = pd.DataFrame(excel_rows)

        # Summary KPIs
        n_total = len(edf)
        n_good = len(edf[edf["verdict_key"] == "good"])
        n_caution = len(edf[edf["verdict_key"] == "caution"])
        n_bad = len(edf[edf["verdict_key"].isin(["bad", "loss"])])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Всег�� артикулов", n_total)
        k2.metric("Рекомендуется", n_good)
        k3.metric("Осторожно", n_caution)
        k4.metric("Не рекомендуется", n_bad)

        # HTML table
        hdr = (
            "<tr>"
            "<th>#</th><th>Артикул</th><th>Категория</th>"
            "<th>Цена сейчас</th><th>Цена акции</th><th>Скидка</th>"
            "<th>Цена после СПП</th>"
            "<th>Прибыль\nсейчас</th><th>Прибыль\nпо акции</th>"
            "<th>Продажи\nсейчас</th><th>Нужно\nпродавать</th>"
            "<th>Рост\nтемпа</th><th>Вердикт</th>"
            "</tr>"
        )

        rows_html = ""
        for idx, erow in enumerate(excel_rows, 1):
            pcls = "pos" if erow["promo_profit_per_unit"] > 0 else "neg"
            tr = erow["tempo_ratio"]
            tcls = "pos" if tr and tr <= 1.5 else ("warn" if tr and tr <= 3.0 else "neg")

            rows_html += (
                f"<tr>"
                f'<td class="ctr" style="color:#94a3b8">{idx}</td>'
                f'<td style="font-weight:600">{wb_link(erow["nm_id"], erow["supplier_article"])}<br>'
                f'<span style="font-size:10px;color:#94a3b8">{erow["nm_id"]}</span></td>'
                f'<td>{erow["subject"]}</td>'
                f'<td class="num">{_fmt(erow["avg_price_before_spp"])}</td>'
                f'<td class="num" style="font-weight:700">{_fmt(erow["promo_price"])}</td>'
                f'<td class="ctr">{erow["discount_pct"]:.0f}%</td>'
                f'<td class="num">{_fmt(erow["promo_price_after_spp"])}</td>'
                f'<td class="num">{_fmt2(erow["profit_per_unit"])}</td>'
                f'<td class="num {pcls}">{_fmt2(erow["promo_profit_per_unit"])}</td>'
                f'<td class="ctr">{_fmt2(erow["current_actual_sales"])}</td>'
                f'<td class="ctr {tcls}">{_fmt2(erow["break_even_sales"])}</td>'
                f'<td class="ctr {tcls}">{_fmt2(tr)}x</td>'
                f'<td class="ctr"><span class="badge-cell" style="background:{erow["verdict_bg"]};'
                f'color:{erow["verdict_color"]};border:1px solid {erow["verdict_color"]}30">'
                f'{erow["verdict_label"]}</span></td>'
                f"</tr>"
            )

        table_html = (
            f'<div class="art-wrap"><table class="art-t" data-sortable>'
            f"<thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>{SORT_JS}"
        )
        render_table(table_html)

        # Export
        export_edf = edf[[
            "supplier_article", "nm_id", "subject",
            "avg_price_before_spp", "promo_price", "discount_pct",
            "promo_price_after_spp", "profit_per_unit", "promo_profit_per_unit",
            "current_actual_sales", "break_even_sales", "tempo_ratio", "verdict_label",
        ]].copy()
        export_edf.columns = [
            "Артикул", "nm_id", "Категория", "Цена сейчас", "Цена акции",
            "Скидка %", "Цена после СПП", "Прибыль сейчас", "Прибыль по акции",
            "Продажи/день", "Нужно продавать/день", "Рост темпа", "Вердикт",
        ]

        st.download_button(
            "\U0001f4e5 Скачать результат Excel",
            _to_excel(export_edf, sheet_name="Результат"),
            "promo_excel_result.xlsx",
            _XLSX_MIME,
            key="excel_result_dl",
        )


# ══════════════════════════════════════════════════════════════
# TAB 4 — Ручной ввод (что-если без данных артикула)
# ══════════════════════════════════════════════════════════════

with tab_manual:
    st.markdown("### Что-если калькулятор (ручной ввод)")
    st.caption(
        "Введите цифры вручную — без привязки к существующему артикулу. "
        "Удобно для проверки гипотез «а что, если закупка подешевеет» или "
        "«а что, если СПП будет 25%, а не 18%»."
    )

    mcol_a, mcol_b = st.columns(2)
    with mcol_a:
        st.markdown("#### Текущая ситуация")
        m_price_before = st.number_input(
            "Цена до СПП, ₽", min_value=0.0, value=1500.0, step=10.0,
            format="%.0f", key="m_price_before",
        )
        m_spp_pct = st.number_input(
            "СПП, %", min_value=0.0, max_value=100.0, value=18.0, step=0.5,
            format="%.1f", key="m_spp_pct",
        )
        m_cost = st.number_input(
            "Себестоимость, ₽/ед.", min_value=0.0, value=400.0, step=10.0,
            format="%.0f", key="m_cost",
        )
        m_logistics = st.number_input(
            "Логистика, ₽/ед.", min_value=0.0, value=70.0, step=5.0,
            format="%.0f", key="m_logistics",
        )
        m_commission_pct = st.number_input(
            "Комиссия WB, %", min_value=0.0, max_value=50.0, value=17.0, step=0.5,
            format="%.1f", key="m_commission_pct",
        )
        m_orders_day = st.number_input(
            "Среднее заказов / день, шт.", min_value=0.0, value=5.0, step=0.5,
            format="%.1f", key="m_orders_day",
        )
        m_buyout = st.number_input(
            "Процент выкупа, %", min_value=1.0, max_value=100.0, value=85.0, step=1.0,
            format="%.0f", key="m_buyout",
        )

    with mcol_b:
        st.markdown("#### Промо-сценарий")
        m_promo_price = st.number_input(
            "Промо-цена до СПП, ₽", min_value=0.0,
            value=round(m_price_before * 0.8, 0),
            step=10.0, format="%.0f", key="m_promo_price",
        )
        m_discount_pct = (1 - m_promo_price / m_price_before) * 100 if m_price_before > 0 else 0
        st.caption(f"Скидка: **{m_discount_pct:.1f}%**")

        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Current unit economics
        cur_price_after_spp = m_price_before * (1 - m_spp_pct / 100)
        cur_commission = cur_price_after_spp * m_commission_pct / 100
        cur_profit_unit = cur_price_after_spp - cur_commission - m_logistics - m_cost
        cur_daily_profit = cur_profit_unit * m_orders_day * m_buyout / 100

        # Promo unit economics
        promo_price_after_spp = m_promo_price * (1 - m_spp_pct / 100)
        promo_commission = promo_price_after_spp * m_commission_pct / 100
        promo_profit_unit = promo_price_after_spp - promo_commission - m_logistics - m_cost

        if promo_profit_unit > 0 and cur_daily_profit > 0:
            break_even_sales = cur_daily_profit / promo_profit_unit
            tempo_ratio = break_even_sales / (m_orders_day * m_buyout / 100) if m_orders_day > 0 else float("inf")
        else:
            break_even_sales = None
            tempo_ratio = float("inf")

        verdict_key = _classify(promo_profit_unit, tempo_ratio)
        vs = _VERDICT_STYLES[verdict_key]

        st.markdown(
            f'<div class="promo-card" style="--accent: {vs["border"]}; margin-top: 0;">'
            f'<div class="card-title">Результат</div>'
            f'<div class="metrics-grid">'
            f'<div class="metric-item"><div class="ml">Цена после СПП (текущая)</div>'
            f'<div class="mv">{_fmt(cur_price_after_spp)} ₽</div></div>'
            f'<div class="metric-item"><div class="ml">Цена после СПП (промо)</div>'
            f'<div class="mv">{_fmt(promo_price_after_spp)} ₽</div></div>'
            f'<div class="metric-item"><div class="ml">Прибыль/ед. сейчас</div>'
            f'<div class="mv {"green" if cur_profit_unit > 0 else "red"}">{_fmt2(cur_profit_unit)} ₽</div></div>'
            f'<div class="metric-item"><div class="ml">Прибыль/ед. по промо</div>'
            f'<div class="mv {"green" if promo_profit_unit > 0 else "red"}">{_fmt2(promo_profit_unit)} ₽</div></div>'
            f'<div class="metric-item"><div class="ml">Нужно продавать/день</div>'
            f'<div class="mv">{_fmt2(break_even_sales) if break_even_sales else "—"} шт.</div></div>'
            f'<div class="metric-item"><div class="ml">Рост темпа</div>'
            f'<div class="mv">{_fmt2(tempo_ratio) if tempo_ratio != float("inf") else "∞"}x</div></div>'
            f'</div>'
            f'<div style="text-align:center;margin-top:1rem">'
            f'<span class="verdict-badge" style="background:{vs["bg"]};color:{vs["color"]};border:2px solid {vs["border"]}">'
            f'{vs["label"]}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Tests ────────────────────────────────────────────────
    with st.expander("🧪 Проверить формулы (тест)", expanded=False):
        st.markdown(
            """
            Проверим, что расчёт верный на простом примере:

            - Цена до СПП = 1000 ₽, СПП = 20 % → цена после СПП = 800 ₽
            - Комиссия 15 % от 800 = 120 ₽
            - Логистика 50 ₽, себест. 300 ₽
            - Прибыль/ед. = 800 − 120 − 50 − 300 = **330 ₽**

            Промо-цена 800 ₽ → после СПП = 640 ₽, комиссия 15 % = 96 ₽
            → прибыль/ед. = 640 − 96 − 50 − 300 = **194 ₽**

            Если продавалось 10 шт/день × 80 % выкупа = 8 шт/день
            → текущая дневная прибыль = 330 × 8 = 2 640 ₽
            → нужно продавать 2 640 / 194 ≈ **13,6 шт/день**
            → рост темпа ≈ 13,6 / 8 ≈ **1,70×**
            """
        )
        if st.button("Провести тест", key="run_test"):
            # Test case
            tp = 1000.0
            ts = 20.0
            tc = 300.0
            tl = 50.0
            tcm = 15.0
            to = 10.0
            tb = 80.0
            tpromo = 800.0

            test_price_post = tp * (1 - ts / 100)
            test_commission = test_price_post * tcm / 100
            test_profit_unit = test_price_post - test_commission - tl - tc
            test_daily = test_profit_unit * to * tb / 100

            test_promo_post = tpromo * (1 - ts / 100)
            test_promo_comm = test_promo_post * tcm / 100
            test_promo_profit = test_promo_post - test_promo_comm - tl - tc
            test_break_even = test_daily / test_promo_profit if test_promo_profit > 0 else 0
            test_tempo = test_break_even / (to * tb / 100)

            exp_profit = 330.0
            exp_promo_profit = 194.0
            exp_break_even = 13.61  # approx
            exp_tempo = 1.70

            ok_profit = abs(test_profit_unit - exp_profit) < 0.1
            ok_promo = abs(test_promo_profit - exp_promo_profit) < 0.1
            ok_be = abs(test_break_even - exp_break_even) < 0.1
            ok_tempo = abs(test_tempo - exp_tempo) < 0.01

            st.markdown(f"""
            | Показатель | Ожидается | Получено | ✓/✗ |
            |:-----------|:---------:|:--------:|:---:|
            | Прибыль/ед. сейчас | {exp_profit:.2f} ₽ | {test_profit_unit:.2f} ₽ | {"✅" if ok_profit else "❌"} |
            | Прибыль/ед. по промо | {exp_promo_profit:.2f} ₽ | {test_promo_profit:.2f} ₽ | {"✅" if ok_promo else "❌"} |
            | Нужно продавать/день | {exp_break_even:.2f} шт | {test_break_even:.2f} шт | {"✅" if ok_be else "❌"} |
            | Рост темпа | {exp_tempo:.2f}x | {test_tempo:.2f}x | {"✅" if ok_tempo else "❌"} |
            """)
            if all([ok_profit, ok_promo, ok_be, ok_tempo]):
                st.success("Все формулы работают корректно ✅")
            else:
                st.error("Один из тестов не прошёл ❌ — проверьте логику расчётов")
