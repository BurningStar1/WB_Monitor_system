"""Калькулятор акций — оценка целесообразности участия в промо-акциях WB."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from marts import fetch_dataframe, PROMO_BASELINE_QUERY
from styles import inject_global_styles, format_currency, format_pct

# ── Page setup ────────────────────────────────────────────────

inject_global_styles()
st.title("\U0001f3f7 \u041a\u0430\u043b\u044c\u043a\u0443\u043b\u044f\u0442\u043e\u0440 \u0430\u043a\u0446\u0438\u0439")

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


def _calc_promo(row: pd.Series, promo_price: float) -> dict:
    """Calculate all promo metrics for a single article."""
    avg_spp_pct = float(row["avg_spp_pct"])
    avg_price_after = float(row["avg_price_after_spp"])
    commission_per_unit = float(row["commission_per_unit"])
    cost_per_unit = float(row["cost_per_unit"])
    profit_per_unit = float(row["profit_per_unit"])
    avg_orders_day = float(row["avg_orders_day"])
    buyout_pct = float(row["buyout_pct"])

    promo_price_after_spp = promo_price * (1 - avg_spp_pct / 100)

    # Proportional commission
    if avg_price_after > 0:
        promo_commission = promo_price_after_spp * (commission_per_unit / avg_price_after)
    else:
        promo_commission = 0

    promo_profit_per_unit = promo_price_after_spp - promo_commission - cost_per_unit

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

df = fetch_dataframe(PROMO_BASELINE_QUERY, {})

if df.empty:
    st.info("\u041d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445 \u043f\u043e \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u0430\u043c \u0437\u0430 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0435 30 \u0434\u043d\u0435\u0439")
    st.stop()

# Build display label for selectbox
df["_label"] = df["supplier_article"].astype(str) + "  |  " + df["nm_id"].astype(str)

# ── Tabs ──────────────────────────────────────────────────────

tab_single, tab_batch = st.tabs([
    "\U0001f50d \u0420\u0430\u0441\u0447\u0451\u0442 \u043f\u043e \u0430\u0440\u0442\u0438\u043a\u0443\u043b\u0443",
    "\U0001f4ca \u041c\u0430\u0441\u0441\u043e\u0432\u044b\u0439 \u0430\u043d\u0430\u043b\u0438\u0437",
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

        calc_clicked = st.button("\U0001f4b0 \u0420\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u0442\u044c", use_container_width=True)

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
        apply_batch = st.button("\U0001f680 \u0420\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u0442\u044c \u0434\u043b\u044f \u0432\u0441\u0435\u0445", use_container_width=True)

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
                f'<td style="font-weight:600">{brow["supplier_article"]}<br>'
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
            f'<div class="art-wrap"><table class="art-t">'
            f"<thead>{hdr}</thead><tbody>{rows_html}</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

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
            "\U0001f4e5 \u0421\u043a\u0430\u0447\u0430\u0442\u044c CSV",
            export_df.to_csv(index=False).encode("utf-8-sig"),
            "promo_analysis.csv",
            "text/csv",
        )
