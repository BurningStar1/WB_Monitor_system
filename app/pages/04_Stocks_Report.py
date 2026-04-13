"""Остатки на складах — отчёт с капитализацией и аналитикой по складам."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from marts import fetch_dataframe, STOCKS_QUERY, STOCKS_BY_WH_QUERY
from styles import inject_global_styles
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("\U0001f3ed Остатки на складах")

# ── Formatting helpers ───────────────────────────────────────


def _fmt(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:,.0f}".replace(",", " ")


def _fmt_pct(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:.0f}%"


def _fmt_price(v):
    if pd.isna(v) or v == 0:
        return ""
    return f"{v:,.0f}".replace(",", " ") + " \u20bd"


# ── CSS ──────────────────────────────────────────────────────

TABLE_CSS = """
<style>
.stk-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin:1rem 0;border:1px solid #e2e8f0}
.stk{border-collapse:collapse;width:100%;font-size:12px;font-family:Inter,system-ui,sans-serif;
  background:#fff;color:#1e293b}
.stk th{background:#f1f5f9;padding:8px 10px;border-bottom:2px solid #cbd5e1;
  border-right:1px solid #e2e8f0;font-weight:600;font-size:11px;color:#475569;
  text-align:center;white-space:nowrap}
.stk td{padding:6px 10px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  white-space:nowrap;font-size:12px}
.stk tbody tr:nth-child(even){background:#fafbfc}
.stk tbody tr:hover{background:#eef2ff}
.stk .num{text-align:right}
.stk .ctr{text-align:center}
.stk .zero-row{background:#fff5f5 !important}
.stk .pos{color:#16a34a;font-weight:700}
.stk .neg{color:#dc2626;font-weight:700}
.stk tfoot td{background:#f1f5f9;font-weight:700;border-top:2px solid #cbd5e1}
</style>
"""

# ── Load data ────────────────────────────────────────────────

df = fetch_dataframe(STOCKS_QUERY)

if df.empty:
    st.info("Нет данных об остатках")
    st.stop()

# Ensure numeric columns
for col in ("quantity_full", "quantity", "in_way_to_client",
            "in_way_from_client", "price", "discount"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Capitalization column
df["cost_value"] = df["quantity_full"] * df["price"] * (1 - df["discount"] / 100)

# ── Sidebar filters ──────────────────────────────────────────

warehouses = sorted(df["warehouse_name"].dropna().unique()) if "warehouse_name" in df.columns else []
brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
subjects = sorted(df["subject"].dropna().unique()) if "subject" in df.columns else []

with st.sidebar:
    st.header("Фильтры")
    sel_wh = st.multiselect("Склад", warehouses, default=[])
    sel_brand = st.multiselect("Бренд", brands, default=[])
    sel_subj = st.multiselect("Предмет", subjects, default=[])

filt = df.copy()
if sel_wh:
    filt = filt[filt["warehouse_name"].isin(sel_wh)]
if sel_brand:
    filt = filt[filt["brand"].isin(sel_brand)]
if sel_subj:
    filt = filt[filt["subject"].isin(sel_subj)]

if filt.empty:
    st.warning("Нет данных по выбранным фильтрам")
    st.stop()

# ── KPI cards ────────────────────────────────────────────────

unique_articles = filt["supplier_article"].nunique() if "supplier_article" in filt.columns else len(filt)
total_qty = int(filt["quantity_full"].sum())
in_way_client = int(filt["in_way_to_client"].sum())
capitalization = filt["cost_value"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Позиций", f"{unique_articles:,}".replace(",", " "))
c2.metric("Общий остаток", f"{total_qty:,}".replace(",", " ") + " шт.")
c3.metric("В пути к клиенту", f"{in_way_client:,}".replace(",", " ") + " шт.")
c4.metric("Капитализация", f"{capitalization:,.0f}".replace(",", " ") + " \u20bd")

# ── Tabs ─────────────────────────────────────────────────────

tab_articles, tab_warehouses = st.tabs(["По артикулам", "По складам"])

# ══════════════════════════════════════════════════════════════
# TAB 1: By articles
# ══════════════════════════════════════════════════════════════

with tab_articles:
    agg = (
        filt.groupby(["nm_id", "supplier_article"], as_index=False)
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            quantity_full=("quantity_full", "sum"),
            in_way_to_client=("in_way_to_client", "sum"),
            in_way_from_client=("in_way_from_client", "sum"),
            price=("price", "first"),
            discount=("discount", "first"),
            cost_value=("cost_value", "sum"),
        )
        .sort_values("quantity_full", ascending=False)
    )

    hdr = (
        "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
        "<th>Остаток</th><th>В пути к клиенту</th><th>В пути от клиента</th>"
        "<th>Цена</th><th>Скидка%</th><th>Стоимость</th></tr>"
    )

    rows = ""
    t_qty = t_way_c = t_way_f = t_cost = 0
    for idx, (_, r) in enumerate(agg.iterrows(), start=1):
        qty = int(r["quantity_full"])
        way_c = int(r["in_way_to_client"])
        way_f = int(r["in_way_from_client"])
        cost = float(r["cost_value"])
        t_qty += qty
        t_way_c += way_c
        t_way_f += way_f
        t_cost += cost

        row_cls = ' class="zero-row"' if qty == 0 else ""
        rows += (
            f"<tr{row_cls}>"
            f'<td class="ctr">{idx}</td>'
            f"<td>{r['supplier_article']}</td>"
            f"<td>{r['subject']}</td>"
            f"<td>{r['brand']}</td>"
            f'<td class="num">{_fmt(qty)}</td>'
            f'<td class="num">{_fmt(way_c)}</td>'
            f'<td class="num">{_fmt(way_f)}</td>'
            f'<td class="num">{_fmt_price(r["price"])}</td>'
            f'<td class="ctr">{_fmt_pct(r["discount"])}</td>'
            f'<td class="num">{_fmt_price(cost)}</td>'
            f"</tr>"
        )

    foot = (
        f"<tr><td></td><td colspan='3'><b>ИТОГО</b></td>"
        f'<td class="num"><b>{_fmt(t_qty)}</b></td>'
        f'<td class="num"><b>{_fmt(t_way_c)}</b></td>'
        f'<td class="num"><b>{_fmt(t_way_f)}</b></td>'
        f"<td></td><td></td>"
        f'<td class="num"><b>{_fmt_price(t_cost)}</b></td></tr>'
    )

    html = (
        f'{TABLE_CSS}<div class="stk-wrap"><table class="stk">'
        f"<thead>{hdr}</thead><tbody>{rows}</tbody>"
        f"<tfoot>{foot}</tfoot></table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    st.caption(f"Строк: {len(agg)}")

# ══════════════════════════════════════════════════════════════
# TAB 2: By warehouse
# ══════════════════════════════════════════════════════════════

with tab_warehouses:
    wh_df = fetch_dataframe(STOCKS_BY_WH_QUERY)

    if wh_df.empty:
        st.info("Нет данных по складам")
    else:
        for col in ("quantity_full", "quantity"):
            if col in wh_df.columns:
                wh_df[col] = pd.to_numeric(wh_df[col], errors="coerce").fillna(0)
        # share_pct comes from the view; cost_value not available at warehouse level
        if "share_pct" in wh_df.columns:
            wh_df["share_pct"] = pd.to_numeric(wh_df["share_pct"], errors="coerce").fillna(0)

        # Apply same filters
        if sel_wh:
            wh_df = wh_df[wh_df["warehouse_name"].isin(sel_wh)]
        if sel_brand and "brand" in wh_df.columns:
            wh_df = wh_df[wh_df["brand"].isin(sel_brand)]

        wh_agg = (
            wh_df.groupby("warehouse_name", as_index=False)
            .agg(quantity_full=("quantity_full", "sum"))
            .sort_values("quantity_full", ascending=False)
        )

        if not wh_agg.empty:
            # Bar chart
            fig = px.bar(
                wh_agg,
                y="warehouse_name",
                x="quantity_full",
                orientation="h",
                labels={"warehouse_name": "Склад", "quantity_full": "Остаток, шт"},
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=20, t=10, b=10),
                height=max(250, len(wh_agg) * 32),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Warehouse table
            wh_hdr = (
                "<tr><th>#</th><th>Склад</th><th>Остаток, шт</th>"
                "<th>Доля, %</th></tr>"
            )
            wh_rows = ""
            grand_qty = int(wh_agg["quantity_full"].sum())
            for idx, (_, r) in enumerate(wh_agg.iterrows(), start=1):
                qty = int(r["quantity_full"])
                share = (qty / grand_qty * 100) if grand_qty else 0
                wh_rows += (
                    f'<tr><td class="ctr">{idx}</td>'
                    f"<td>{r['warehouse_name']}</td>"
                    f'<td class="num">{_fmt(qty)}</td>'
                    f'<td class="ctr">{share:.1f}%</td></tr>'
                )
            wh_foot = (
                f'<tr><td></td><td><b>ИТОГО</b></td>'
                f'<td class="num"><b>{_fmt(grand_qty)}</b></td>'
                f'<td class="ctr"><b>100%</b></td></tr>'
            )
            wh_html = (
                f'{TABLE_CSS}<div class="stk-wrap"><table class="stk">'
                f"<thead>{wh_hdr}</thead><tbody>{wh_rows}</tbody>"
                f"<tfoot>{wh_foot}</tfoot></table></div>"
            )
            st.markdown(wh_html, unsafe_allow_html=True)

# ── CSV download ─────────────────────────────────────────────

st.download_button(
    "\U0001f4e5 Скачать CSV",
    filt.to_csv(index=False).encode("utf-8-sig"),
    "stocks_report.csv",
    "text/csv",
)
