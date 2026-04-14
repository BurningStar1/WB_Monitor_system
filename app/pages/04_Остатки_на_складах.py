"""Остатки на складах — отчёт с капитализацией и аналитикой по складам."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from datetime import date, timedelta
import numpy as np
from marts import fetch_dataframe, STOCKS_QUERY, STOCKS_BY_WH_QUERY, ORDERS_DAILY_AMOUNT_QUERY
from styles import plotly_defaults,  inject_global_styles, fmt_number, fmt_pct_tbl, table_css, PLOTLY_LAYOUT, PLOTLY_COLORS, SORT_JS, wb_link, render_table, export_buttons
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("\U0001f3ed Остатки на складах")

# ── Formatting helpers ───────────────────────────────────────


def _fmt_price(v):
    if pd.isna(v) or v == 0:
        return ""
    return fmt_number(v) + " \u20bd"


TABLE_CSS = table_css("stk") + '<style>.stk .zero-row{background:#fff5f5 !important}</style>'

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

# ── Days of supply (avg daily orders over last 30 days) ─────
_dos_to = date.today()
_dos_from = _dos_to - timedelta(days=29)
_ord = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, {"d_from": str(_dos_from), "d_to": str(_dos_to)})
if not _ord.empty and "nm_id" in _ord.columns:
    _avg = (
        _ord.groupby("nm_id")["orders_count"].sum().div(30.0)
        .reset_index(name="avg_daily_orders")
    )
    df = df.merge(_avg, on="nm_id", how="left")
else:
    df["avg_daily_orders"] = 0.0
df["avg_daily_orders"] = df["avg_daily_orders"].fillna(0.0)
df["days_of_supply"] = np.where(
    df["avg_daily_orders"] > 0, df["quantity_full"] / df["avg_daily_orders"], np.inf
)

# ── Sidebar filters ──────────────────────────────────────────

warehouses = sorted(df["warehouse_name"].dropna().unique()) if "warehouse_name" in df.columns else []
brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
subjects = sorted(df["subject"].dropna().unique()) if "subject" in df.columns else []
_fc1, _fc2, _fc3 = st.columns(3)
with _fc1:
    sel_wh = st.multiselect("Склад", warehouses, default=[])
with _fc2:
    sel_brand = st.multiselect("Бренд", brands, default=[])
with _fc3:
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
            avg_daily_orders=("avg_daily_orders", "first"),
        )
        .sort_values("quantity_full", ascending=False)
    )
    agg["days_of_supply"] = np.where(
        agg["avg_daily_orders"] > 0,
        agg["quantity_full"] / agg["avg_daily_orders"],
        np.inf,
    )

    hdr = (
        "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
        "<th>Остаток</th><th>Ср./день</th><th>Запас, дн</th>"
        "<th>В пути к клиенту</th><th>В пути от клиента</th>"
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
        avg_d = float(r.get("avg_daily_orders", 0) or 0)
        dos = r.get("days_of_supply", np.inf)
        if np.isinf(dos) or np.isnan(dos):
            dos_txt = "∞" if qty > 0 else ""
            dos_cls = ""
        else:
            dos_txt = f"{dos:.1f}"
            if dos < 7:
                dos_cls = "neg"
            elif dos > 60:
                dos_cls = "pos"
            else:
                dos_cls = ""
        rows += (
            f"<tr{row_cls}>"
            f'<td class="ctr">{idx}</td>'
            f"<td>{wb_link(r['nm_id'], r['supplier_article'])}</td>"
            f"<td>{r['subject']}</td>"
            f"<td>{r['brand']}</td>"
            f'<td class="num">{fmt_number(qty)}</td>'
            f'<td class="num">{avg_d:.1f}</td>'
            f'<td class="num {dos_cls}"><b>{dos_txt}</b></td>'
            f'<td class="num">{fmt_number(way_c)}</td>'
            f'<td class="num">{fmt_number(way_f)}</td>'
            f'<td class="num">{_fmt_price(r["price"])}</td>'
            f'<td class="ctr">{fmt_pct_tbl(r["discount"])}</td>'
            f'<td class="num">{_fmt_price(cost)}</td>'
            f"</tr>"
        )

    foot = (
        f"<tr><td></td><td colspan='3'><b>ИТОГО</b></td>"
        f'<td class="num"><b>{fmt_number(t_qty)}</b></td>'
        f"<td></td><td></td>"
        f'<td class="num"><b>{fmt_number(t_way_c)}</b></td>'
        f'<td class="num"><b>{fmt_number(t_way_f)}</b></td>'
        f"<td></td><td></td>"
        f'<td class="num"><b>{_fmt_price(t_cost)}</b></td></tr>'
    )

    html = (
        f'{TABLE_CSS}<div class="stk-wrap"><table class="stk" data-sortable>'
        f"<thead>{hdr}</thead><tbody>{rows}</tbody>"
        f"<tfoot>{foot}</tfoot></table></div>{SORT_JS}"
    )
    render_table(html)
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

        # Apply same filters (including subject)
        if sel_subj and "subject" in wh_df.columns:
            wh_df = wh_df[wh_df["subject"].isin(sel_subj)]
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
                color_discrete_sequence=[PLOTLY_COLORS["indigo"]],
            )
            fig.update_traces(
                marker=dict(line=dict(width=0.5, color="#4f46e5")),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Остаток: %{x:,.0f} шт.<extra></extra>"
                ),
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=20, t=10, b=10),
                height=max(350, len(wh_agg) * 34),
                bargap=0.25,
            )
            plotly_defaults(fig)
            st.plotly_chart(fig, width="stretch")

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
                    f'<td class="num">{fmt_number(qty)}</td>'
                    f'<td class="ctr">{share:.1f}%</td></tr>'
                )
            wh_foot = (
                f'<tr><td></td><td><b>ИТОГО</b></td>'
                f'<td class="num"><b>{fmt_number(grand_qty)}</b></td>'
                f'<td class="ctr"><b>100%</b></td></tr>'
            )
            wh_html = (
                f'{TABLE_CSS}<div class="stk-wrap"><table class="stk" data-sortable>'
                f"<thead>{wh_hdr}</thead><tbody>{wh_rows}</tbody>"
                f"<tfoot>{wh_foot}</tfoot></table></div>{SORT_JS}"
            )
            render_table(wh_html)

# ── Export ────────────────────────────────────────────────────

export_buttons(filt, "stocks_report", sheet_name="Stocks")
