"""Потребность в поставках — расчёт необходимого объёма закупок."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from marts import fetch_dataframe, SUPPLY_NEEDS_QUERY
from styles import inject_global_styles, fmt_number, fmt_pct_tbl, table_css, PLOTLY_LAYOUT, SORT_JS, render_table, export_buttons
from auth import check_auth, logout

# ── Helpers ───────────────────────────────────────────────────



def _status_badge(status: str) -> str:
    """Return colored HTML badge for stock status."""
    colors = {
        "Критично": ("#dc2626", "#fef2f2", "#fecaca"),
        "Низкий запас": ("#ea580c", "#fff7ed", "#fed7aa"),
        "Норма": ("#16a34a", "#f0fdf4", "#bbf7d0"),
        "Достаточно": ("#2563eb", "#eff6ff", "#bfdbfe"),
    }
    fg, bg, border = colors.get(status, ("#64748b", "#f8fafc", "#e2e8f0"))
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'font-size:10px;font-weight:700;color:{fg};background:{bg};'
        f'border:1px solid {border};white-space:nowrap">{status}</span>'
    )


def _days_color(days) -> str:
    """Return CSS color for days_of_stock value."""
    if pd.isna(days) or days == 0:
        return "#dc2626"
    if days < 7:
        return "#dc2626"
    if days < 15:
        return "#ea580c"
    if days < 30:
        return "#16a34a"
    return "#2563eb"


# ── Page setup ────────────────────────────────────────────────

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("📦 Потребность в поставках")

with st.expander("ℹ️ Как рассчитывается потребность", expanded=False):
    st.markdown(
        """
        **Цель отчёта** — понять, сколько товара закупить/отгрузить,
        чтобы остатка хватило на заданный горизонт.

        **Формула:**
        `Потребность = (Целевой_запас_дней × Средний_заказ_в_день) − Текущий_остаток`

        **Целевой запас** — на сколько дней продаж должно хватать остатка
        (по умолчанию 30). Включает в себя:
        - время доставки на склад WB (~3–7 дней)
        - страховой запас (~7 дней)
        - оборот до следующей поставки

        **Статусы артикулов:**
        - 🔴 **Критично** — <7 дней запаса, срочно нужна поставка
        - 🟠 **Низкий запас** — 7–15 дней
        - 🟢 **Норма** — 15–30 дней (целевой уровень)
        - 🔵 **Достаточно** — >30 дней, возможно, избыток

        Фильтр **«Мин. заказов/день»** отсекает слабо продающиеся артикулы.
        """
    )

# ── Filters ──────────────────────────────────────────────────
fcol1, fcol2 = st.columns(2)
with fcol1:
    target_days = st.number_input(
        "Целевой запас (дней)", min_value=1, max_value=180, value=30, step=1,
        help="На сколько дней продаж должно хватить остатка",
    )
with fcol2:
    min_orders = st.number_input(
        "Мин. заказов/день", min_value=0.0, max_value=100.0, value=0.5, step=0.1,
        help="Отсечь артикулы с низким спросом",
    )

# ── Load data ─────────────────────────────────────────────────

df = fetch_dataframe(SUPPLY_NEEDS_QUERY, {})

if df.empty:
    st.info("Нет данных для расчёта потребностей в поставках")
    st.stop()

# ── Entity filters ────────────────────────────────────────────

brands = sorted(df["brand"].dropna().unique()) if "brand" in df.columns else []
subjects = sorted(df["subject"].dropna().unique()) if "subject" in df.columns else []
fcol3, fcol4 = st.columns(2)
with fcol3:
    sel_brands = st.multiselect("Бренд", brands)
with fcol4:
    sel_subjects = st.multiselect("Предмет", subjects)

if sel_brands:
    df = df[df["brand"].isin(sel_brands)]
if sel_subjects:
    df = df[df["subject"].isin(sel_subjects)]

# Filter by minimum orders/day
df = df[df["avg_orders_day"] >= min_orders].copy()

if df.empty:
    st.info("Нет артикулов, удовлетворяющих фильтрам")
    st.stop()

# ── Calculated columns ────────────────────────────────────────

# Ensure numeric
for col in ["avg_orders_day", "avg_sales_day", "buyout_pct",
            "current_stock", "days_of_stock", "unit_cost"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Need to supply = max(0, target_days * avg_orders_day - current_stock)
df["need_to_supply"] = (target_days * df["avg_orders_day"] - df["current_stock"]).clip(lower=0).round(0).astype(int)

# Supply cost
df["supply_cost"] = (df["need_to_supply"] * df["unit_cost"]).round(0)

# Status
def _assign_status(row):
    d = row["days_of_stock"]
    if pd.isna(d) or d == 0 or d < 7:
        return "Критично"
    if d < 15:
        return "Низкий запас"
    if d < target_days:
        return "Норма"
    return "Достаточно"

df["status"] = df.apply(_assign_status, axis=1)

# Sort by days_of_stock ascending (most urgent first), NaN/0 first
df["_sort_key"] = df["days_of_stock"].replace(0, -1).fillna(-1)
df = df.sort_values("_sort_key", ascending=True).reset_index(drop=True)
df = df.drop(columns=["_sort_key"])

# ── KPI cards ─────────────────────────────────────────────────

total_articles = len(df)
need_supply = int((df["need_to_supply"] > 0).sum())
critical = int(((df["days_of_stock"] < 7) | (df["days_of_stock"] == 0) | df["days_of_stock"].isna()).sum())
total_supply_cost = df["supply_cost"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Всего артикулов", f"{total_articles:,}".replace(",", " "))
c2.metric("Требуют поставки", f"{need_supply:,}".replace(",", " "))
c3.metric("Критичные", f"{critical:,}".replace(",", " "))
c4.metric("Сумма поставки", f"{total_supply_cost:,.0f} \u20bd".replace(",", " "))

_no_cost = int((df["unit_cost"] == 0).sum())
_cost_note = f"  \n⚠️ У **{_no_cost}** артикулов не указана себестоимость — сумма поставки занижена." if _no_cost else ""
st.caption(
    f"* Расчёт на основе средних заказов за 30 дней, целевой запас — {target_days} дн.{_cost_note}"
)

# ── Pagination ────────────────────────────────────────────────

c_pg1, c_pg2, c_pg3 = st.columns([1, 1, 4])
with c_pg1:
    page_size = st.selectbox("Строк", [10, 25, 50, 100], index=0, label_visibility="collapsed")
total_rows = len(df)
total_pages = max((total_rows - 1) // page_size + 1, 1)
with c_pg2:
    page = st.number_input(
        "Стр.", min_value=1, max_value=total_pages, value=1, label_visibility="collapsed"
    )
start_idx = (page - 1) * page_size
end_idx = min(start_idx + page_size, total_rows)
display = df.iloc[start_idx:end_idx]

# ── Build HTML table ──────────────────────────────────────────

TABLE_CSS = """
<style>
.art-wrap{overflow-x:auto;border-radius:12px;box-shadow:0 2px 12px rgba(15,23,42,.08);
  margin-bottom:1rem;border:1px solid #e2e8f0}
.art-t{border-collapse:collapse;width:max-content;min-width:100%;
  font-size:12px;font-family:Inter,system-ui,sans-serif;background:#fff;color:#1e293b}

/* Header */
.art-t thead th{background:#f1f5f9;position:sticky;top:0;z-index:3;
  padding:6px 8px;border-bottom:2px solid #cbd5e1;border-right:1px solid #e2e8f0;
  font-weight:600;font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.3px;
  text-align:center;white-space:nowrap;vertical-align:bottom}
.art-t thead th:last-child{border-right:none}

/* Cells */
.art-t td{border-bottom:1px solid #f1f5f9;border-right:1px solid #f8fafc;
  padding:5px 8px;white-space:nowrap;vertical-align:middle}
.art-t td:last-child{border-right:none}

/* Zebra + hover */
.art-t tbody tr:nth-child(even){background:#fafbfc}
.art-t tbody tr:hover{background:#eef2ff}

/* Alignment helpers */
.art-t .num{text-align:right}.art-t .ctr{text-align:center}

/* Row number */
.art-t .rn{color:#94a3b8;font-size:11px;text-align:center;min-width:24px}

/* Summary / totals row */
.art-t tfoot td{background:#f1f5f9;font-weight:700;font-size:12px;
  border-top:2px solid #cbd5e1;padding:7px 8px;color:#1e293b}
</style>
"""

# ── Header ────────────────────────────────────────────────────

hdr = "<tr>"
hdr += "<th>#</th>"
hdr += "<th>Артикул</th>"
hdr += "<th>Предмет</th>"
hdr += "<th>Ср. заказов/<br>день</th>"
hdr += "<th>Ср. продаж/<br>день</th>"
hdr += "<th>% выкупа</th>"
hdr += "<th>Остаток</th>"
hdr += "<th>Дней<br>запаса</th>"
hdr += "<th>Статус</th>"
hdr += "<th>К поставке<br>(шт)</th>"
hdr += "<th>Сумма<br>поставки</th>"
hdr += "</tr>"

# ── Data rows ─────────────────────────────────────────────────

rows = ""
for idx, (_, row) in enumerate(display.iterrows(), start=start_idx + 1):
    art = row.get("supplier_article", "")
    subj = row.get("subject", "")
    avg_ord = float(row.get("avg_orders_day", 0))
    avg_sal = float(row.get("avg_sales_day", 0))
    buyout = float(row.get("buyout_pct", 0))
    stock = int(row.get("current_stock", 0))
    days = row.get("days_of_stock", 0)
    days_val = int(days) if not pd.isna(days) and days > 0 else 0
    status = row.get("status", "")
    need = int(row.get("need_to_supply", 0))
    cost = float(row.get("supply_cost", 0))

    days_clr = _days_color(days)
    need_style = 'font-weight:700' if need > 0 else ''

    tr = "<tr>"
    tr += f'<td class="rn">{idx}</td>'
    tr += f'<td style="font-weight:600;font-size:11px">{art}</td>'
    tr += f'<td style="font-size:11px;color:#64748b">{subj}</td>'
    tr += f'<td class="num">{fmt_number(avg_ord, 1)}</td>'
    tr += f'<td class="num">{fmt_number(avg_sal, 1)}</td>'
    tr += f'<td class="ctr">{fmt_pct_tbl(buyout)}</td>'
    tr += f'<td class="num">{fmt_number(stock)}</td>'
    tr += f'<td class="ctr" style="color:{days_clr};font-weight:700">{days_val if days_val > 0 else "0"}</td>'
    tr += f'<td class="ctr">{_status_badge(status)}</td>'
    tr += f'<td class="num" style="{need_style}">{fmt_number(need)}</td>'
    tr += f'<td class="num">{fmt_number(cost)}</td>'
    tr += "</tr>"
    rows += tr

# ── Totals row ────────────────────────────────────────────────

tot_stock = int(display["current_stock"].sum())
tot_need = int(display["need_to_supply"].sum())
tot_cost = display["supply_cost"].sum()

ftr = "<tr>"
ftr += '<td></td><td><b>Итого</b></td><td></td>'
ftr += '<td></td><td></td><td></td>'
ftr += f'<td class="num">{fmt_number(tot_stock)}</td>'
ftr += '<td></td><td></td>'
ftr += f'<td class="num" style="font-weight:700">{fmt_number(tot_need)}</td>'
ftr += f'<td class="num" style="font-weight:700">{fmt_number(tot_cost)}</td>'
ftr += "</tr>"

html = (
    f'{TABLE_CSS}<div class="art-wrap"><table class="art-t" data-sortable>'
    f'<thead>{hdr}</thead><tbody>{rows}</tbody>'
    f'<tfoot>{ftr}</tfoot></table></div>{SORT_JS}'
)

render_table(html)

st.caption(f"Показано {start_idx + 1}\u2013{end_idx} из {total_rows}")

# ── Export ────────────────────────────────────────────────────

export_buttons(df, "supply_needs", sheet_name="Needs")
