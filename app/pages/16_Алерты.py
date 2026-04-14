"""Алерты — сводка критичных событий, требующих внимания.

Три блока:
  1. Низкий остаток (артикулы, у которых закончится товар меньше чем через X дней)
  2. Убыточные артикулы (прибыль < 0 за выбранный период)
  3. Падение заказов неделя-к-неделе (WoW-падение > порога)
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from marts import (
    fetch_dataframe,
    FIN_PROFIT_QUERY,
    ORDERS_DAILY_AMOUNT_QUERY,
    STOCKS_QUERY,
)
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, wb_link,
    export_buttons, paginate, render_sortable_table,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🚨 Алерты")
st.caption("Критичные события, которые нужно решить в первую очередь")

# ── Пороги ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    days_of_supply_thr = st.number_input("Порог запаса (дней)", min_value=1, max_value=60, value=7, step=1)
with c2:
    drop_thr = st.number_input("Порог падения заказов (%)", min_value=5, max_value=90, value=30, step=5)
with c3:
    loss_thr = st.number_input("Порог убытка (₽)", min_value=0, max_value=1_000_000, value=0, step=100)
with c4:
    period_days = st.number_input("Окно анализа (дней)", min_value=7, max_value=90, value=28, step=7)

d_to = date.today()
d_from = d_to - timedelta(days=int(period_days) - 1)
params = {"d_from": str(d_from), "d_to": str(d_to)}

# ── Данные ──────────────────────────────────────────────────
fin_df = fetch_dataframe(FIN_PROFIT_QUERY, params)
ord_df = fetch_dataframe(ORDERS_DAILY_AMOUNT_QUERY, params)
stk_df = fetch_dataframe(STOCKS_QUERY, {})

# ── Блок 1. Низкий запас ────────────────────────────────────
st.markdown("### 📉 Низкий запас")
if ord_df.empty or stk_df.empty:
    st.info("Нет данных по заказам или остаткам.")
    low_stock = pd.DataFrame()
else:
    # Средние дневные заказы по nm_id
    rate = (
        ord_df.groupby("nm_id")["orders_count"].sum()
        .reset_index(name="orders_total")
    )
    rate["avg_daily"] = rate["orders_total"] / float(period_days)
    stk_agg = stk_df.groupby("nm_id").agg(
        stock=("quantity_full", "sum"),
        supplier_article=("supplier_article", "first") if "supplier_article" in stk_df.columns else ("nm_id", "count"),
    ).reset_index()
    merged = stk_agg.merge(rate, on="nm_id", how="left").fillna(0)
    merged["days_of_supply"] = np.where(
        merged["avg_daily"] > 0, merged["stock"] / merged["avg_daily"], np.inf
    )
    # subject/brand из ord_df (если есть)
    if "subject" in ord_df.columns:
        meta = (
            ord_df.groupby("nm_id").agg(
                subject=("subject", "first"),
                brand=("brand", "first") if "brand" in ord_df.columns else ("nm_id", "first"),
            ).reset_index()
        )
        merged = merged.merge(meta, on="nm_id", how="left")
    low_stock = merged[
        (merged["avg_daily"] > 0)
        & (merged["days_of_supply"] < int(days_of_supply_thr))
        & (merged["stock"] >= 0)
    ].sort_values("days_of_supply").reset_index(drop=True)

    if low_stock.empty:
        st.success("Нет артикулов с критично низким запасом.")
    else:
        st.warning(f"Артикулов с запасом меньше {int(days_of_supply_thr)} дней: **{len(low_stock)}**")
        hdr = (
            "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
            "<th>Остаток</th><th>Средн./день</th><th>Запас, дней</th></tr>"
        )
        ls_disp, _ls_s, _ls_e, _ls_t = paginate(low_stock, "ls", default_size=50)
        rows = ""
        for i, r in enumerate(ls_disp.itertuples(), _ls_s + 1):
            sa = getattr(r, "supplier_article", "") or ""
            subj = getattr(r, "subject", "") or ""
            brand = getattr(r, "brand", "") or ""
            dos = r.days_of_supply if np.isfinite(r.days_of_supply) else 999
            dos_cls = "neg" if dos < max(3, int(days_of_supply_thr) / 2) else ""
            rows += (
                f'<tr><td class="ctr" style="color:#94a3b8">{i}</td>'
                f'<td><b>{wb_link(int(r.nm_id), sa)}</b></td>'
                f'<td>{subj}</td><td>{brand}</td>'
                f'<td class="num">{int(r.stock)}</td>'
                f'<td class="num">{r.avg_daily:.1f}</td>'
                f'<td class="num {dos_cls}"><b>{dos:.1f}</b></td></tr>'
            )
        render_sortable_table("ls", hdr, rows, height=420)
        export_buttons(low_stock, "alerts_low_stock", key="low_stock", sheet_name="LowStock")

# ── Блок 2. Убыточные артикулы ──────────────────────────────
st.markdown("### 💸 Убыточные артикулы")
if fin_df.empty:
    st.info("Нет данных по финансам.")
    loss_df = pd.DataFrame()
else:
    loss_df = (
        fin_df.groupby(["nm_id", "supplier_article"])
        .agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            sales_count=("sales_count", "sum"),
            ppvz_for_pay=("ppvz_for_pay", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    loss_df["margin_pct"] = np.where(
        loss_df["ppvz_for_pay"] > 0,
        (loss_df["profit"] / loss_df["ppvz_for_pay"] * 100).round(1),
        0,
    )
    loss_df = loss_df[loss_df["profit"] < -float(loss_thr)].sort_values("profit").reset_index(drop=True)
    if loss_df.empty:
        st.success("Нет убыточных артикулов в выбранный период.")
    else:
        total_loss = float(loss_df["profit"].sum())
        st.warning(
            f"Убыточных артикулов: **{len(loss_df)}**, совокупный убыток: **{fmt_number(total_loss)} ₽**"
        )
        hdr = (
            "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Бренд</th>"
            "<th>Продажи</th><th>Выручка</th><th>Прибыль</th><th>Маржа</th></tr>"
        )
        ld_disp, _ld_s, _ld_e, _ld_t = paginate(loss_df, "ls2", default_size=100)
        rows = ""
        for i, r in enumerate(ld_disp.itertuples(), _ld_s + 1):
            rows += (
                f'<tr><td class="ctr" style="color:#94a3b8">{i}</td>'
                f'<td><b>{wb_link(int(r.nm_id), r.supplier_article or "")}</b></td>'
                f'<td>{r.subject or ""}</td><td>{r.brand or ""}</td>'
                f'<td class="num">{int(r.sales_count)}</td>'
                f'<td class="num">{fmt_number(r.ppvz_for_pay)}</td>'
                f'<td class="num neg">{fmt_number(r.profit)}</td>'
                f'<td class="ctr neg">{fmt_pct_tbl(r.margin_pct)}</td></tr>'
            )
        render_sortable_table("ls2", hdr, rows, height=500)
        export_buttons(loss_df, "alerts_loss", key="loss", sheet_name="Loss")

# ── Блок 3. Падение заказов WoW ─────────────────────────────
st.markdown("### ↘️ Падение заказов неделя-к-неделе")
if ord_df.empty:
    st.info("Нет данных по заказам.")
else:
    ord_df["order_date"] = pd.to_datetime(ord_df["order_date"])
    last_7 = d_to - timedelta(days=6)
    prev_7_start = d_to - timedelta(days=13)
    prev_7_end = d_to - timedelta(days=7)
    cur_mask = ord_df["order_date"] >= pd.Timestamp(last_7)
    prev_mask = (ord_df["order_date"] >= pd.Timestamp(prev_7_start)) & (
        ord_df["order_date"] <= pd.Timestamp(prev_7_end)
    )
    cur = (
        ord_df[cur_mask].groupby(["nm_id", "supplier_article"]).agg(
            orders_cur=("orders_count", "sum"),
            amount_cur=("orders_amount", "sum"),
        ).reset_index()
    )
    prev = (
        ord_df[prev_mask].groupby(["nm_id", "supplier_article"]).agg(
            orders_prev=("orders_count", "sum"),
            amount_prev=("orders_amount", "sum"),
        ).reset_index()
    )
    wow = cur.merge(prev, on=["nm_id", "supplier_article"], how="outer").fillna(0)
    wow["delta_pct"] = np.where(
        wow["orders_prev"] > 0,
        ((wow["orders_cur"] - wow["orders_prev"]) / wow["orders_prev"] * 100).round(1),
        np.where(wow["orders_cur"] > 0, 100.0, 0.0),
    )
    # Falling articles with enough baseline volume (avoid noise on single-unit items)
    wow_fall = wow[
        (wow["orders_prev"] >= 3) & (wow["delta_pct"] <= -float(drop_thr))
    ].sort_values("delta_pct").reset_index(drop=True)
    # Meta join
    if "subject" in ord_df.columns:
        meta2 = (
            ord_df.groupby(["nm_id", "supplier_article"])
            .agg(
                subject=("subject", "first"),
                brand=("brand", "first") if "brand" in ord_df.columns else ("nm_id", "first"),
            )
            .reset_index()
        )
        wow_fall = wow_fall.merge(meta2, on=["nm_id", "supplier_article"], how="left")
    if wow_fall.empty:
        st.success(f"Нет артикулов с падением > {int(drop_thr)}% неделя к неделе.")
    else:
        st.warning(f"Артикулов с падением ≥ {int(drop_thr)}%: **{len(wow_fall)}**")
        hdr = (
            "<tr><th>#</th><th>Артикул</th><th>Предмет</th>"
            "<th>Заказы (тек. 7 дн)</th><th>Заказы (пред. 7 дн)</th><th>Δ%</th></tr>"
        )
        wf_disp, _wf_s, _wf_e, _wf_t = paginate(wow_fall, "wf", default_size=100)
        rows = ""
        for i, r in enumerate(wf_disp.itertuples(), _wf_s + 1):
            subj = getattr(r, "subject", "") or ""
            rows += (
                f'<tr><td class="ctr" style="color:#94a3b8">{i}</td>'
                f'<td><b>{wb_link(int(r.nm_id), r.supplier_article or "")}</b></td>'
                f'<td>{subj}</td>'
                f'<td class="num">{int(r.orders_cur)}</td>'
                f'<td class="num">{int(r.orders_prev)}</td>'
                f'<td class="ctr neg"><b>{r.delta_pct:+.1f}%</b></td></tr>'
            )
        render_sortable_table("wf", hdr, rows, height=500)
        export_buttons(wow_fall, "alerts_wow_drop", key="wow_drop", sheet_name="WoWDrop")
