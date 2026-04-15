"""Юнит-экономика — пошаговый P&L на один проданный товар (CM1/CM2/CM3).

Содержит три режима:
1. **По данным из БД** — автоматический расчёт по финансовым отчётам WB.
2. **Ручной расчёт** — единичный юнит, все параметры вводятся вручную.
3. **Импорт из Excel** — шаблон + загрузка файла с параметрами массива
   артикулов + выгрузка рассчитанной матрицы.
"""
import io
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from marts import fetch_dataframe, FIN_PROFIT_QUERY
from styles import (
    inject_global_styles, fmt_number, fmt_pct_tbl, format_currency,
    date_filter_bar, PLOTLY_LAYOUT, PLOTLY_COLORS, wb_link,
    export_buttons, paginate, plotly_defaults, render_sortable_table,
)
from auth import check_auth, logout

inject_global_styles()

if not check_auth():
    st.stop()
logout()
st.title("🧮 Юнит-экономика")
st.caption("CM1 → CM2 → CM3: разложение одного проданного юнита на составляющие")

with st.expander("ℹ️ Что такое CM1 / CM2 / CM3", expanded=False):
    st.markdown(
        """
        **Contribution Margin** — маржинальная прибыль на один проданный товар,
        с пошаговым вычитанием категорий расходов:

        - **CM1** = Выручка за ед. − Комиссия WB
          *(маржа после комиссии площадки)*
        - **CM2** = CM1 − Логистика − Хранение − Штрафы
          *(маржа после услуг WB — что осталось после логистики)*
        - **CM3** = CM2 − Себестоимость
          *(чистая товарная маржа — до налогов)*

        **Как читать:**
        - **CM3 < 0** → убыточный юнит, разобраться со стоимостью/ценой
        - **CM1 < 0** → продаём ниже цены плюс комиссия (часто — акции, демпинг)
        - **CM2 ≈ CM1** → очень эффективная логистика (крупные лёгкие товары)

        Расчёт делается на *единицу* — делим суммы на число продаж.

        **Три режима вкладок:**
        - *По данным из БД* — реальные цифры из финансовых отчётов WB
        - *Ручной расчёт* — ввод параметров одного юнита вручную (what-if)
        - *Импорт из Excel* — массовый расчёт по шаблону
        """
    )

_tab_db, _tab_manual, _tab_excel = st.tabs(
    ["📊 По данным из БД", "✏️ Ручной расчёт", "📥 Импорт / шаблон Excel"]
)

# ═════════════════════════════════════════════════════════════
# TAB 1: DB-driven (existing logic)
# ═════════════════════════════════════════════════════════════
with _tab_db:
    d_from, d_to = date_filter_bar("unit", default_days=30)
    params = {"d_from": str(d_from), "d_to": str(d_to)}

    df = fetch_dataframe(FIN_PROFIT_QUERY, params)
    if df.empty:
        st.info("Нет данных за выбранный период")
    else:
        # Ensure numeric
        for col in [
            "sales_count", "ppvz_for_pay", "commission_amount", "logistics_amount",
            "storage_amount", "penalty_amount", "cost_amount", "profit",
            "acceptance_amount", "acquiring_amount", "deduction_amount",
            "additional_payment_amount", "tax_amount",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Aggregate per article
        agg = df.groupby(["nm_id", "supplier_article"]).agg(
            subject=("subject", "first"),
            brand=("brand", "first"),
            units=("sales_count", "sum"),
            revenue=("ppvz_for_pay", "sum"),
            commission=("commission_amount", "sum"),
            logistics=("logistics_amount", "sum"),
            storage=("storage_amount", "sum"),
            penalty=("penalty_amount", "sum"),
            cost=("cost_amount", "sum"),
            tax=("tax_amount", "sum"),
            profit=("profit", "sum"),
        ).reset_index()

        # Only keep articles with sales
        agg = agg[agg["units"] > 0].copy()
        if agg.empty:
            st.info("Нет артикулов с продажами в выбранный период")
        else:
            # Per-unit metrics
            for metric in ("revenue", "commission", "logistics", "storage", "penalty", "cost", "tax", "profit"):
                agg[f"{metric}_per_unit"] = agg[metric] / agg["units"]

            # Contribution margins
            agg["cm1"] = agg["revenue_per_unit"] - agg["commission_per_unit"]
            agg["cm2"] = agg["cm1"] - agg["logistics_per_unit"] - agg["storage_per_unit"] - agg["penalty_per_unit"]
            agg["cm3"] = agg["cm2"] - agg["cost_per_unit"]
            agg["cm1_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm1"] / agg["revenue_per_unit"] * 100, 0)
            agg["cm2_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm2"] / agg["revenue_per_unit"] * 100, 0)
            agg["cm3_pct"] = np.where(agg["revenue_per_unit"] > 0, agg["cm3"] / agg["revenue_per_unit"] * 100, 0)

            # Aggregate totals for waterfall
            total_units = float(agg["units"].sum())
            avg_rev = agg["revenue"].sum() / total_units
            avg_comm = agg["commission"].sum() / total_units
            avg_logi = agg["logistics"].sum() / total_units
            avg_stor = agg["storage"].sum() / total_units
            avg_pen = agg["penalty"].sum() / total_units
            avg_cost = agg["cost"].sum() / total_units
            avg_tax = agg["tax"].sum() / total_units
            avg_profit = agg["profit"].sum() / total_units
            avg_cm1 = avg_rev - avg_comm
            avg_cm2 = avg_cm1 - avg_logi - avg_stor - avg_pen
            avg_cm3 = avg_cm2 - avg_cost

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Юнитов", f"{int(total_units):,}".replace(",", " "))
            c2.metric("Средний чек", format_currency(avg_rev))
            c3.metric("CM1 / ед.", format_currency(avg_cm1), f"{avg_cm1 / avg_rev * 100:.1f}%" if avg_rev else "–")
            c4.metric("CM2 / ед.", format_currency(avg_cm2), f"{avg_cm2 / avg_rev * 100:.1f}%" if avg_rev else "–")
            c5.metric("Прибыль / ед.", format_currency(avg_profit), f"{avg_profit / avg_rev * 100:.1f}%" if avg_rev else "–")

            # Waterfall: unit-level decomposition
            st.markdown("### Юнит-экономика (средний юнит)")
            _labels = [
                "Средний чек", "− Комиссия WB", "= CM1",
                "− Логистика", "− Хранение", "− Штрафы", "= CM2",
                "− Себестоимость", "= CM3", "− Налог", "= Прибыль",
            ]
            _vals = [
                avg_rev, -avg_comm, 0,
                -avg_logi, -avg_stor, -avg_pen, 0,
                -avg_cost, 0,
                -avg_tax, 0,
            ]
            _measure = [
                "absolute", "relative", "total",
                "relative", "relative", "relative", "total",
                "relative", "total",
                "relative", "total",
            ]
            _text = [fmt_number(abs(v)) if v else "" for v in _vals]
            _vals[2] = avg_cm1
            _vals[6] = avg_cm2
            _vals[8] = avg_cm3
            _vals[10] = avg_profit
            _text[2] = fmt_number(avg_cm1)
            _text[6] = fmt_number(avg_cm2)
            _text[8] = fmt_number(avg_cm3)
            _text[10] = fmt_number(avg_profit)

            fig = go.Figure(go.Waterfall(
                x=_labels,
                y=_vals,
                measure=_measure,
                connector=dict(line=dict(color="#cbd5e1", width=1, dash="dash")),
                increasing_marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color="white", width=1.5)),
                decreasing_marker=dict(color=PLOTLY_COLORS["rose"], line=dict(color="white", width=1.5)),
                totals_marker=dict(color=PLOTLY_COLORS["blue_dark"], line=dict(color="white", width=1.5)),
                text=_text,
                textposition="outside",
                textfont=dict(size=11, color="#334155", family="Inter, system-ui, sans-serif"),
                hovertemplate="<b>%{x}</b><br>%{y:,.2f} ₽<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                yaxis_title="₽ на юнит", showlegend=False,
                margin=dict(l=10, r=10, t=30, b=40),
                bargap=0.25,
            )
            plotly_defaults(fig)
            st.plotly_chart(fig, width="stretch")

            # Per-article table
            st.markdown("### Детализация по артикулам")
            st.caption("CM1 = Выручка − Комиссия · CM2 = CM1 − (Логистика + Хранение + Штрафы) · CM3 = CM2 − Себестоимость")

            # Sort and filter
            _fc1, _fc2 = st.columns(2)
            with _fc1:
                sort_col = st.selectbox(
                    "Сортировать по",
                    ["cm3", "cm2", "cm1", "profit", "units", "revenue_per_unit", "profit_per_unit"],
                    format_func=lambda x: {
                        "cm3": "CM3 (убывание)", "cm2": "CM2", "cm1": "CM1",
                        "profit": "Прибыль", "units": "Юниты", "revenue_per_unit": "Цена за юнит",
                        "profit_per_unit": "Прибыль за юнит",
                    }[x],
                )
            with _fc2:
                only_negative = st.checkbox("Только убыточные (CM3 < 0)")

            shown = agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
            if only_negative:
                shown = shown[shown["cm3"] < 0].reset_index(drop=True)

            if shown.empty:
                st.info("Нет артикулов по выбранному фильтру")
            else:
                display, start, end, total = paginate(shown, "unit_art", default_size=50)

                hdr = (
                    "<tr><th>#</th><th>Артикул</th><th>Предмет</th><th>Юниты</th>"
                    "<th>Цена / ед.</th><th>Комисс. / ед.</th><th>CM1 / ед.</th><th>%</th>"
                    "<th>Лог. / ед.</th><th>Хран. / ед.</th><th>CM2 / ед.</th><th>%</th>"
                    "<th>Себест. / ед.</th><th>CM3 / ед.</th><th>%</th></tr>"
                )
                rows = ""
                for i, (_, r) in enumerate(display.iterrows()):
                    idx = start + i + 1
                    cm1_cls = "pos" if r["cm1"] > 0 else "neg"
                    cm2_cls = "pos" if r["cm2"] > 0 else "neg"
                    cm3_cls = "pos" if r["cm3"] > 0 else "neg"
                    rows += (
                        f'<tr><td class="ctr" style="color:#94a3b8">{idx}</td>'
                        f'<td><b>{wb_link(r["nm_id"], r["supplier_article"])}</b></td>'
                        f'<td>{r["subject"]}</td>'
                        f'<td class="num">{int(r["units"])}</td>'
                        f'<td class="num">{fmt_number(r["revenue_per_unit"])}</td>'
                        f'<td class="num">{fmt_number(r["commission_per_unit"])}</td>'
                        f'<td class="num {cm1_cls}"><b>{fmt_number(r["cm1"])}</b></td>'
                        f'<td class="ctr {cm1_cls}">{fmt_pct_tbl(r["cm1_pct"])}</td>'
                        f'<td class="num">{fmt_number(r["logistics_per_unit"])}</td>'
                        f'<td class="num">{fmt_number(r["storage_per_unit"])}</td>'
                        f'<td class="num {cm2_cls}"><b>{fmt_number(r["cm2"])}</b></td>'
                        f'<td class="ctr {cm2_cls}">{fmt_pct_tbl(r["cm2_pct"])}</td>'
                        f'<td class="num">{fmt_number(r["cost_per_unit"])}</td>'
                        f'<td class="num {cm3_cls}"><b>{fmt_number(r["cm3"])}</b></td>'
                        f'<td class="ctr {cm3_cls}">{fmt_pct_tbl(r["cm3_pct"])}</td>'
                        f'</tr>'
                    )
                render_sortable_table("ue", hdr, rows)

                export_buttons(agg, "unit_economics", sheet_name="UnitEconomics")


# ═════════════════════════════════════════════════════════════
# Helper: unit-economics calculator used by Tabs 2 and 3
# ═════════════════════════════════════════════════════════════
def _calc_unit(*, price, spp_pct, commission_pct, logistics, storage,
               penalty, cost, tax_pct):
    """Calculate CM1/CM2/CM3 for one unit, given user-supplied params.

    price          — цена до СПП, ₽
    spp_pct        — скидка постоянного покупателя, %
    commission_pct — плановая комиссия WB, %
    logistics      — логистика на ед., ₽
    storage        — хранение на ед., ₽
    penalty        — штрафы на ед., ₽
    cost           — закупочная себестоимость на ед., ₽
    tax_pct        — ставка налога (УСН), %
    """
    # Реализация после СПП
    retail_after_spp = price * (1 - spp_pct / 100)
    # К перечислению = после СПП − комиссия (упрощённо)
    commission_amt = retail_after_spp * commission_pct / 100
    cm1 = retail_after_spp - commission_amt
    cm2 = cm1 - logistics - storage - penalty
    cm3 = cm2 - cost
    # Налог берётся от К перечислению (упрощённо для УСН «Доход»)
    tax_amt = retail_after_spp * tax_pct / 100
    profit = cm3 - tax_amt
    return {
        "price": price,
        "retail_after_spp": retail_after_spp,
        "commission_amt": commission_amt,
        "cm1": cm1,
        "cm2": cm2,
        "cm3": cm3,
        "tax_amt": tax_amt,
        "profit": profit,
        "cm1_pct": cm1 / price * 100 if price else 0,
        "cm2_pct": cm2 / price * 100 if price else 0,
        "cm3_pct": cm3 / price * 100 if price else 0,
        "margin_pct": profit / price * 100 if price else 0,
    }


# ═════════════════════════════════════════════════════════════
# TAB 2: Manual single-unit calculator
# ═════════════════════════════════════════════════════════════
with _tab_manual:
    st.markdown("### Единичный юнит — what-if расчёт")
    st.caption(
        "Введите параметры одного товара вручную — удобно для прикидки при запуске "
        "новой карточки или при смене условий."
    )

    _fc1, _fc2, _fc3 = st.columns(3)
    with _fc1:
        m_price = st.number_input(
            "Цена до СПП, ₽", min_value=0.0, value=1000.0, step=10.0, key="ue_m_price"
        )
        m_spp = st.number_input(
            "СПП, %", min_value=0.0, max_value=100.0, value=20.0, step=0.5, key="ue_m_spp"
        )
        m_commission = st.number_input(
            "Комиссия WB, %", min_value=0.0, max_value=100.0, value=15.0, step=0.5, key="ue_m_comm"
        )
    with _fc2:
        m_logistics = st.number_input(
            "Логистика на ед., ₽", min_value=0.0, value=50.0, step=5.0, key="ue_m_log"
        )
        m_storage = st.number_input(
            "Хранение на ед., ₽", min_value=0.0, value=5.0, step=1.0, key="ue_m_stor"
        )
        m_penalty = st.number_input(
            "Штрафы на ед., ₽", min_value=0.0, value=0.0, step=1.0, key="ue_m_pen"
        )
    with _fc3:
        m_cost = st.number_input(
            "Себестоимость, ₽", min_value=0.0, value=300.0, step=10.0, key="ue_m_cost"
        )
        m_tax = st.number_input(
            "Налог (УСН), %", min_value=0.0, max_value=100.0, value=7.0, step=0.1, key="ue_m_tax"
        )
        m_qty = st.number_input(
            "Прогноз продаж, шт", min_value=0, value=100, step=1, key="ue_m_qty"
        )

    r = _calc_unit(
        price=m_price, spp_pct=m_spp, commission_pct=m_commission,
        logistics=m_logistics, storage=m_storage, penalty=m_penalty,
        cost=m_cost, tax_pct=m_tax,
    )

    # KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("После СПП", format_currency(r["retail_after_spp"]))
    k2.metric("CM1 / ед.", format_currency(r["cm1"]), f"{r['cm1_pct']:.1f}%")
    k3.metric("CM2 / ед.", format_currency(r["cm2"]), f"{r['cm2_pct']:.1f}%")
    k4.metric("CM3 / ед.", format_currency(r["cm3"]), f"{r['cm3_pct']:.1f}%")
    k5.metric("Прибыль / ед.", format_currency(r["profit"]), f"{r['margin_pct']:.1f}%")

    # Verdict
    if r["profit"] > 0 and r["margin_pct"] >= 15:
        st.success(f"✅ Хорошая экономика. Прибыль/ед. {format_currency(r['profit'])}, маржа {r['margin_pct']:.1f}%.")
    elif r["profit"] > 0:
        st.warning(f"⚠️ Прибыль есть, но маржа низкая — {r['margin_pct']:.1f}%. Имеет смысл проверить логистику/комиссию.")
    elif r["cm3"] > 0:
        st.warning(f"⚠️ CM3 положительная, но налог съедает прибыль. Рассмотрите смену режима налогообложения.")
    elif r["cm1"] > 0:
        st.error(f"❌ Логистика/хранение съедают маржу. CM2 = {format_currency(r['cm2'])}, CM3 = {format_currency(r['cm3'])}.")
    else:
        st.error(f"❌ Продаём ниже цены + комиссии. CM1 = {format_currency(r['cm1'])}.")

    # Waterfall
    st.markdown("#### Разложение (waterfall)")
    _lbl = [
        "Цена до СПП", "− СПП", "= После СПП", "− Комиссия WB", "= CM1",
        "− Логистика", "− Хранение", "− Штрафы", "= CM2",
        "− Себестоимость", "= CM3", "− Налог", "= Прибыль / ед.",
    ]
    _v = [
        r["price"], r["retail_after_spp"] - r["price"], 0,
        -r["commission_amt"], 0,
        -m_logistics, -m_storage, -m_penalty, 0,
        -m_cost, 0,
        -r["tax_amt"], 0,
    ]
    _m = [
        "absolute", "relative", "total", "relative", "total",
        "relative", "relative", "relative", "total",
        "relative", "total",
        "relative", "total",
    ]
    _t = [fmt_number(abs(v)) if v else "" for v in _v]
    _v[2] = r["retail_after_spp"]; _t[2] = fmt_number(r["retail_after_spp"])
    _v[4] = r["cm1"]; _t[4] = fmt_number(r["cm1"])
    _v[8] = r["cm2"]; _t[8] = fmt_number(r["cm2"])
    _v[10] = r["cm3"]; _t[10] = fmt_number(r["cm3"])
    _v[12] = r["profit"]; _t[12] = fmt_number(r["profit"])

    fig = go.Figure(go.Waterfall(
        x=_lbl, y=_v, measure=_m,
        connector=dict(line=dict(color="#cbd5e1", width=1, dash="dash")),
        increasing_marker=dict(color=PLOTLY_COLORS["blue"], line=dict(color="white", width=1.5)),
        decreasing_marker=dict(color=PLOTLY_COLORS["rose"], line=dict(color="white", width=1.5)),
        totals_marker=dict(color=PLOTLY_COLORS["blue_dark"], line=dict(color="white", width=1.5)),
        text=_t, textposition="outside",
        textfont=dict(size=11, color="#334155", family="Inter, system-ui, sans-serif"),
        hovertemplate="<b>%{x}</b><br>%{y:,.2f} ₽<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis_title="₽ на юнит", showlegend=False,
        margin=dict(l=10, r=10, t=30, b=40),
        bargap=0.25,
    )
    plotly_defaults(fig)
    st.plotly_chart(fig, width="stretch")

    # At-volume summary
    if m_qty > 0:
        st.markdown("#### При прогнозе продаж")
        k1, k2, k3 = st.columns(3)
        k1.metric("Выручка до СПП", format_currency(m_price * m_qty))
        k2.metric("К перечислению", format_currency(r["cm1"] * m_qty))
        k3.metric("Прибыль", format_currency(r["profit"] * m_qty))


# ═════════════════════════════════════════════════════════════
# TAB 3: Excel import / template
# ═════════════════════════════════════════════════════════════
with _tab_excel:
    st.markdown("### Массовый расчёт по шаблону")
    st.caption(
        "Скачайте шаблон, заполните ваши артикулы и параметры, загрузите обратно — "
        "получите рассчитанную CM1/CM2/CM3-матрицу с возможностью скачать в Excel."
    )

    # ── Template download
    _tpl_rows = [
        {
            "supplier_article": "TEST-001",
            "subject": "Футболка",
            "price_before_spp": 1000,
            "spp_pct": 20,
            "commission_pct": 15,
            "logistics_per_unit": 50,
            "storage_per_unit": 5,
            "penalty_per_unit": 0,
            "cost_per_unit": 300,
            "tax_pct": 7,
            "qty_forecast": 100,
        },
        {
            "supplier_article": "TEST-002",
            "subject": "Носки",
            "price_before_spp": 300,
            "spp_pct": 10,
            "commission_pct": 15,
            "logistics_per_unit": 20,
            "storage_per_unit": 2,
            "penalty_per_unit": 0,
            "cost_per_unit": 80,
            "tax_pct": 7,
            "qty_forecast": 500,
        },
    ]
    tpl_df = pd.DataFrame(_tpl_rows)
    tpl_buf = io.BytesIO()
    with pd.ExcelWriter(tpl_buf, engine="openpyxl") as w:
        tpl_df.to_excel(w, index=False, sheet_name="UnitEconomics")
    st.download_button(
        "📄 Скачать шаблон (Excel)",
        tpl_buf.getvalue(),
        "unit_economics_template.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_ue_template",
    )

    with st.expander("📋 Описание колонок шаблона", expanded=False):
        st.markdown(
            """
            | Колонка | Описание |
            |---|---|
            | `supplier_article` | Артикул продавца (обязательно) |
            | `subject` | Название предмета (опционально) |
            | `price_before_spp` | Цена до СПП, ₽ (обязательно) |
            | `spp_pct` | Скидка постоянного покупателя, % (0–100) |
            | `commission_pct` | Плановая комиссия WB, % (обычно 15–25) |
            | `logistics_per_unit` | Логистика на единицу, ₽ |
            | `storage_per_unit` | Хранение на единицу, ₽ |
            | `penalty_per_unit` | Штрафы на единицу, ₽ (обычно 0) |
            | `cost_per_unit` | Закупочная стоимость единицы, ₽ |
            | `tax_pct` | УСН-ставка, % (например 7) |
            | `qty_forecast` | Прогноз продаж, шт (для расчёта итогов) |
            """
        )

    st.markdown("---")
    up = st.file_uploader(
        "Загрузите заполненный шаблон (.xlsx)",
        type=["xlsx"], key="ue_upload",
    )

    if up is not None:
        try:
            in_df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {type(e).__name__}: {e}")
            st.stop()

        # Validate required columns
        _required = ["supplier_article", "price_before_spp"]
        _missing = [c for c in _required if c not in in_df.columns]
        if _missing:
            st.error(f"В файле нет обязательных колонок: {', '.join(_missing)}")
        else:
            # Fill defaults for optional columns
            _defaults = {
                "subject": "",
                "spp_pct": 0.0,
                "commission_pct": 15.0,
                "logistics_per_unit": 0.0,
                "storage_per_unit": 0.0,
                "penalty_per_unit": 0.0,
                "cost_per_unit": 0.0,
                "tax_pct": 0.0,
                "qty_forecast": 0,
            }
            for col, default in _defaults.items():
                if col not in in_df.columns:
                    in_df[col] = default
                else:
                    in_df[col] = pd.to_numeric(in_df[col], errors="coerce").fillna(default) \
                        if col != "subject" else in_df[col].fillna("")

            in_df["price_before_spp"] = pd.to_numeric(in_df["price_before_spp"], errors="coerce").fillna(0.0)
            in_df = in_df[in_df["price_before_spp"] > 0].reset_index(drop=True)

            if in_df.empty:
                st.warning("Нет строк с положительной ценой.")
            else:
                # Calculate for each row
                calc_rows = []
                for _, rr in in_df.iterrows():
                    c = _calc_unit(
                        price=float(rr["price_before_spp"]),
                        spp_pct=float(rr["spp_pct"]),
                        commission_pct=float(rr["commission_pct"]),
                        logistics=float(rr["logistics_per_unit"]),
                        storage=float(rr["storage_per_unit"]),
                        penalty=float(rr["penalty_per_unit"]),
                        cost=float(rr["cost_per_unit"]),
                        tax_pct=float(rr["tax_pct"]),
                    )
                    qty = float(rr.get("qty_forecast", 0) or 0)
                    calc_rows.append({
                        "supplier_article": rr["supplier_article"],
                        "subject": rr.get("subject", ""),
                        "price_before_spp": c["price"],
                        "retail_after_spp": c["retail_after_spp"],
                        "commission": c["commission_amt"],
                        "logistics": float(rr["logistics_per_unit"]),
                        "storage": float(rr["storage_per_unit"]),
                        "penalty": float(rr["penalty_per_unit"]),
                        "cost": float(rr["cost_per_unit"]),
                        "tax": c["tax_amt"],
                        "cm1": c["cm1"],
                        "cm2": c["cm2"],
                        "cm3": c["cm3"],
                        "profit_per_unit": c["profit"],
                        "margin_pct": c["margin_pct"],
                        "qty_forecast": qty,
                        "total_revenue": c["price"] * qty,
                        "total_profit": c["profit"] * qty,
                    })
                out_df = pd.DataFrame(calc_rows)

                # Summary KPIs for import
                _tot_units = out_df["qty_forecast"].sum()
                _tot_rev = out_df["total_revenue"].sum()
                _tot_profit = out_df["total_profit"].sum()
                _neg_cm3 = int((out_df["cm3"] < 0).sum())

                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Строк", len(out_df))
                k2.metric("Юнитов (прогноз)", f"{int(_tot_units):,}".replace(",", " "))
                k3.metric("Выручка", format_currency(_tot_rev))
                k4.metric("Прибыль", format_currency(_tot_profit))
                k5.metric("CM3 < 0", _neg_cm3, help="Число убыточных артикулов")

                # Render table
                st.markdown("#### Результаты")
                hdr2 = (
                    "<tr><th>#</th><th>Артикул</th><th>Предмет</th>"
                    "<th>Цена до СПП</th><th>После СПП</th>"
                    "<th>CM1</th><th>CM2</th><th>CM3</th>"
                    "<th>Прибыль / ед.</th><th>Маржа, %</th>"
                    "<th>Прогноз шт</th><th>Итог. прибыль</th></tr>"
                )
                rows2 = ""
                for i, (_, rr) in enumerate(out_df.iterrows(), 1):
                    cm1_cls = "pos" if rr["cm1"] > 0 else "neg"
                    cm2_cls = "pos" if rr["cm2"] > 0 else "neg"
                    cm3_cls = "pos" if rr["cm3"] > 0 else "neg"
                    p_cls = "pos" if rr["profit_per_unit"] > 0 else "neg"
                    m_cls = "pos" if rr["margin_pct"] > 0 else "neg"
                    tp_cls = "pos" if rr["total_profit"] > 0 else "neg"
                    rows2 += (
                        f'<tr><td class="ctr" style="color:#94a3b8">{i}</td>'
                        f'<td><b>{rr["supplier_article"]}</b></td>'
                        f'<td>{rr.get("subject","")}</td>'
                        f'<td class="num">{fmt_number(rr["price_before_spp"])}</td>'
                        f'<td class="num">{fmt_number(rr["retail_after_spp"])}</td>'
                        f'<td class="num {cm1_cls}">{fmt_number(rr["cm1"])}</td>'
                        f'<td class="num {cm2_cls}">{fmt_number(rr["cm2"])}</td>'
                        f'<td class="num {cm3_cls}"><b>{fmt_number(rr["cm3"])}</b></td>'
                        f'<td class="num {p_cls}">{fmt_number(rr["profit_per_unit"])}</td>'
                        f'<td class="ctr {m_cls}">{fmt_pct_tbl(rr["margin_pct"])}</td>'
                        f'<td class="num">{fmt_number(rr["qty_forecast"])}</td>'
                        f'<td class="num {tp_cls}"><b>{fmt_number(rr["total_profit"])}</b></td>'
                        f'</tr>'
                    )
                render_sortable_table("uex", hdr2, rows2)

                export_buttons(out_df, "unit_economics_calc", key="ue_import", sheet_name="UnitCalc")

    else:
        st.info("Загрузите заполненный шаблон Excel, чтобы увидеть массовый расчёт.")
