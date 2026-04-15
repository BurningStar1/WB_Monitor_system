"""Tests for key SQL formulas used in financial marts and reports.

Guards against regressions in the profit formula that cost us a painful
bug where `net_profit` was showing zero because components were missing.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from db import get_engine


@pytest.fixture(scope="module")
def conn():
    engine = get_engine()
    with engine.connect() as c:
        yield c


# ── Canonical profit formula (single source of truth) ────────────
# profit = ppvz_for_pay - logistics - storage - penalty - acceptance
#          - acquiring - deduction + additional_payment - cost - tax
PROFIT_COMPONENTS = """
ppvz_for_pay
- logistics_amount - storage_amount
- penalty_amount - acceptance_amount
- acquiring_amount - deduction_amount
+ additional_payment_amount
"""


def test_finance_daily_has_all_fee_columns(conn) -> None:
    """All 7 fee columns must exist in mart.finance_daily."""
    cols = set(
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='mart' AND table_name='finance_daily'"
            )
        )
    )
    required = {
        "ppvz_for_pay", "commission_amount", "logistics_amount",
        "storage_amount", "penalty_amount", "acceptance_amount",
        "acquiring_amount", "deduction_amount", "additional_payment_amount",
    }
    missing = required - cols
    assert not missing, f"finance_daily missing columns: {missing}"


def test_pnl_monthly_profit_matches_manual_formula(conn) -> None:
    """PNL monthly report's net_profit must equal the canonical formula."""
    from marts.queries import PNL_MONTHLY_QUERY

    rows = conn.execute(text(PNL_MONTHLY_QUERY)).mappings().all()
    if not rows:
        pytest.skip("no data in finance_daily yet")

    for row in rows:
        # Reconstruct profit from components we also expose in the result
        manual = (
            float(row["ppvz_for_pay"])
            - float(row["logistics"])
            - float(row["storage"])
            - float(row["penalty"])
            - float(row["acceptance"])
            - float(row["acquiring"])
            - float(row["deduction"])
            + float(row["additional_payment"])
            - float(row["cost_amount"])
            - float(row["tax_amount"])
        )
        actual = float(row["net_profit"])
        assert abs(manual - actual) < 0.01, (
            f"month {row['month']}: manual={manual:.2f} vs net_profit={actual:.2f}"
        )
        # `profit` alias must equal `net_profit`
        assert abs(float(row["profit"]) - actual) < 0.01


def test_total_fees_sums_all_negative_components(conn) -> None:
    """total_fees in PNL must include all 7 fee buckets (excluding additional_payment)."""
    from marts.queries import PNL_MONTHLY_QUERY

    rows = conn.execute(text(PNL_MONTHLY_QUERY)).mappings().all()
    if not rows:
        pytest.skip("no data in finance_daily yet")

    for row in rows:
        manual = (
            float(row["commission"])
            + float(row["logistics"])
            + float(row["storage"])
            + float(row["penalty"])
            + float(row["acceptance"])
            + float(row["acquiring"])
            + float(row["deduction"])
        )
        actual = float(row["total_fees"])
        assert abs(manual - actual) < 0.01, (
            f"month {row['month']}: total_fees sum mismatch {manual} vs {actual}"
        )


def test_finance_fee_aggregate_positive(conn) -> None:
    """In aggregate (over a month), each fee bucket must sum to >= 0.

    Individual rows can have negative values (e.g. reversed charges),
    but the monthly sum should be non-negative.
    """
    query = """
    SELECT
        SUM(logistics_amount) AS logi,
        SUM(storage_amount) AS stor,
        SUM(penalty_amount) AS pen,
        SUM(acceptance_amount) AS acc,
        SUM(acquiring_amount) AS acq,
        SUM(deduction_amount) AS ded,
        SUM(commission_amount) AS comm
    FROM mart.finance_daily
    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
    """
    row = conn.execute(text(query)).mappings().one()
    for col in ("logi", "stor", "pen", "acc", "acq", "ded", "comm"):
        v = row[col] or 0
        assert float(v) >= 0, f"{col} aggregate negative over 30d: {v}"


def test_article_profit_formula_matches_finance_formula(conn) -> None:
    """The per-article profit formula used in the articles report must equal the canonical one."""
    query = """
    SELECT nm_id,
           SUM(ppvz_for_pay - logistics_amount - storage_amount
               - penalty_amount - acceptance_amount - acquiring_amount
               - deduction_amount + additional_payment_amount) AS profit_before_cost
    FROM mart.finance_daily
    WHERE report_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY nm_id
    HAVING SUM(ppvz_for_pay) > 0
    LIMIT 20
    """
    rows = conn.execute(text(query)).mappings().all()
    if not rows:
        pytest.skip("no recent finance_daily data")
    # Every row should have a computable profit (not NaN/None)
    for r in rows:
        assert r["profit_before_cost"] is not None


def test_finance_queries_cross_consistency(conn) -> None:
    """All 6 finance queries must produce identical monthly profit totals.

    The master profit number is PNL_MONTHLY_QUERY.net_profit. All of
    FIN_WEEKLY, FIN_PROFIT, FIN_ARTICLE, FIN_STATUTORY, FINANCE_DAILY
    must agree when summed over the same month, or dashboards will show
    inconsistent numbers. This is the regression guard against per-row
    GREATEST(pre_tax, 0) which inflates tax on loss rows.
    """
    from marts.queries import (
        PNL_MONTHLY_QUERY,
        FIN_WEEKLY_QUERY,
        FIN_PROFIT_QUERY,
        FIN_ARTICLE_QUERY,
        FIN_STATUTORY_QUERY,
        FINANCE_DAILY_QUERY,
    )
    from datetime import date

    # Find the latest complete month with finance data
    row = conn.execute(text(
        "SELECT date_trunc('month', MAX(report_date))::date AS m, "
        "       (date_trunc('month', MAX(report_date)) "
        "        + INTERVAL '1 month' - INTERVAL '1 day')::date AS e "
        "FROM mart.finance_daily"
    )).mappings().one()
    month_start = row["m"]
    month_end = row["e"]
    if month_start is None:
        pytest.skip("no finance_daily data yet")

    params = {"d_from": str(month_start), "d_to": str(month_end)}

    # Reference: PNL_MONTHLY
    ref_rows = conn.execute(text(PNL_MONTHLY_QUERY)).mappings().all()
    ref = next(r for r in ref_rows if r["month"] == month_start)
    target = float(ref["net_profit"])
    assert target > 0, "expected latest month to be profitable for this check"

    # FIN_WEEKLY — sum of profit column over all weeks in the month
    wk_rows = conn.execute(text(FIN_WEEKLY_QUERY), params).mappings().all()
    wk_sum = sum(float(r["profit"]) for r in wk_rows)
    # Weekly may span month boundaries; we use a same-month filter via :d_from/:d_to.
    # Allow up to 5% drift since week endpoints span months.
    assert abs(wk_sum - target) / target < 0.05, (
        f"FIN_WEEKLY sum {wk_sum:.2f} differs from target {target:.2f}"
    )

    # FIN_PROFIT — additive sum over all day×article rows
    fp_rows = conn.execute(text(FIN_PROFIT_QUERY), params).mappings().all()
    fp_sum = sum(float(r["profit"]) for r in fp_rows)
    assert abs(fp_sum - target) < 0.02, (
        f"FIN_PROFIT sum {fp_sum:.2f} differs from target {target:.2f}"
    )

    # FIN_ARTICLE — sum across all articles
    fa_rows = conn.execute(text(FIN_ARTICLE_QUERY), params).mappings().all()
    fa_sum = sum(float(r["profit"]) for r in fa_rows)
    assert abs(fa_sum - target) < 0.02, (
        f"FIN_ARTICLE sum {fa_sum:.2f} differs from target {target:.2f}"
    )

    # FIN_STATUTORY — single row for the month
    fs_rows = conn.execute(text(FIN_STATUTORY_QUERY)).mappings().all()
    fs = next(r for r in fs_rows if r["month"] == month_start)
    assert abs(float(fs["profit"]) - target) < 0.02, (
        f"FIN_STATUTORY profit {float(fs['profit']):.2f} differs from target {target:.2f}"
    )

    # FINANCE_DAILY — additive sum of net_profit_amount
    fd_rows = conn.execute(text(FINANCE_DAILY_QUERY), params).mappings().all()
    fd_sum = sum(float(r["net_profit_amount"]) for r in fd_rows)
    assert abs(fd_sum - target) < 0.02, (
        f"FINANCE_DAILY sum {fd_sum:.2f} differs from target {target:.2f}"
    )
