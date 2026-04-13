from __future__ import annotations

import pandas as pd


def assert_non_negative(df: pd.DataFrame, columns: list[str]) -> list[str]:
    issues = []
    for col in columns:
        if col in df.columns and (df[col].fillna(0) < 0).any():
            issues.append(f"Column {col} has negative values")
    return issues


def assert_not_null(df: pd.DataFrame, columns: list[str]) -> list[str]:
    issues = []
    for col in columns:
        if col in df.columns and df[col].isna().any():
            issues.append(f"Column {col} has null values")
    return issues
