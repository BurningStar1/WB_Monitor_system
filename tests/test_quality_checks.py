import pandas as pd

from checks import assert_non_negative, assert_not_null


def test_non_negative() -> None:
    df = pd.DataFrame({"a": [1, 2, 0]})
    assert assert_non_negative(df, ["a"]) == []


def test_not_null_detects_issue() -> None:
    df = pd.DataFrame({"a": [1, None]})
    issues = assert_not_null(df, ["a"])
    assert len(issues) == 1
