from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_xlsx(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path)
