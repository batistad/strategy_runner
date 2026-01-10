from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol, Sequence

import pandas as pd


class DataProvider(Protocol):
    def fetch_eod(
        self, symbols: Iterable[str], history_days: int, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        ...


@dataclass
class Bar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


_COL_RENAMES = {
    "date": "date",
    "datetime": "date",
    "timestamp": "date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjclose": "Close",
    "adj_close": "Close",
    "volume": "Volume",
}

_REQUIRED_COLS = ("Open", "High", "Low", "Close")


def normalize_ohlcv(
    df: pd.DataFrame, required: Sequence[str] = _REQUIRED_COLS
) -> pd.DataFrame:
    """Return a copy of *df* with a canonical OHLCV schema.

    - Renames columns case-insensitively to Title-case (or ``date`` for time).
    - Coerces the ``date`` column to UTC datetimes and sorts by it.
    - Drops duplicate dates (keep last) and rows without a ``Close`` value.
    - Validates required columns and leaves extra columns untouched.
    """

    if df is None:
        raise ValueError("Input DataFrame is None")

    d = df.copy()
    if d.empty:
        return d

    rename = {}
    for col in d.columns:
        key = str(col).strip().lower()
        if key in _COL_RENAMES:
            rename[col] = _COL_RENAMES[key]
    if rename:
        d = d.rename(columns=rename)

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
        d = d.dropna(subset=["date"])

    d = d.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    d = d.dropna(subset=["Close"])

    return d.reset_index(drop=True)
