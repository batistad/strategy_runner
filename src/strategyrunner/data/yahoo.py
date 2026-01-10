from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None

from .base import normalize_ohlcv


def fetch_eod(
    symbols: Iterable[str], history_days: int, interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("Install extra: pip install .[yahoo]")
    symbols = list(symbols)
    if not symbols:
        return {}
    tickers = " ".join(symbols)
    df = yf.download(
        tickers=tickers,
        period=f"{history_days}d",
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("Yahoo Finance returned no data")
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for sym in symbols:
            sdf = df[sym]
            sdf.index.name = "date"
            out[sym] = normalize_ohlcv(sdf.reset_index())
    else:
        sdf = df.copy()
        sdf.index.name = "date"
        out[symbols[0]] = normalize_ohlcv(sdf.reset_index())
    return out
