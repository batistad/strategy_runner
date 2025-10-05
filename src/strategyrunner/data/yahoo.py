from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None


def fetch_eod(
    symbols: Iterable[str], history_days: int, interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("Install extra: pip install .[yahoo]")
    tickers = " ".join(symbols)
    df = yf.download(
        tickers=tickers,
        period=f"{history_days}d",
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for sym in symbols:
            sdf = df[
                sym
            ]  # .rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            sdf.index.name = "date"
            out[sym] = sdf.reset_index()
    else:
        sdf = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        sdf.index.name = "date"
        out[list(symbols)[0]] = sdf.reset_index()
    return out
