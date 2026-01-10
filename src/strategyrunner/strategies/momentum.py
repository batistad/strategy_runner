from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from strategyrunner.data.base import normalize_ohlcv


@dataclass
class StrategyParams:
    lookback: int = 126
    top_n: int = 3
    min_volume: int = 1_000_000
    cash_buffer: float = 0.02  # keep some cash
    max_weight: float = 0.33


def compute_signal(df: pd.DataFrame, lookback: int, min_volume: int) -> float:
    clean = normalize_ohlcv(df, required=("Open", "High", "Low", "Close", "Volume"))
    d = clean.tail(lookback)
    if len(d) < lookback:
        return float("nan")
    if d["Volume"].mean() < min_volume:
        return float("nan")
    return (d["Close"].iloc[-1] / d["Close"].iloc[0]) - 1.0


def run_strategy(
    data: Dict[str, pd.DataFrame], params: StrategyParams
) -> Tuple[List[dict], dict]:
    # Rank by momentum
    scores = []
    for sym, df in data.items():
        sig = compute_signal(df, params.lookback, params.min_volume)
        scores.append((sym, sig))

    ranked = [
        s
        for s in sorted(
            scores, key=lambda x: (x[1] if x[1] == x[1] else -1e9), reverse=True
        )
        if s[1] == s[1]
    ]
    picks = ranked[: params.top_n]

    # Weights
    n = max(1, len(picks))
    w = min(params.max_weight, (1.0 - params.cash_buffer) / n)

    trades = [
        {
            "symbol": sym,
            "action": "BUY",
            "weight": round(w, 4),
            "score": round(float(sig), 6),
        }
        for sym, sig in picks
    ]

    metrics = {
        "universe": len(data),
        "picked": len(trades),
        "avg_score": (
            float(pd.Series([p[1] for p in picks]).mean()) if picks else float("nan")
        ),
    }
    return trades, metrics
