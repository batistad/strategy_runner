from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Moving averages ---


def _sma(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).mean()


def _ema(x: pd.Series, w: int) -> pd.Series:
    return x.ewm(span=w, adjust=False).mean()


def _wma(x: pd.Series, w: int) -> pd.Series:
    weights = np.arange(1, w + 1, dtype=float)
    return x.rolling(w).apply(
        lambda a: float(np.dot(a, weights)) / weights.sum(), raw=True
    )


def _hma(x: pd.Series, w: int) -> pd.Series:
    half = max(1, w // 2)
    sqrtw = max(1, int(np.sqrt(w)))
    wma_half = _wma(x, half)
    wma_full = _wma(x, w)
    return _wma(2 * wma_half - wma_full, sqrtw)


def _alma(x: pd.Series, w: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    m = offset * (w - 1)
    s = w / sigma
    idx = np.arange(w, dtype=float)
    weights = np.exp(-((idx - m) ** 2) / (2 * s * s))
    weights /= weights.sum()
    return x.rolling(w).apply(lambda a: float(np.dot(a, weights)), raw=True)


def _dema(x: pd.Series, w: int) -> pd.Series:
    e1 = _ema(x, w)
    e2 = _ema(e1, w)
    return 2 * e1 - e2


def _tema(x: pd.Series, w: int) -> pd.Series:
    e1 = _ema(x, w)
    e2 = _ema(e1, w)
    e3 = _ema(e2, w)
    return 3 * (e1 - e2) + e3


def _trima(x: pd.Series, w: int) -> pd.Series:
    w1 = int(np.ceil(w / 2))
    w2 = int(np.floor(w / 2))
    return _sma(_sma(x, w1), max(1, w2))


def _vwma(close: pd.Series, vol: pd.Series, w: int) -> pd.Series:
    pv = (close * vol).rolling(w).sum()
    vs = vol.rolling(w).sum()
    return pv / vs


def _kama(close: pd.Series, w: int) -> pd.Series:
    change = close.diff(w).abs()
    volatility = close.diff().abs().rolling(w).sum()
    er = change / volatility
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=close.index, dtype=float)
    start = close.first_valid_index()
    if start is not None:
        kama.loc[start] = close.loc[start]
        for t in range(close.index.get_loc(start) + 1, len(close)):
            prev = kama.iloc[t - 1]
            price = close.iloc[t]
            alpha = sc.iloc[t] if not np.isnan(sc.iloc[t]) else slow_sc**2
            kama.iloc[t] = prev + alpha * (price - prev)
    return kama


_MA_FUNCS = {
    "SMA": lambda c, v, w: _sma(c, w),
    "EMA": lambda c, v, w: _ema(c, w),
    "WMA": lambda c, v, w: _wma(c, w),
    "HMA": lambda c, v, w: _hma(c, w),
    "ALMA": lambda c, v, w: _alma(c, w),
    "DEMA": lambda c, v, w: _dema(c, w),
    "TEMA": lambda c, v, w: _tema(c, w),
    "TRIMA": lambda c, v, w: _trima(c, w),
    "VWMA": lambda c, v, w: _vwma(c, v, w),
    "KAMA": lambda c, v, w: _kama(c, w),
}

# --- Chandelier Exit ---


def _atr_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _chandelier_long(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int, mult: float
) -> pd.Series:
    atr = _atr_wilder(high, low, close, length)
    hhv = high.rolling(length).max()
    return hhv - mult * atr


def _apply_stop_with_cooldown_signed(
    base_signal_t1: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    length: int,
    mult: float,
    enable_short: bool,
) -> pd.Series:
    stop_long = _chandelier_long(high, low, close, length, mult)
    stop_short = None
    if enable_short:
        atr = _atr_wilder(high, low, close, length)
        llv = low.rolling(length).min()
        stop_short = llv + mult * atr
    base = base_signal_t1.fillna(0).astype(int).to_numpy()
    c = close.to_numpy(dtype=float)
    sl = stop_long.to_numpy(dtype=float)
    ss = stop_short.to_numpy(dtype=float) if stop_short is not None else None
    pos = base.copy()
    cool_long = seen_zero_long = False
    cool_short = seen_zero_short = False
    for t in range(1, len(base)):
        desired = base[t]
        if pos[t - 1] == 1 and c[t] < sl[t]:
            desired = 0
            cool_long = True
            seen_zero_long = False
        if ss is not None and pos[t - 1] == -1 and c[t] > ss[t]:
            desired = 0
            cool_short = True
            seen_zero_short = False
        if cool_long:
            if not seen_zero_long:
                if desired == 0:
                    seen_zero_long = True
                else:
                    desired = 0
            else:
                if not (base[t - 1] == 0 and base[t] == 1):
                    desired = 0
                else:
                    cool_long = False
        if cool_short:
            if not seen_zero_short:
                if desired == 0:
                    seen_zero_short = True
                else:
                    desired = 0
            else:
                if not (base[t - 1] == 0 and base[t] == -1):
                    desired = 0
                else:
                    cool_short = False
        pos[t] = desired
    return pd.Series(pos, index=base_signal_t1.index)


# --- Params ---


@dataclass
class StrategyParams:
    ma_type: str = "EMA"
    fast: int = 55
    slow: int = 155
    shorting: str = "none"  # none | short_only | long_short
    chandelier_len: Optional[int] = None
    chandelier_mult: Optional[float] = None


# --- internals ---


def _signal_from_ma(df: pd.DataFrame, p: StrategyParams) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain Close column")
    c = df["Close"].astype(float)
    v = (
        df["Volume"].astype(float)
        if "Volume" in df.columns
        else pd.Series(index=df.index, dtype=float)
    )
    key = p.ma_type.upper()
    if key not in _MA_FUNCS:
        raise ValueError(f"Unsupported ma_type: {p.ma_type}")
    f = _MA_FUNCS[key](c, v, int(p.fast))
    s = _MA_FUNCS[key](c, v, int(p.slow))
    up = (f > s).astype(int)
    dn = (f < s).astype(int)
    if p.shorting == "none":
        sig_now = up
    elif p.shorting == "short_only":
        sig_now = -dn
    else:
        sig_now = up - dn  # long_short
    return sig_now.shift(1).fillna(0).astype(int)  # next-bar


def _apply_overlays(
    df: pd.DataFrame, sig_t1: pd.Series, p: StrategyParams
) -> pd.Series:
    if p.chandelier_len is None or p.chandelier_mult is None:
        return sig_t1
    return _apply_stop_with_cooldown_signed(
        sig_t1,
        df["Close"],
        df.get("High", df["Close"]),
        df.get("Low", df["Close"]),
        int(p.chandelier_len),
        float(p.chandelier_mult),
        p.shorting != "none",
    )


def _action_from_transition(prev: int, curr: int) -> str:
    if prev == curr:
        return "HOLD"
    trans = (prev, curr)
    mapping = {
        (0, 1): "BUY",
        (1, 0): "SELL",
        (0, -1): "SHORT",
        (-1, 0): "COVER",
        (1, -1): "FLIP_TO_SHORT",
        (-1, 1): "FLIP_TO_LONG",
    }
    return mapping.get(trans, "HOLD")


# --- public API ---


def run_strategy(
    data: Dict[str, pd.DataFrame],
    params_by_symbol: Dict[str, StrategyParams],
    last_signals: Dict[str, int],
) -> Tuple[List[dict], dict, Dict[str, int]]:
    """Return explicit actions per symbol and new last_signals.
    Output trades entries:
      {symbol, action, signal_prev, signal_curr}
    """
    signals_curr: Dict[str, int] = {}
    trades: List[dict] = []

    for sym, df in data.items():
        p = params_by_symbol.get(sym)
        if p is None:
            continue
        sig_t1 = _signal_from_ma(df, p)
        sig_final = _apply_overlays(df, sig_t1, p)
        s = int(sig_final.iloc[-1]) if len(sig_final) else 0
        signals_curr[sym] = s

    for sym in data.keys():
        curr = signals_curr.get(sym, 0)
        prev = int(last_signals.get(sym, 0))
        action = _action_from_transition(prev, curr)
        trades.append(
            {
                "symbol": sym,
                "action": action,
                "signal_prev": prev,
                "signal_curr": curr,
            }
        )

    metrics = {
        "universe": len(data),
        "active": sum(1 for s in signals_curr.values() if s != 0),
    }
    return trades, metrics, signals_curr
