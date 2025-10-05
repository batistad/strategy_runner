from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pprint
import time
from typing import Any, Dict

import pandas as pd
import yaml

from ..data import yahoo as yahoo_data
from ..strategies.crossover import StrategyParams as XParams
from ..strategies.crossover import run_strategy as run_crossover
from ..strategies.momentum import StrategyParams as MomentumParams
from ..strategies.momentum import run_strategy as run_momentum
from ..utils.calendars import is_trading_day, session_close_utc
from ..utils.logging import setup_logging
from ..utils.webhooks import post_json

log = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_state(path: str) -> dict:
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def _save_state(path: str, state: dict) -> None:
    json.dump(state, open(path, "w"), indent=2)


def _wait_until_after_close(market: str, buffer_minutes: int) -> None:
    close_utc = session_close_utc(market)
    now = dt.datetime.now(dt.timezone.utc)
    target = close_utc + dt.timedelta(minutes=buffer_minutes)
    if now < target:
        wait_s = (target - now).total_seconds()
        log.info("Waiting %.0fs for post-close window…", wait_s)
        time.sleep(wait_s)  # keep short for demo; replace with wait_s in real run


def _maybe_resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if not rule:
        return df
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in cols:
        agg["Volume"] = "sum"
    out = (
        df.set_index("date")
        .resample(rule)
        .agg(agg)
        .dropna(subset=["Close"])
        .reset_index()
    )
    out.rename(columns={"index": "date"}, inplace=True)
    return out


def _merge_symbol_params(symbols: list[str], params_cfg: dict) -> Dict[str, dict]:
    dflt = params_cfg.get("defaults", {})
    per = params_cfg.get("per_symbol", {}) or {}
    merged: Dict[str, dict] = {}
    for s in symbols:
        m = dict(dflt)
        m.update(per.get(s, {}))
        merged[s] = m
    return merged


def run_daily(config_path: str, asof: str | None, dry: bool) -> bool:
    setup_logging()
    cfg = _load_cfg(config_path)

    market = cfg.get("market", "XNAS")
    buffer_min = int(cfg.get("session_buffer_minutes", 15))
    symbols = cfg["symbols"]

    params_cfg = cfg.get("params", {})
    strategy_name = str(params_cfg.get("name", "momentum")).lower()

    state_path = cfg.get("state", {}).get("path", ".runner_state.json")
    state = _load_state(state_path)
    last_signals: Dict[str, int] = state.get("last_signals", {})

    today = dt.date.fromisoformat(asof) if asof else dt.date.today()
    if not asof and not is_trading_day(market, today):
        log.info("Non-trading day for %s, exiting.", market)
        return True

    if not asof:
        _wait_until_after_close(market, buffer_min)

    data_cfg = cfg.get("data", {})
    provider = data_cfg.get("provider", "yahoo")
    history_days = int(data_cfg.get("history_days", 400))
    interval = data_cfg.get("interval", "1d")

    if provider == "yahoo":
        raw = yahoo_data.fetch_eod(
            symbols, history_days=history_days, interval=interval
        )
        log.info("Fetched data from Yahoo for %d symbols", len(raw))
    else:
        raise RuntimeError(f"Unknown data provider: {provider}")

    if strategy_name == "crossover":
        merged_params = _merge_symbol_params(symbols, params_cfg)
        # Optional resample per the DEFAULT rule (per-symbol override allowed too)
        data = {}
        for s, df in raw.items():
            rule = merged_params[s].get("resample")
            data[s] = _maybe_resample(df, rule)
        trades, metrics, new_signals = run_crossover(
            data,
            {
                s: XParams(
                    ma_type=str(merged_params[s].get("ma_type", "EMA")),
                    fast=int(merged_params[s].get("fast", 55)),
                    slow=int(merged_params[s].get("slow", 155)),
                    shorting=str(merged_params[s].get("shorting", "none")),
                    chandelier_len=(
                        None
                        if merged_params[s].get("chandelier_len") in (None, "", "null")
                        else int(merged_params[s].get("chandelier_len"))
                    ),
                    chandelier_mult=(
                        None
                        if merged_params[s].get("chandelier_mult") in (None, "", "null")
                        else float(merged_params[s].get("chandelier_mult"))
                    ),
                )
                for s in symbols
            },
            last_signals,
        )
        used_params = merged_params
    else:
        data = raw
        mparams = MomentumParams(
            **params_cfg, max_weight=cfg.get("risk", {}).get("max_weight", 0.33)
        )
        trades, metrics = run_momentum(data, mparams)
        new_signals = {}
        used_params = params_cfg

    payload: dict[str, Any] = {
        "asof": (asof or today.isoformat()),
        "market": market,
        "symbols": symbols,
        "strategy": strategy_name,
        "params": used_params,
        "trades": trades,
        "metrics": metrics,
    }

    state.update(
        {
            "last_asof": payload["asof"],
            "last_metrics": metrics,
            "last_signals": new_signals,
        }
    )
    _save_state(state_path, state)

    wh_cfg = cfg.get("webhook", {})
    if wh_cfg.get("enabled", True) and not dry:
        url_env = wh_cfg.get("url_env")
        if not url_env:
            raise RuntimeError("webhook.url_env not set in config")
        url = os.getenv(url_env)
        if not url:
            raise RuntimeError(f"Environment variable {url_env} is not set")
        secret_env = wh_cfg.get("secret_env")
        timeout = int(wh_cfg.get("timeout", 10))
        post_json(url, payload, secret_env=secret_env, timeout=timeout)
    else:
        log.info("Dry run or webhook disabled; payload not sent.")
        log.info(
            "TRADES PAYLOAD:\n%s", pprint.pformat(payload, compact=False, width=100)
        )

    log.info("Done. %d trades, metrics=%s", len(trades), metrics)
    return True
