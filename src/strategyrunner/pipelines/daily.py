"""Daily pipeline orchestration for strategy execution.

Orchestrates data fetching, strategy execution, position tracking, and webhook dispatch.
Supports multiple instances, strategies, and stateful position/signal management.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pprint
import time
from typing import Any, Dict, List

import pandas as pd

from .. import config as cfg_module
from ..data import yahoo as yahoo_data
from ..models import PositionTracker, SizingCalculator, expand_action_to_deltas
from ..state import load_state, save_state, merge_signals
from ..strategies.crossover import StrategyParams as XParams
from ..strategies.crossover import run_strategy as run_crossover
from ..strategies.momentum import StrategyParams as MomentumParams
from ..strategies.momentum import run_strategy as run_momentum
from ..utils.calendars import is_trading_day, session_close_utc
from ..utils import constants as K
from ..utils.logging import setup_logging
from ..utils.webhooks import post_json

log = logging.getLogger(__name__)


def _wait_until_after_close(market: str, buffer_minutes: int) -> None:
    """Wait until market close + buffer before returning."""
    close_utc = session_close_utc(market)
    now = dt.datetime.now(dt.timezone.utc)
    target = close_utc + dt.timedelta(minutes=buffer_minutes)
    if now < target:
        wait_s = (target - now).total_seconds()
        log.info("Waiting %.0fs for post-close window…", wait_s)
        time.sleep(wait_s)


def _truncate_to_asof(df: pd.DataFrame, asof_date: dt.date | None) -> pd.DataFrame:
    """Filter dataframe to rows on/before asof_date."""
    if asof_date is None:
        return df
    d = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    return d[d["date"].dt.date <= asof_date]


def _maybe_resample(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """Resample OHLCV bars to coarser frequency if rule is specified."""
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


def _last_close_price(df: pd.DataFrame) -> float:
    """Extract last close price from dataframe."""
    return float(df[K.COL_CLOSE].iloc[-1]) if len(df) and K.COL_CLOSE in df.columns else 0.0


class _SafeDict(dict):
    """Dict that returns placeholder strings for missing keys instead of raising."""
    def __missing__(self, key):
        return "{" + key + "}"


def _render_message(template: str, ctx: dict) -> str:
    """Safely render message template with context (missing keys stay as {placeholders})."""
    try:
        return str(template).format_map(_SafeDict(ctx))
    except Exception:
        return str(template)


def _summarize_trades(trades: List[dict]) -> str:
    """Generate human-readable trade summary string."""
    if not trades:
        return "no-trades"
    parts = []
    for t in trades:
        act = t.get("action") or (f"TARGET={t.get('target')}" if t.get("type") == "TARGET" else "HOLD")
        parts.append(f"{t.get('instance', t.get('symbol'))}:{act}:qty={t.get('qty')}@{t.get('price')}")
    return ", ".join(parts)


def run_daily(config_path: str, asof: str | None, dry: bool) -> bool:
    """Execute daily strategy pipeline.

    1. Load config and state
    2. Check trading day; wait for market close if needed
    3. Fetch data per base symbol
    4. Run strategy (crossover or momentum)
    5. Compute position deltas and expand to trades
    6. Dispatch webhooks (dry-run logs preview; live commits state on success)

    Args:
        config_path: Path to YAML config file.
        asof: Optional YYYY-MM-DD override for backtesting.
        dry: If True, skip webhooks and log payload preview.

    Returns:
        True on success, False otherwise.
    """
    setup_logging()
    cfg = cfg_module.load_config(config_path)

    market = cfg.get(K.KEY_MARKET, K.DEFAULT_MARKET)
    buffer_min = int(cfg.get(K.KEY_SESSION_BUFFER_MIN, K.DEFAULT_SESSION_BUFFER_MINUTES))
    symbols = cfg[K.KEY_SYMBOLS]

    params_cfg = cfg.get(K.KEY_PARAMS, {})
    strategy_name = str(params_cfg.get(K.KEY_PARAMS_NAME, K.STRATEGY_MOMENTUM)).lower()

    state_path = cfg.get(K.KEY_STATE, {}).get(K.KEY_STATE_PATH, ".runner_state.json")
    dry_state_path = cfg.get(K.KEY_STATE, {}).get(K.KEY_STATE_DRY_PATH) or (state_path + ".dry")

    # Load state with dry-run isolation
    if dry:
        state = load_state(dry_state_path) or load_state(state_path)
        state_target_path = dry_state_path
    else:
        state = load_state(state_path)
        state_target_path = state_path
    log.info("State file (load/save): %s", state_target_path)

    last_signals: Dict[str, int] = state.get(K.KEY_STATE_LAST_SIGNALS, {})
    positions: Dict[str, int] = state.get(K.KEY_STATE_POSITIONS, {})

    # Align previous signal to position sign if already in trade
    for pid, qty in positions.items():
        if isinstance(qty, int) and qty != 0:
            last_signals[pid] = 1 if qty > 0 else -1

    # Check trading day
    asof_date = dt.date.fromisoformat(asof) if asof else None
    today = asof_date if asof_date else dt.date.today()
    if not asof and not is_trading_day(market, today):
        log.info("Non-trading day for %s, exiting.", market)
        return True

    if not asof:
        _wait_until_after_close(market, buffer_min)

    # Fetch data
    data_cfg = cfg.get(K.KEY_DATA, {})
    provider = data_cfg.get(K.KEY_DATA_PROVIDER, K.PROVIDER_YAHOO)
    history_days = int(data_cfg.get(K.KEY_DATA_HISTORY_DAYS, K.DEFAULT_HISTORY_DAYS))
    interval = data_cfg.get(K.KEY_DATA_INTERVAL, K.DEFAULT_INTERVAL)

    instances, base_symbols = cfg_module.expand_instances(cfg, symbols)

    if provider == K.PROVIDER_YAHOO:
        raw = yahoo_data.fetch_eod(base_symbols, history_days=history_days, interval=interval)
        log.info("Fetched data from Yahoo for %d symbols", len(raw))
    else:
        raise RuntimeError(f"Unknown data provider: {provider}")

    # Run strategy
    if strategy_name == K.STRATEGY_CROSSOVER:
        merged_params_by_instance = cfg_module.merge_params_for_instances(instances, params_cfg)

        data_by_instance: Dict[str, pd.DataFrame] = {}
        prices_by_instance: Dict[str, float] = {}
        for it in instances:
            pid = it[K.KEY_INSTANCE_ID]
            base = it[K.KEY_INSTANCE_SYMBOL]
            df = raw[base]
            rule = merged_params_by_instance[pid].get("resample")
            df_resampled = _maybe_resample(df, rule)
            df_cut = _truncate_to_asof(df_resampled, asof_date)
            data_by_instance[pid] = df_cut
            prices_by_instance[pid] = _last_close_price(df_cut)

        # Build strategy params
        xparams_by_instance: Dict[str, XParams] = {}
        for pid, mp in merged_params_by_instance.items():
            xparams_by_instance[pid] = XParams(
                ma_type=str(mp.get("ma_type", "EMA")),
                fast=int(mp.get("fast", 55)),
                slow=int(mp.get("slow", 155)),
                shorting=str(mp.get("shorting", "none")),
                chandelier_len=(
                    None
                    if mp.get("chandelier_len") in (None, "", "null")
                    else int(mp.get("chandelier_len"))
                ),
                chandelier_mult=(
                    None
                    if mp.get("chandelier_mult") in (None, "", "null")
                    else float(mp.get("chandelier_mult"))
                ),
                actions_mode=str(mp.get("actions_mode", "verbose")),
            )

        trades_raw, metrics, new_signals = run_crossover(
            data_by_instance, xparams_by_instance, last_signals
        )

        # Expand trades to legs
        pid_to_base = {it[K.KEY_INSTANCE_ID]: it[K.KEY_INSTANCE_SYMBOL] for it in instances}
        trades: List[dict] = []
        proposed_positions = {}
        for t in trades_raw:
            pid = t.get(K.KEY_TRADE_SYMBOL)
            base = pid_to_base.get(pid, pid)
            price = prices_by_instance.get(pid, 0.0)
            sizing = cfg_module.resolve_instance_sizing(cfg, next((it for it in instances if it[K.KEY_INSTANCE_ID] == pid), {}))
            sizing_calc = SizingCalculator(sizing)
            entry_qty = sizing_calc.compute_entry_qty(price)

            curr_pos = int(positions.get(pid, 0))
            tracker = PositionTracker(curr_pos)
            deltas: List[int] = []
            if t.get(K.KEY_TRADE_TYPE) == "TARGET":
                deltas = tracker.deltas_to_target(int(t.get(K.KEY_TRADE_TARGET, 0)), entry_qty)
                action_label = f"TARGET={t.get(K.KEY_TRADE_TARGET)}"
            else:
                action_label = t.get(K.KEY_TRADE_ACTION, "HOLD")
                deltas = expand_action_to_deltas(action_label, entry_qty, curr_pos, t.get(K.KEY_TRADE_SIGNAL_CURR))

            # Expand deltas to legs
            for i, d in enumerate(deltas):
                if d == 0:
                    continue
                order_side = "BUY" if d > 0 else "SELL"
                qty = abs(d)
                new_pos = tracker.apply_delta(d)
                leg = {
                    K.KEY_TRADE_SYMBOL: base,
                    K.KEY_TRADE_INSTANCE: pid,
                    K.KEY_TRADE_SIGNAL_PREV: t[K.KEY_TRADE_SIGNAL_PREV],
                    K.KEY_TRADE_SIGNAL_CURR: t[K.KEY_TRADE_SIGNAL_CURR],
                    K.KEY_TRADE_PRICE: price,
                    K.KEY_TRADE_QTY: qty,
                    K.KEY_TRADE_NOTIONAL: float(qty * price),
                    K.KEY_TRADE_ORDER: order_side,
                    K.KEY_TRADE_POS_BEFORE: curr_pos,
                    K.KEY_TRADE_POS_AFTER: new_pos,
                    K.KEY_TRADE_LEG_INDEX: i,
                }
                if t.get(K.KEY_TRADE_TYPE) == "TARGET":
                    leg[K.KEY_TRADE_TYPE] = "TARGET"
                    leg[K.KEY_TRADE_TARGET] = t.get(K.KEY_TRADE_TARGET)
                else:
                    leg[K.KEY_TRADE_ACTION] = action_label
                trades.append(leg)
                curr_pos = new_pos

            proposed_positions[pid] = curr_pos

        used_params = merged_params_by_instance
    else:
        # Momentum - filter out 'name' key which is not a momentum param
        momentum_config = {k: v for k, v in params_cfg.items() if k != K.KEY_PARAMS_NAME}
        mparams = MomentumParams(
            **momentum_config, max_weight=cfg.get(K.KEY_RISK, {}).get("max_weight", 0.33)
        )
        trades, metrics = run_momentum(raw, mparams)
        new_signals = {}
        used_params = params_cfg
        proposed_positions = {}

    # Build payload
    payload_base: dict[str, Any] = {
        "asof": (asof or today.isoformat()),
        K.KEY_MARKET: market,
        K.KEY_SYMBOLS: base_symbols if cfg.get(K.KEY_INSTANCES) else symbols,
        K.KEY_INSTANCES: [{K.KEY_INSTANCE_ID: it[K.KEY_INSTANCE_ID], K.KEY_INSTANCE_SYMBOL: it[K.KEY_INSTANCE_SYMBOL]} for it in instances],
        "strategy": strategy_name,
        K.KEY_PARAMS: used_params,
        "metrics": metrics,
    }

    # Prepare state commits
    prev_positions = dict(positions)
    prev_signals = dict(state.get(K.KEY_STATE_LAST_SIGNALS, {}))
    committed_positions = dict(prev_positions)
    committed_signals = dict(prev_signals)

    # Webhook dispatch
    if not dry:
        if strategy_name == K.STRATEGY_CROSSOVER:
            trades_by_pid: Dict[str, List[dict]] = {}
            for t in trades:
                trades_by_pid.setdefault(t["instance"], []).append(t)

            for it in instances:
                pid = it[K.KEY_INSTANCE_ID]
                inst_wh = cfg_module.resolve_instance_webhook(cfg, it)
                if not inst_wh.get(K.KEY_WEBHOOK_ENABLED, True):
                    continue
                send_metrics = bool(inst_wh.get(K.KEY_WEBHOOK_SEND_METRICS, True))
                trades_i = trades_by_pid.get(pid, [])
                if not trades_i and not send_metrics:
                    continue

                payload = dict(payload_base)
                payload["trades"] = trades_i
                payload["instance"] = {K.KEY_INSTANCE_ID: pid, K.KEY_INSTANCE_SYMBOL: it[K.KEY_INSTANCE_SYMBOL]}
                payload[K.KEY_PARAMS] = {pid: used_params.get(pid, {})}

                # Message templating
                tmpl = inst_wh.get(K.KEY_WEBHOOK_MESSAGE_TEMPLATE)
                if tmpl:
                    first = trades_i[0] if trades_i else {}
                    ctx = {
                        "asof": payload["asof"],
                        "market": market,
                        "strategy": strategy_name,
                        K.KEY_INSTANCE_SYMBOL: it[K.KEY_INSTANCE_SYMBOL],
                        "instance": payload["instance"],
                        "params": payload["params"][pid],
                        "metrics": metrics,
                        "trades": trades_i,
                        "trades_summary": _summarize_trades(trades_i),
                        "trade": first,
                        "action_or_target": (first.get("action") if first and "action" in first else f"TARGET={first.get('target')}" if first and "target" in first else "HOLD"),
                    }
                    payload["message"] = _render_message(tmpl, ctx)

                url_env = inst_wh.get(K.KEY_WEBHOOK_URL_ENV)
                if not url_env:
                    raise RuntimeError("webhook.url_env not set (instance or global)")
                url = os.getenv(url_env)
                if not url:
                    raise RuntimeError(f"Environment variable {url_env} is not set")
                secret_env = inst_wh.get(K.KEY_WEBHOOK_SECRET_ENV)
                timeout = int(inst_wh.get(K.KEY_WEBHOOK_TIMEOUT, K.DEFAULT_WEBHOOK_TIMEOUT))
                try:
                    post_json(url, payload, secret_env=secret_env, timeout=timeout)
                    if pid in (proposed_positions or {}):
                        committed_positions[pid] = proposed_positions[pid]
                    if pid in (new_signals or {}):
                        committed_signals[pid] = int(new_signals[pid])
                except Exception as e:
                    log.error("Webhook for instance %s failed: %s", pid, e)
                    continue

        else:
            # Momentum
            wh_cfg = cfg.get(K.KEY_WEBHOOK, {})
            if wh_cfg.get(K.KEY_WEBHOOK_ENABLED, True):
                payload = dict(payload_base)
                payload["trades"] = trades
                url_env = wh_cfg.get(K.KEY_WEBHOOK_URL_ENV)
                if not url_env:
                    raise RuntimeError("webhook.url_env not set in config")
                url = os.getenv(url_env)
                if not url:
                    raise RuntimeError(f"Environment variable {url_env} is not set")
                secret_env = wh_cfg.get(K.KEY_WEBHOOK_SECRET_ENV)
                timeout = int(wh_cfg.get(K.KEY_WEBHOOK_TIMEOUT, K.DEFAULT_WEBHOOK_TIMEOUT))
                try:
                    post_json(url, payload, secret_env=secret_env, timeout=timeout)
                    for pid, sig in (new_signals or {}).items():
                        committed_signals[pid] = int(sig)
                except Exception as e:
                    log.error("Webhook failed (momentum): %s", e)

    else:
        # Dry-run
        if strategy_name == K.STRATEGY_CROSSOVER:
            for pid, ppos in (proposed_positions or {}).items():
                committed_positions[pid] = ppos
            for pid, sig in (new_signals or {}).items():
                committed_signals[pid] = int(sig)
        else:
            for pid, sig in (new_signals or {}).items():
                committed_signals[pid] = int(sig)
        payload_preview = dict(payload_base)
        payload_preview["trades"] = trades
        log.info("Dry run; payload preview: %s", pprint.pformat(payload_preview, compact=False, width=100))

    # Save state
    merged_signals = merge_signals(prev_signals, committed_signals)
    state.update({
        K.KEY_STATE_LAST_ASOF: payload_base["asof"],
        K.KEY_STATE_LAST_METRICS: metrics,
        K.KEY_STATE_LAST_SIGNALS: merged_signals,
        K.KEY_STATE_POSITIONS: committed_positions,
    })
    if dry:
        state["mode"] = "dry"
    else:
        state.pop("mode", None)

    save_state(state_target_path, state)
    log.info("Done. %d trades, metrics=%s", len(trades), metrics)
    return True
