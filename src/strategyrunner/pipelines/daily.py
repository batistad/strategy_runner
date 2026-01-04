from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pprint
import time
from decimal import Decimal
from typing import Any, Dict, List, Tuple

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
        time.sleep(wait_s)


def _truncate_to_asof(df: pd.DataFrame, asof_date: dt.date | None) -> pd.DataFrame:
    if asof_date is None:
        return df
    d = df.copy()
    # ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(d["date"]):
        d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce")
    return d[d["date"].dt.date <= asof_date]


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


def _expand_instances(cfg: dict, symbols: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (instances, unique_base_symbols).
    Supports legacy configs with only `symbols`, and new configs with:
    instances:
        - id: TSLA-long
        symbol: TSLA
        overrides: { shorting: none, fast: 55, slow: 155, resample: D }
        - id: TSLA-short
        symbol: TSLA
        overrides: { shorting: short_only, fast: 21, slow: 55, resample: D }
    """
    if cfg.get("instances"):
        instances: List[Dict[str, Any]] = []
        for ins in cfg["instances"]:
            instances.append(
                {
                    "id": str(ins["id"]),
                    "symbol": str(ins["symbol"]),
                    "overrides": dict(ins.get("overrides", {}) or {}),
                    "webhook": ins.get("webhook"),
                    "sizing": ins.get("sizing"),
                }
            )
    else:
        instances = [
            {"id": s, "symbol": s, "overrides": {}, "webhook": None, "sizing": None} 
            for s in symbols
        ]

    base_symbols = sorted({it["symbol"] for it in instances})
    return instances, base_symbols

def _merge_params_for_instances(instances: List[Dict[str, Any]], params_cfg: dict) -> Dict[str, dict]:
    """defaults -> per_symbol[base] -> instance.overrides, keyed by instance id."""
    dflt = params_cfg.get("defaults", {}) or {}
    per = params_cfg.get("per_symbol", {}) or {}
    out: Dict[str, dict] = {}
    for it in instances:
        base = it["symbol"]
        pid = it["id"]
        m = dict(dflt)
        m.update(dict(per.get(base, {}) or {}))
        m.update(dict(it.get("overrides", {}) or {}))
        out[pid] = m
    return out


# --- Webhook helpers (per-instance dispatch) ---

class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _resolve_instance_webhook(cfg: dict, inst: dict) -> dict:
    base = dict(cfg.get("webhook", {}) or {})
    wh_ref = inst.get("webhook")
    if isinstance(wh_ref, str):  # named webhook
        named = (cfg.get("webhooks", {}) or {}).get(wh_ref, {})
        base.update(named)
    elif isinstance(wh_ref, dict):  # inline overrides
        base.update(wh_ref)
    return base


def _summarize_trades(trades: List[dict]) -> str:
    if not trades:
        return "no-trades"
    parts = []
    for t in trades:
        act = t.get("action") or (f"TARGET={t.get('target')}" if t.get("type") == "TARGET" else "HOLD")
        parts.append(f"{t.get('instance', t.get('symbol'))}:{act}:qty={t.get('qty')}@{t.get('price')}")
    return ", ".join(parts)


def _render_message(template: str, ctx: dict) -> str:
    """Safely render a message template with context using SafeDict.
    Any missing keys will be left in {braces} rather than raising.
    """
    try:
        return str(template).format_map(_SafeDict(ctx))
    except Exception:
        return str(template)


# --- Sizing + Position helpers ---

def _resolve_instance_sizing(cfg: dict, inst: dict) -> dict:
    base = dict(cfg.get("sizing", {}) or {})
    if isinstance(inst.get("sizing"), dict):
        base.update(inst["sizing"])  # override
    return base


def _last_close_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1]) if len(df) and "Close" in df.columns else 0.0


def _round_qty(qty: Decimal, lot_size: int, min_qty: int) -> int:
    if lot_size <= 0:
        lot_size = 1
    # floor to lot multiple
    lots = (qty // Decimal(lot_size)) * Decimal(lot_size)
    out = int(lots)
    return max(out, int(min_qty)) if out > 0 else 0


def _compute_entry_qty(sizing: dict, price: float) -> int:
    mode = str(sizing.get("mode", "notional")).lower()
    lot_size = int(sizing.get("lot_size", 1))
    min_qty = int(sizing.get("min_qty", 1))

    if price <= 0:
        return 0

    if mode == "fixed":
        base_qty = int(sizing.get("fixed_qty", 1))
        return _round_qty(Decimal(base_qty), lot_size, min_qty)

    if mode in ("percent_of_cash", "percent"):
        pct = float(sizing.get("percent", 0.0))
        cash_env = sizing.get("cash_env")
        cash = float(os.getenv(cash_env, "0") or 0) if cash_env else 0.0
        notional = max(0.0, cash * pct)
    else:
        notional = float(sizing.get("notional", 0.0))

    raw_qty = Decimal(notional) / Decimal(price) if price > 0 else Decimal(0)
    return _round_qty(raw_qty, lot_size, min_qty)


def _apply_position_delta(curr_pos: int, delta: int) -> int:
    return int(curr_pos + delta)


def _expand_action_to_deltas(
    action: str,
    entry_qty: int,
    curr_pos: int,
    signal_curr: int | None = None,
) -> List[int]:
    """
    Returns signed deltas (buy>0, sell<0) for an action, across modes.

    Simple mode:
      HOLD, BUY (enter long), SELL (enter short), CLOSELONG, CLOSESHORT

    Verbose mode (still supported):
      BUY, SELL (exit long), SHORT, COVER, FLIP_TO_LONG, FLIP_TO_SHORT

    Disambiguation for 'SELL':
      - If signal_curr == -1  -> treat as enter short
      - If signal_curr == 0   -> treat as exit long
      - Fallback: if already long -> exit long, else -> enter short
    """
    if action in ("HOLD", None):
        return []

    # --- Simple mode canonical names ---
    if action == "BUY":  # enter long
        if curr_pos > 0:
            return []
        if curr_pos < 0:
            return [abs(curr_pos), max(entry_qty, 0)]
        return [max(entry_qty, 0)]

    if action == "SELL":  # enter short in simple mode
        if signal_curr == -1:
            # enter short
            if curr_pos < 0:
                return []
            if curr_pos > 0:
                return [-max(curr_pos, 0), -max(entry_qty, 0)]
            return [-max(entry_qty, 0)]
        if signal_curr == 0:
            # exit long
            return [-max(curr_pos, 0)] if curr_pos > 0 else []
        # Fallback heuristic:
        if curr_pos > 0:
            return [-max(curr_pos, 0)]
        return [-max(entry_qty, 0)]

    if action in ("CLOSELONG",):
        return [-max(curr_pos, 0)] if curr_pos > 0 else []

    if action in ("CLOSESHORT",):
        return [abs(min(curr_pos, 0))] if curr_pos < 0 else []

    # --- Verbose compatibility ---
    if action in ("SHORT", "ENTER_SHORT"):
        if curr_pos < 0:
            return []
        if curr_pos > 0:
            return [-max(curr_pos, 0), -max(entry_qty, 0)]
        return [-max(entry_qty, 0)]

    if action in ("SELL", "EXIT_LONG"):
        return [-max(curr_pos, 0)] if curr_pos > 0 else []

    if action in ("COVER", "EXIT_SHORT"):
        return [abs(min(curr_pos, 0))] if curr_pos < 0 else []

    if action == "FLIP_TO_SHORT":
        legs = []
        if curr_pos > 0:
            legs.append(-curr_pos)
        legs.append(-max(entry_qty, 0))
        return legs

    if action == "FLIP_TO_LONG":
        legs = []
        if curr_pos < 0:
            legs.append(abs(curr_pos))
        legs.append(max(entry_qty, 0))
        return legs

    return []


def _deltas_from_target(target: int, entry_qty: int, curr_pos: int) -> List[int]:
    """Compute deltas to move from curr_pos to desired signed position size.
       We interpret target ∈ {-1,0,1} as direction *entry_qty* (absolute size).
    """
    desired = int(target) * int(entry_qty)
    delta = desired - int(curr_pos)
    if delta == 0:
        return []
    # Represent flips as two legs for clarity when crossing zero
    if curr_pos > 0 and delta < 0 and desired < 0:
        # sell current, then short desired abs
        return [-curr_pos, desired]
    if curr_pos < 0 and delta > 0 and desired > 0:
        # cover current, then buy desired
        return [abs(curr_pos), desired]
    return [delta]


def run_daily(config_path: str, asof: str | None, dry: bool) -> bool:
    setup_logging()
    cfg = _load_cfg(config_path)

    market = cfg.get("market", "XNAS")
    buffer_min = int(cfg.get("session_buffer_minutes", 15))
    symbols = cfg["symbols"]

    params_cfg = cfg.get("params", {})
    strategy_name = str(params_cfg.get("name", "momentum")).lower()

    state_path = cfg.get("state", {}).get("path", ".runner_state.json")

    # --- Dry-run state isolation ---
    dry_state_path = cfg.get("state", {}).get("dry_path") or (state_path + ".dry")
    if dry:
        # Seed dry state from prior dry file if it exists; otherwise from the real file.
        state = _load_state(dry_state_path)
        if not state:
            state = _load_state(state_path)
        state_target_path = dry_state_path
    else:
        state = _load_state(state_path)
        state_target_path = state_path
    log.info("State file (load/save): %s", state_target_path)

    last_signals: Dict[str, int] = state.get("last_signals", {})
    positions: Dict[str, int] = state.get("positions", {})

    # If we already have a live position, align the previous signal to its sign
    for pid, qty in positions.items():
        if isinstance(qty, int) and qty != 0:
            last_signals[pid] = 1 if qty > 0 else -1

    asof_date = dt.date.fromisoformat(asof) if asof else None
    today = asof_date if asof_date else dt.date.today()
    if not asof and not is_trading_day(market, today):
        log.info("Non-trading day for %s, exiting.", market)
        return True

    if not asof:
        _wait_until_after_close(market, buffer_min)

    data_cfg = cfg.get("data", {})
    provider = data_cfg.get("provider", "yahoo")
    history_days = int(data_cfg.get("history_days", 400))
    interval = data_cfg.get("interval", "1d")

    # --- Fetch once per *base symbol* (not per instance) ---
    instances, base_symbols = _expand_instances(cfg, symbols)
    fetch_list = base_symbols


    if provider == "yahoo":
        raw = yahoo_data.fetch_eod(
            fetch_list, history_days=history_days, interval=interval
        )
        log.info("Fetched data from Yahoo for %d symbols", len(raw))
    else:
        raise RuntimeError(f"Unknown data provider: {provider}")

    if strategy_name == "crossover":
        # Per-instance merged params: defaults -> per_symbol[base] -> overrides
        merged_params_by_instance = _merge_params_for_instances(instances, params_cfg)

        # Optional per-instance resample and build data dict keyed by *instance id*
        data_by_instance: Dict[str, pd.DataFrame] = {}
        prices_by_instance: Dict[str, float] = {}
        for it in instances:
            pid = it["id"]
            base = it["symbol"]
            df = raw[base]
            rule = merged_params_by_instance[pid].get("resample")
            df_resampled = _maybe_resample(df, rule)
            df_cut = _truncate_to_asof(df_resampled, asof_date)
            data_by_instance[pid] = df_cut
            # use the price from the truncated (resampled) frame for sizing
            prices_by_instance[pid] = _last_close_price(df_cut)

        # Build StrategyParams per instance id
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

        # Run crossover once over the *instance-keyed* dicts
        trades_raw, metrics, new_signals = run_crossover(
            data_by_instance, xparams_by_instance, last_signals
        )

        # Map trades back to include base symbol and instance id
        pid_to_base = {it["id"]: it["symbol"] for it in instances}
        sizing_by_pid = {it["id"]: _resolve_instance_sizing(cfg, it) for it in instances}
        trades: List[dict] = []
        proposed_positions = {}
        for t in trades_raw:
            pid = t.get("symbol")  # in this call, symbol==instance id
            base = pid_to_base.get(pid, pid)
            price = prices_by_instance.get(pid, 0.0)
            sizing = sizing_by_pid[pid]
            entry_qty = _compute_entry_qty(sizing, price)
            # Determine deltas to execute based on action/target and current position
            curr_pos = int(positions.get(pid, 0))
            deltas: List[int] = []
            if t.get("type") == "TARGET":
                deltas = _deltas_from_target(int(t.get("target", 0)), entry_qty, curr_pos)
                action_label = f"TARGET={t.get('target')}"
            else:
                action_label = t.get("action", "HOLD")
                deltas = _expand_action_to_deltas(
                    action_label, entry_qty, curr_pos, t.get("signal_curr")
                )

            # For each delta, create an order leg and update local position accumulator
            for i, d in enumerate(deltas):
                if d == 0:
                    continue
                order_side = "BUY" if d > 0 else "SELL"
                qty = abs(d)
                new_pos = _apply_position_delta(curr_pos, d)
                leg = {
                    "symbol": base,
                    "instance": pid,
                    "signal_prev": t["signal_prev"],
                    "signal_curr": t["signal_curr"],
                    "price": price,
                    "qty": qty,
                    "notional": float(qty * price),
                    "order": order_side,   # BUY / SELL
                    "pos_before": curr_pos,
                    "pos_after": new_pos,
                    "leg_index": i,
                }
                # Keep high-level fields for compatibility
                if t.get("type") == "TARGET":
                    leg["type"] = "TARGET"
                    leg["target"] = t.get("target")
                else:
                    leg["action"] = action_label
                trades.append(leg)
                curr_pos = new_pos  # advance for multi-leg flips

            # collect proposed end positions; do NOT commit yet
            proposed_positions[pid] = curr_pos

        used_params = merged_params_by_instance
    else:
        data = raw
        mparams = MomentumParams(
            **params_cfg, max_weight=cfg.get("risk", {}).get("max_weight", 0.33)
        )
        trades, metrics = run_momentum(data, mparams)
        new_signals = {}
        used_params = params_cfg

    payload_base: dict[str, Any] = {
        "asof": (asof or today.isoformat()),
        "market": market,
        "symbols": base_symbols if cfg.get("instances") else symbols,
        "instances": [{"id": it["id"], "symbol": it["symbol"]} for it in instances],
        "strategy": strategy_name,
        "params": used_params,
        "metrics": metrics,
    }

    # Prepare commit/revert containers
    prev_positions = dict(positions)                     # snapshot
    prev_signals   = dict(state.get("last_signals", {})) # snapshot

    committed_positions = dict(prev_positions)           # start as previous
    committed_signals   = dict(prev_signals)             # start as previous

    # --- Webhook dispatch ---
    if not dry:
        # Group trades by instance (crossover) or send once (momentum)
        if strategy_name == "crossover":
            trades_by_pid: Dict[str, List[dict]] = {}
            for t in trades:
                trades_by_pid.setdefault(t["instance"], []).append(t)

            for it in instances:
                pid = it["id"]
                inst_wh = _resolve_instance_webhook(cfg, it)
                if not inst_wh.get("enabled", True):
                    continue
                send_metrics = bool(inst_wh.get("send_metrics", True))
                trades_i = trades_by_pid.get(pid, [])
                if not trades_i and not send_metrics:
                    continue  # nothing to send

                payload = dict(payload_base)
                payload["trades"] = trades_i
                payload["instance"] = {"id": pid, "symbol": it["symbol"]}
                payload["params"] = {pid: used_params.get(pid, {})}

                # Optional message templating
                tmpl = inst_wh.get("message_template")
                if tmpl:
                    first = trades_i[0] if trades_i else {}
                    ctx = {
                        "asof": payload["asof"],
                        "market": market,
                        "strategy": strategy_name,
                        "symbol": it["symbol"],
                        "instance": payload["instance"],
                        "params": payload["params"][pid],
                        "metrics": metrics,
                        "trades": trades_i,
                        "trades_summary": _summarize_trades(trades_i),
                        "trade": first,
                        "action_or_target": (first.get("action") if first and "action" in first else f"TARGET={first.get('target')}" if first and "target" in first else "HOLD"),
                    }
                    payload["message"] = _render_message(tmpl, ctx)

                url_env = inst_wh.get("url_env")
                if not url_env:
                    raise RuntimeError("webhook.url_env not set (instance or global)")
                url = os.getenv(url_env)
                if not url:
                    raise RuntimeError(f"Environment variable {url_env} is not set")
                secret_env = inst_wh.get("secret_env")
                timeout = int(inst_wh.get("timeout", 10))
                try:
                    post_json(url, payload, secret_env=secret_env, timeout=timeout)
                    # --- COMMIT on success ---
                    # Position: only for crossover instances that had proposed positions
                    if pid in (proposed_positions or {}):
                        committed_positions[pid] = proposed_positions[pid]
                    # Signal: only for instances that produced a new signal
                    if pid in (new_signals or {}):
                        committed_signals[pid] = int(new_signals[pid])
                except Exception as e:
                    log.error("Webhook for instance %s failed: %s", pid, e)
                    # --- REVERT: keep previous state for this instance ---
                    # committed_positions[pid] and committed_signals[pid] already hold previous values
                    continue

        else:
            # momentum: send once using global webhook
            wh_cfg = cfg.get("webhook", {})
            if wh_cfg.get("enabled", True):
                payload = dict(payload_base)
                payload["trades"] = trades
                url_env = wh_cfg.get("url_env")
                if not url_env:
                    raise RuntimeError("webhook.url_env not set in config")
                url = os.getenv(url_env)
                if not url:
                    raise RuntimeError(f"Environment variable {url_env} is not set")
                secret_env = wh_cfg.get("secret_env")
                timeout = int(wh_cfg.get("timeout", 10))
                try:
                    post_json(url, payload, secret_env=secret_env, timeout=timeout)
                    for pid, sig in (new_signals or {}).items():
                        committed_signals[pid] = int(sig)
                except Exception as e:
                    log.error("Webhook failed (momentum): %s", e)
                    # keep previous signals

    else:
        # DRY RUN: commit all proposed positions & new signals (simulate success)
        if strategy_name == "crossover":
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

    # Final save: merge signals to avoid clobbering untouched instances
    merged_signals = dict(prev_signals)
    merged_signals.update(committed_signals)

    state.update({
        "last_asof": payload_base["asof"],
        "last_metrics": metrics,
        "last_signals": merged_signals,
        "positions": committed_positions,
    })

    # Tag the state so it’s obvious which file is which
    if dry:
        state["mode"] = "dry"
    else:
        state.pop("mode", None)

    _save_state(state_target_path, state)


    log.info("Done. %d trades, metrics=%s", len(trades), metrics)
    return True
