# Strategy Runner

A config-driven daily runner for trading strategies. This repository provides a small production-oriented runner that fetches market data, computes signals for configured strategies, emits trade actions, and posts a webhook payload (or logs) for downstream execution.

This README focuses on project-wide usage and the crossover strategy implementation.

---

## Table of contents
- Overview
- Installation
- Quick start (CLI)
- Configuration
- Crossover strategy (detailed)
  - Supported MAs
  - Signal logic & timing
  - Shorting modes
  - Chandelier (ATR) stop overlay
  - Outputs
- Data providers & resampling
- Development & testing
- Files of interest
- Next steps

---

## Overview
- Main entry: `src/strategyrunner/cli.py` → runs the daily pipeline.
- Orchestration: `src/strategyrunner/pipelines/daily.py`.
- Strategies live in `src/strategyrunner/strategies/` — this README documents the crossover strategy in depth.
- State (last signals/metrics) persists to a JSON file (configurable).

---

## Installation
Recommended: create and activate a virtual environment.

Windows (PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e .
# To use Yahoo provider:
pip install -e .[yahoo]
```

---

## Quick start (CLI)
Prepare a config (use `config.example.yaml` as a template). Example run:
```powershell
strategyrunner run --config config.example.yaml
# dry-run for debugging:
strategyrunner run --config config.example.yaml --dry
# run for a historical date (uses next-bar semantics):
strategyrunner run --config config.example.yaml --asof 2025-09-12 --dry
```

---

## Configuration
Key sections in `config.example.yaml`:
- `market` — market identifier.
- `symbols` — list of tickers.
- `params` — strategy selector and param defaults / per-symbol overrides.
  - `params.name`: `"crossover"` or `"momentum"`.
  - `params.defaults` / `params.per_symbol` for `ma_type`, `fast`, `slow`, `shorting`, `chandelier_len`, `chandelier_mult`, `resample`.
- `data` — provider, interval, history_days.
- `webhook` — env var for URL, optional secret env var, timeout.
- `state.path` — where to persist last signals/metrics.

---

## Crossover strategy (detailed)

Location: `src/strategyrunner/strategies/crossover.py`

Purpose: generate actionable long/short/flat signals per symbol using moving-average crossovers, with optional ATR-based Chandelier stops and cooldown behavior to avoid immediate re-entry after stops.

### Supported moving averages
The strategy implements (and accepts `ma_type`):
- SMA, EMA, WMA, HMA, ALMA, DEMA, TEMA, TRIMA, VWMA, KAMA

ALMA uses sensible defaults (offset/sigma) in code. See the file for exact formulas.

### Parameters (per symbol)
- `ma_type` (string) — one of the MAs above (default in example: `EMA`)
- `fast` (int) — fast MA window (default 55)
- `slow` (int) — slow MA window (default 155)
- `shorting` (string) — `none` | `short_only` | `long_short`
- `chandelier_len` (int or null) — ATR lookback (null disables)
- `chandelier_mult` (float or null) — ATR multiplier for stop (null disables)
- `resample` (pandas offset alias or null) — optional per-symbol resampling (e.g., `D`, `W`, `M`)

### Signal logic & timing
- Primary signal derived from fast MA vs slow MA crossover:
  - fast > slow → bullish cross → long signal
  - fast < slow → bearish cross → flat (or short depending on shorting)
- Next-bar execution: signals are shifted so a crossover observed on bar t becomes actionable on bar t+1 (no look-ahead).
- The strategy computes the most recent shifted signal for each symbol, compares it with stored `last_signals` from state, and emits trade actions describing transitions (enter/exit/flip).

### Shorting modes
- `none` — long-only. Signals: 0 or 1.
- `short_only` — only allow shorts. Signals: 0 or -1.
- `long_short` — allow both sides. Signals: -1, 0, 1.

### Chandelier (ATR) stop overlay
- Optional overlay that computes ATR (Wilder) over `chandelier_len` and applies a `chandelier_mult` multiplier to create a trailing stop level.
- When the stop is hit, it forces exit of the current position and applies a cooldown that prevents immediate re-entry until a fresh MA crossover occurs.
- In `long_short` mode the stop logic supports analogous behavior for short positions.

### Outputs from run_strategy
Function signature (used by the pipeline):
- `run_crossover(data: Dict[str, pd.DataFrame], params_map: Dict[str, StrategyParams], last_signals: Dict[str,int]) -> (trades, metrics, new_signals)`

Returned:
- `trades`: list of dicts with keys like `symbol`, `action`, `signal_prev`, `signal_curr`.
- `metrics`: runtime metrics (universe size, active count, etc.).
- `new_signals`: dict of current signals per symbol (stored to state).

---

## Data providers & resampling
- Built-in provider: Yahoo via `yfinance` (optional extra). Implemented in `src/strategyrunner/data/yahoo.py`.
- Pipeline expects OHLCV labeled columns (Title-case: `Open`, `High`, `Low`, `Close`, `Volume`) for resampling. If using alternate sources, normalize column names to this schema before invoking the strategy.

---

## Development & testing
- Run unit tests (if present) with pytest:
```powershell
pip install -e .[test]
pytest -q
```
- Logging setup is in `src/strategyrunner/utils/logging.py`.
- State file default: `.runner_state.json` (ignored by .gitignore).

---

## Files of interest
- Pipeline: `src/strategyrunner/pipelines/daily.py`
- Crossover strategy: `src/strategyrunner/strategies/crossover.py`
- Momentum strategy: `src/strategyrunner/strategies/momentum.py`
- Data: `src/strategyrunner/data/yahoo.py`
- Webhook helper: `src/strategyrunner/utils/webhooks.py`
- Config example: `config.example.yaml`

---
```