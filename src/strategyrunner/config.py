"""Configuration loading, validation, and parameter merging."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import yaml

from .utils import constants as K

log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {path} is empty or invalid YAML")
    return cfg


def expand_instances(
    cfg: dict, symbols: List[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (instances, unique_base_symbols).

    Supports legacy configs with only `symbols`, and new configs with:
        instances:
          - id: TSLA-long
            symbol: TSLA
            overrides: { shorting: none, fast: 55 }
          - id: TSLA-short
            symbol: TSLA
            overrides: { shorting: short_only, fast: 21 }
    """
    if cfg.get(K.KEY_INSTANCES):
        instances: List[Dict[str, Any]] = []
        for ins in cfg[K.KEY_INSTANCES]:
            instances.append(
                {
                    K.KEY_INSTANCE_ID: str(ins[K.KEY_INSTANCE_ID]),
                    K.KEY_INSTANCE_SYMBOL: str(ins[K.KEY_INSTANCE_SYMBOL]),
                    K.KEY_INSTANCE_OVERRIDES: dict(ins.get(K.KEY_INSTANCE_OVERRIDES, {}) or {}),
                    K.KEY_INSTANCE_WEBHOOK: ins.get(K.KEY_INSTANCE_WEBHOOK),
                    K.KEY_INSTANCE_SIZING: ins.get(K.KEY_INSTANCE_SIZING),
                }
            )
    else:
        instances = [
            {K.KEY_INSTANCE_ID: s, K.KEY_INSTANCE_SYMBOL: s, K.KEY_INSTANCE_OVERRIDES: {}, K.KEY_INSTANCE_WEBHOOK: None, K.KEY_INSTANCE_SIZING: None}
            for s in symbols
        ]

    base_symbols = sorted({it[K.KEY_INSTANCE_SYMBOL] for it in instances})
    return instances, base_symbols


def merge_params_for_instances(
    instances: List[Dict[str, Any]], params_cfg: dict
) -> Dict[str, dict]:
    """Merge params: defaults -> per_symbol[base] -> instance.overrides.

    Returns dict keyed by instance id.
    """
    dflt = params_cfg.get(K.KEY_PARAMS_DEFAULTS, {}) or {}
    per = params_cfg.get(K.KEY_PARAMS_PER_SYMBOL, {}) or {}
    out: Dict[str, dict] = {}
    for it in instances:
        base = it[K.KEY_INSTANCE_SYMBOL]
        pid = it[K.KEY_INSTANCE_ID]
        m = dict(dflt)
        m.update(dict(per.get(base, {}) or {}))
        m.update(dict(it.get(K.KEY_INSTANCE_OVERRIDES, {}) or {}))
        out[pid] = m
    return out


def resolve_instance_webhook(cfg: dict, inst: dict) -> dict:
    """Resolve webhook config for instance: base -> per-webhook -> inline overrides."""
    base = dict(cfg.get(K.KEY_WEBHOOK, {}) or {})
    wh_ref = inst.get(K.KEY_INSTANCE_WEBHOOK)
    if isinstance(wh_ref, str):  # named webhook
        named = (cfg.get(K.KEY_WEBHOOKS, {}) or {}).get(wh_ref, {})
        base.update(named)
    elif isinstance(wh_ref, dict):  # inline overrides
        base.update(wh_ref)
    return base


def resolve_instance_sizing(cfg: dict, inst: dict) -> dict:
    """Resolve sizing config for instance: base -> instance overrides."""
    base = dict(cfg.get(K.KEY_SIZING, {}) or {})
    if isinstance(inst.get(K.KEY_INSTANCE_SIZING), dict):
        base.update(inst[K.KEY_INSTANCE_SIZING])
    return base
