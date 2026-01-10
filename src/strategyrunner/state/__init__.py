"""State management and persistence."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

log = logging.getLogger(__name__)


def load_state(path: str) -> dict:
    """Load state JSON from disk. Returns empty dict if file does not exist."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load state from %s: %s", path, e)
            return {}
    return {}


def save_state(path: str, state: dict) -> None:
    """Save state dict to disk as JSON."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log.debug("State saved to %s", path)
    except Exception as e:
        log.error("Failed to save state to %s: %s", path, e)
        raise


def merge_signals(prev: Dict[str, int], committed: Dict[str, int]) -> Dict[str, int]:
    """Merge committed signals into previous state without clobbering untouched instances."""
    out = dict(prev)
    out.update(committed)
    return out
