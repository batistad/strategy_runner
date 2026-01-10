"""Position tracking and position sizing calculations."""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import List

log = logging.getLogger(__name__)


class PositionTracker:
    """Tracks current position and computes position deltas for multi-leg trades."""

    def __init__(self, curr_pos: int = 0):
        self.curr_pos = int(curr_pos)

    def apply_delta(self, delta: int) -> int:
        """Apply a signed delta to current position and return new position."""
        self.curr_pos = int(self.curr_pos + delta)
        return self.curr_pos

    def deltas_to_target(self, target: int, entry_qty: int) -> List[int]:
        """Compute deltas to move from curr_pos to desired signed position size.

        Interprets target ∈ {-1,0,1} as direction × entry_qty (absolute size).
        Represents flips as two legs for clarity when crossing zero.
        """
        desired = int(target) * int(entry_qty)
        delta = desired - self.curr_pos
        if delta == 0:
            return []
        # Represent flips as two legs for clarity when crossing zero
        if self.curr_pos > 0 and delta < 0 and desired < 0:
            return [-self.curr_pos, desired]
        if self.curr_pos < 0 and delta > 0 and desired > 0:
            return [abs(self.curr_pos), desired]
        return [delta]


class SizingCalculator:
    """Calculates entry quantity based on sizing config and market price."""

    def __init__(self, sizing: dict):
        self.sizing = dict(sizing or {})

    def compute_entry_qty(self, price: float) -> int:
        """Calculate entry quantity based on mode and config."""
        mode = str(self.sizing.get("mode", "notional")).lower()
        lot_size = int(self.sizing.get("lot_size", 1))
        min_qty = int(self.sizing.get("min_qty", 1))

        if price <= 0:
            return 0

        if mode == "fixed":
            base_qty = int(self.sizing.get("fixed_qty", 1))
            return self._round_qty(Decimal(base_qty), lot_size, min_qty)

        if mode in ("percent_of_cash", "percent"):
            pct = float(self.sizing.get("percent", 0.0))
            cash_env = self.sizing.get("cash_env")
            cash = float(os.getenv(cash_env, "0") or 0) if cash_env else 0.0
            notional = max(0.0, cash * pct)
        else:
            notional = float(self.sizing.get("notional", 0.0))

        raw_qty = Decimal(notional) / Decimal(price) if price > 0 else Decimal(0)
        return self._round_qty(raw_qty, lot_size, min_qty)

    @staticmethod
    def _round_qty(qty: Decimal, lot_size: int, min_qty: int) -> int:
        """Round quantity to lot multiple and enforce minimum."""
        if lot_size <= 0:
            lot_size = 1
        lots = (qty // Decimal(lot_size)) * Decimal(lot_size)
        out = int(lots)
        return max(out, int(min_qty)) if out > 0 else 0


def expand_action_to_deltas(
    action: str,
    entry_qty: int,
    curr_pos: int,
    signal_curr: int | None = None,
) -> List[int]:
    """Return signed deltas (buy>0, sell<0) for an action.

    Handles both simple and verbose action names.
    """
    if action in ("HOLD", None):
        return []

    # Simple mode canonical names
    if action == "BUY":  # enter long
        if curr_pos > 0:
            return []
        if curr_pos < 0:
            return [abs(curr_pos), max(entry_qty, 0)]
        return [max(entry_qty, 0)]

    if action == "SELL":  # enter short in simple mode or exit long
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
        # Fallback heuristic
        if curr_pos > 0:
            return [-max(curr_pos, 0)]
        return [-max(entry_qty, 0)]

    if action in ("CLOSELONG",):
        return [-max(curr_pos, 0)] if curr_pos > 0 else []

    if action in ("CLOSESHORT",):
        return [abs(min(curr_pos, 0))] if curr_pos < 0 else []

    # Verbose compatibility
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
