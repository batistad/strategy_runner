import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.models import PositionTracker, SizingCalculator, expand_action_to_deltas


class PositionTrackerTests(unittest.TestCase):
    def test_apply_delta_from_flat(self):
        tracker = PositionTracker(0)
        self.assertEqual(tracker.apply_delta(10), 10)

    def test_apply_delta_accumulates(self):
        tracker = PositionTracker(5)
        tracker.apply_delta(3)
        self.assertEqual(tracker.curr_pos, 8)
        tracker.apply_delta(-2)
        self.assertEqual(tracker.curr_pos, 6)

    def test_deltas_to_target_from_flat(self):
        tracker = PositionTracker(0)
        deltas = tracker.deltas_to_target(target=1, entry_qty=10)
        self.assertEqual(deltas, [10])

    def test_deltas_to_target_flip_long_to_short(self):
        tracker = PositionTracker(5)
        deltas = tracker.deltas_to_target(target=-1, entry_qty=10)
        self.assertEqual(deltas, [-5, -10])


class SizingCalculatorTests(unittest.TestCase):
    def test_fixed_mode(self):
        sizing = {"mode": "fixed", "fixed_qty": 20, "lot_size": 1}
        calc = SizingCalculator(sizing)
        qty = calc.compute_entry_qty(price=100)
        self.assertEqual(qty, 20)

    def test_notional_mode(self):
        sizing = {"mode": "notional", "notional": 1000}
        calc = SizingCalculator(sizing)
        qty = calc.compute_entry_qty(price=100)
        self.assertEqual(qty, 10)

    def test_lot_rounding(self):
        sizing = {"mode": "notional", "notional": 1050, "lot_size": 10}
        calc = SizingCalculator(sizing)
        qty = calc.compute_entry_qty(price=100)
        self.assertEqual(qty, 10)  # 10.5 rounds down to 10


class ExpandActionToDeltasTests(unittest.TestCase):
    def test_buy_from_flat(self):
        deltas = expand_action_to_deltas("BUY", entry_qty=5, curr_pos=0)
        self.assertEqual(deltas, [5])

    def test_sell_flip_from_long(self):
        deltas = expand_action_to_deltas("SELL", entry_qty=3, curr_pos=5, signal_curr=0)
        self.assertEqual(deltas, [-5])

    def test_flip_to_short(self):
        deltas = expand_action_to_deltas("FLIP_TO_SHORT", entry_qty=10, curr_pos=5)
        self.assertEqual(deltas, [-5, -10])


if __name__ == "__main__":
    unittest.main()
