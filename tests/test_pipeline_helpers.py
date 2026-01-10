import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.models import expand_action_to_deltas
from strategyrunner.pipelines.daily import (
    _render_message,
    _summarize_trades,
)


class PipelineHelperTests(unittest.TestCase):
    def test_expand_action_buy_from_flat(self):
        deltas = expand_action_to_deltas("BUY", entry_qty=10, curr_pos=0, signal_curr=1)
        self.assertEqual(deltas, [10])

    def test_expand_action_sell_exit_long_when_signal_zero(self):
        deltas = expand_action_to_deltas("SELL", entry_qty=3, curr_pos=5, signal_curr=0)
        self.assertEqual(deltas, [-5])

    def test_render_message_leaves_missing_placeholder(self):
        msg = _render_message("Hi {foo} {bar}", {"foo": "X"})
        self.assertEqual(msg, "Hi X {bar}")

    def test_summarize_trades_empty(self):
        self.assertEqual(_summarize_trades([]), "no-trades")


if __name__ == "__main__":
    unittest.main()
