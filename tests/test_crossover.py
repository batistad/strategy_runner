import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.strategies.crossover import StrategyParams, run_strategy


def _df_from_closes(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": 1_000_000,
        }
    )


class CrossoverTests(unittest.TestCase):
    def test_long_buy_action_on_cross(self):
        df = _df_from_closes([1, 2, 3, 4])
        params = {"FOO": StrategyParams(ma_type="SMA", fast=2, slow=3, shorting="none")}
        trades, metrics, new_signals = run_strategy({"FOO": df}, params, last_signals={})
        self.assertEqual(new_signals["FOO"], 1)
        self.assertEqual(trades[0]["action"], "BUY")
        self.assertEqual(trades[0]["signal_prev"], 0)
        self.assertEqual(trades[0]["signal_curr"], 1)

    def test_short_only_emits_short(self):
        df = _df_from_closes([4, 3, 2, 1])
        params = {"FOO": StrategyParams(ma_type="SMA", fast=2, slow=3, shorting="short_only")}
        trades, metrics, new_signals = run_strategy({"FOO": df}, params, last_signals={})
        self.assertEqual(new_signals["FOO"], -1)
        self.assertEqual(trades[0]["action"], "SHORT")

    def test_target_mode_returns_target_entry(self):
        df = _df_from_closes([1, 2, 3, 4])
        params = {
            "FOO": StrategyParams(
                ma_type="SMA", fast=2, slow=3, shorting="none", actions_mode="target"
            )
        }
        trades, metrics, new_signals = run_strategy({"FOO": df}, params, last_signals={})
        self.assertEqual(trades[0]["type"], "TARGET")
        self.assertEqual(trades[0]["target"], 1)

    def test_chandelier_stop_forces_exit_to_flat(self):
        df = _df_from_closes([10, 11, 12, 13, 14, 13])
        params = {
            "FOO": StrategyParams(
                ma_type="EMA",
                fast=2,
                slow=4,
                shorting="none",
                chandelier_len=3,
                chandelier_mult=0,
            )
        }
        trades, metrics, new_signals = run_strategy({"FOO": df}, params, last_signals={"FOO": 1})
        self.assertEqual(new_signals["FOO"], 0)
        self.assertEqual(trades[0]["action"], "SELL")
        self.assertEqual(trades[0]["signal_prev"], 1)
        self.assertEqual(trades[0]["signal_curr"], 0)


if __name__ == "__main__":
    unittest.main()
