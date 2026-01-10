import math
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.strategies.momentum import StrategyParams, compute_signal, run_strategy


class MomentumTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=130, freq="B")
        self.df = pd.DataFrame(
            {
                "date": dates,
                "open": 100,
                "high": 101,
                "low": 99,
                "close": pd.Series(range(len(dates))) + 100,
                "volume": 2_000_000,
            }
        )

    def test_compute_signal_requires_enough_history(self):
        short_df = self.df.head(10)
        sig = compute_signal(short_df, lookback=20, min_volume=1)
        self.assertTrue(math.isnan(sig))

    def test_compute_signal_filters_low_volume(self):
        low_vol = self.df.copy()
        low_vol["volume"] = 10
        sig = compute_signal(low_vol, lookback=20, min_volume=1_000_000)
        self.assertTrue(math.isnan(sig))

    def test_run_strategy_returns_trades(self):
        trades, metrics = run_strategy({"FOO": self.df}, StrategyParams())
        self.assertTrue(trades)
        self.assertIsInstance(metrics, dict)


if __name__ == "__main__":
    unittest.main()
