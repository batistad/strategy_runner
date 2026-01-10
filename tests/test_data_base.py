import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.data.base import normalize_ohlcv


class NormalizeOhlcvTests(unittest.TestCase):
    def test_normalize_ohlcv_renames_and_sorts(self):
        df = pd.DataFrame(
            {
                "datetime": ["2024-01-02", "2024-01-01", "2024-01-01"],
                "open": [2, 1, 1],
                "high": [3, 2, 2],
                "low": [1, 1, 1],
                "close": [2, 1, 1],
                "volume": [100, 200, 200],
            }
        )

        out = normalize_ohlcv(df)

        self.assertEqual(list(out.columns)[:5], ["date", "Open", "High", "Low", "Close"])
        self.assertTrue(out.loc[0, "date"].isoformat().startswith("2024-01-01"))
        self.assertTrue(out.loc[1, "date"].isoformat().startswith("2024-01-02"))
        # duplicate date keeps last instance
        self.assertEqual(out.loc[0, "Volume"], 200)

    def test_normalize_ohlcv_missing_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with self.assertRaises(ValueError):
            normalize_ohlcv(df)


if __name__ == "__main__":
    unittest.main()
