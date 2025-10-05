from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

import pandas as pd


class DataProvider(Protocol):
    def fetch_eod(
        self, symbols: Iterable[str], history_days: int, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        # TODO
        return


@dataclass
class Bar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
