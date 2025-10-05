from strategyrunner.strategies.momentum import StrategyParams, run_strategy


def test_run_strategy_smoke():
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 150, len(dates)),
        "high": np.linspace(100, 150, len(dates)) * 1.01,
        "low": np.linspace(100, 150, len(dates)) * 0.99,
        "close": np.linspace(100, 160, len(dates)),
        "volume": 2_000_000,
    })
    data = {"FOO": df}
    trades, metrics = run_strategy(data, StrategyParams())
    assert trades and isinstance(metrics, dict)