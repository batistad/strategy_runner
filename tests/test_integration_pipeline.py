"""Integration tests for the full daily pipeline.

Tests end-to-end pipeline: config loading → strategy execution → state management.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner import config as cfg_module
from strategyrunner.pipelines import daily


class IntegrationPipelineTests(unittest.TestCase):
    """Integration tests for config → strategy → state pipeline."""

    def setUp(self):
        """Create temporary directories and sample config."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

        # Create minimal test config with crossover strategy
        self.config_crossover = {
            "market": "XNAS",
            "symbols": ["AAPL"],
            "params": {
                "name": "crossover",
                "defaults": {
                    "ma_type": "EMA",
                    "fast": 50,
                    "slow": 100,
                    "shorting": "none",
                    "actions_mode": "verbose",
                },
            },
            "data": {
                "provider": "yahoo",
                "interval": "1d",
                "history_days": 100,
            },
            "webhook": {
                "enabled": False,  # Disable for testing
            },
            "state": {
                "path": str(self.config_dir / "state.json"),
                "dry_path": str(self.config_dir / "state.dry.json"),
            },
        }

        # Create minimal test config with momentum strategy
        self.config_momentum = dict(self.config_crossover)
        self.config_momentum["params"] = {
            "name": "momentum",
            "min_volume": 100000,
        }

        self.config_path_crossover = self.config_dir / "config_crossover.yaml"
        self.config_path_momentum = self.config_dir / "config_momentum.yaml"

        with open(self.config_path_crossover, "w") as f:
            yaml.dump(self.config_crossover, f)
        with open(self.config_path_momentum, "w") as f:
            yaml.dump(self.config_momentum, f)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_config_loads_with_yaml(self):
        """Test that config.load_config correctly loads YAML."""
        cfg = cfg_module.load_config(str(self.config_path_crossover))
        self.assertIn("market", cfg)
        self.assertIn("symbols", cfg)
        self.assertEqual(cfg["market"], "XNAS")

    def test_expand_instances_legacy_symbols(self):
        """Test that expand_instances creates instances from symbol list."""
        cfg = cfg_module.load_config(str(self.config_path_crossover))
        instances, base_symbols = cfg_module.expand_instances(cfg, cfg["symbols"])

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["symbol"], "AAPL")
        self.assertEqual(base_symbols, ["AAPL"])

    def test_merge_params_cascade(self):
        """Test that merge_params_for_instances applies cascade: defaults → per_symbol → overrides."""
        cfg = cfg_module.load_config(str(self.config_path_crossover))
        instances, _ = cfg_module.expand_instances(cfg, cfg["symbols"])

        params_cfg = cfg["params"]
        merged = cfg_module.merge_params_for_instances(instances, params_cfg)

        # Should have AAPL instance
        self.assertIn("AAPL", merged)
        self.assertEqual(merged["AAPL"]["fast"], 50)
        self.assertEqual(merged["AAPL"]["slow"], 100)

    def test_state_persistence_roundtrip(self):
        """Test that state can be saved and loaded."""
        from strategyrunner.state import load_state, save_state

        state_path = self.config_dir / "test_state.json"
        state = {
            "last_signals": {"AAPL": 1},
            "positions": {"AAPL": 100},
            "last_asof": "2025-01-10",
            "last_metrics": {"example": "data"},
        }

        save_state(str(state_path), state)
        loaded = load_state(str(state_path))

        self.assertEqual(loaded["last_signals"], {"AAPL": 1})
        self.assertEqual(loaded["positions"], {"AAPL": 100})

    def test_config_with_instances(self):
        """Test config with explicit instances (per-symbol trading)."""
        config_with_instances = {
            "market": "XNAS",
            "instances": [
                {
                    "id": "AAPL-long",
                    "symbol": "AAPL",
                    "overrides": {"shorting": "none"},
                },
                {
                    "id": "AAPL-short",
                    "symbol": "AAPL",
                    "overrides": {"shorting": "short_only"},
                },
            ],
            "params": {
                "name": "crossover",
                "defaults": {
                    "ma_type": "EMA",
                    "fast": 50,
                    "slow": 100,
                },
            },
        }

        config_path = self.config_dir / "config_instances.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_instances, f)

        cfg = cfg_module.load_config(str(config_path))
        instances, base_symbols = cfg_module.expand_instances(cfg, [])

        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]["id"], "AAPL-long")
        self.assertEqual(instances[1]["id"], "AAPL-short")
        self.assertEqual(base_symbols, ["AAPL"])

    def test_webhook_config_resolution(self):
        """Test that webhook config cascades: global → per-webhook → inline."""
        config_with_webhooks = dict(self.config_crossover)
        config_with_webhooks.update({
            "webhook": {
                "enabled": True,
                "url_env": "WEBHOOK_URL",
                "timeout": 5,
            },
            "webhooks": {
                "slack": {
                    "url_env": "SLACK_URL",
                    "message_template": "Slack: {action}",
                },
            },
            "instances": [
                {
                    "id": "AAPL",
                    "symbol": "AAPL",
                    "webhook": "slack",  # Reference named webhook
                },
            ],
        })

        config_path = self.config_dir / "config_webhooks.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_webhooks, f)

        cfg = cfg_module.load_config(str(config_path))
        instances, _ = cfg_module.expand_instances(cfg, [])

        # Resolve webhook for first instance (should get 'slack' config)
        inst_wh = cfg_module.resolve_instance_webhook(cfg, instances[0])
        self.assertIn("message_template", inst_wh)
        self.assertEqual(inst_wh["message_template"], "Slack: {action}")

    def test_sizing_config_resolution(self):
        """Test that sizing config resolves correctly."""
        config_with_sizing = dict(self.config_crossover)
        config_with_sizing.update({
            "sizing": {
                "mode": "notional",
                "notional": 10000,
            },
            "instances": [
                {
                    "id": "AAPL",
                    "symbol": "AAPL",
                    "sizing": {
                        "mode": "fixed",
                        "fixed_qty": 10,
                    },
                },
            ],
        })

        config_path = self.config_dir / "config_sizing.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_sizing, f)

        cfg = cfg_module.load_config(str(config_path))
        instances, _ = cfg_module.expand_instances(cfg, [])

        # Instance override should win
        sizing = cfg_module.resolve_instance_sizing(cfg, instances[0])
        self.assertEqual(sizing["mode"], "fixed")
        self.assertEqual(sizing["fixed_qty"], 10)

    @patch("strategyrunner.data.yahoo.fetch_eod")
    def test_dry_run_crossover_strategy(self, mock_fetch):
        """Test dry-run execution of crossover strategy."""
        # Mock data: simple uptrend
        mock_data = {
            "AAPL": pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=100, freq="D"),
                "Open": range(100, 200),
                "High": range(101, 201),
                "Low": range(99, 199),
                "Close": range(100, 200),
                "Volume": [1000000] * 100,
            })
        }
        mock_fetch.return_value = mock_data

        cfg = cfg_module.load_config(str(self.config_path_crossover))

        # Run with dry=True, asof date to avoid time-based waits
        result = daily.run_daily(
            str(self.config_path_crossover),
            asof="2025-04-10",  # Date in mock data
            dry=True,
        )

        self.assertTrue(result)

        # Check state was created (even in dry mode)
        state_path = self.config_dir / "state.dry.json"
        self.assertTrue(state_path.exists())

    @patch("strategyrunner.data.yahoo.fetch_eod")
    def test_dry_run_momentum_strategy(self, mock_fetch):
        """Test dry-run execution of momentum strategy."""
        # Mock data
        mock_data = {
            "AAPL": pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=100, freq="D"),
                "Open": range(100, 200),
                "High": range(101, 201),
                "Low": range(99, 199),
                "Close": range(100, 200),
                "Volume": [5000000] * 100,  # High volume
            })
        }
        mock_fetch.return_value = mock_data

        cfg = cfg_module.load_config(str(self.config_path_momentum))

        result = daily.run_daily(
            str(self.config_path_momentum),
            asof="2025-04-10",
            dry=True,
        )

        self.assertTrue(result)

    def test_state_merge_signals_without_clobbering(self):
        """Test that state merge doesn't overwrite untouched instances."""
        from strategyrunner.state import merge_signals

        prev_signals = {"AAPL": 1, "TSLA": -1}
        committed_signals = {"AAPL": 0}  # Only AAPL changed

        merged = merge_signals(prev_signals, committed_signals)

        # AAPL should be updated, TSLA should stay
        self.assertEqual(merged["AAPL"], 0)
        self.assertEqual(merged["TSLA"], -1)


if __name__ == "__main__":
    unittest.main()
