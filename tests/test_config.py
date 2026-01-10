import sys
from pathlib import Path
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strategyrunner.config import (
    expand_instances,
    merge_params_for_instances,
    resolve_instance_sizing,
    resolve_instance_webhook,
)


class ConfigTests(unittest.TestCase):
    def test_expand_instances_legacy_symbols(self):
        cfg = {}
        symbols = ["AAPL", "GOOG"]
        instances, base = expand_instances(cfg, symbols)
        self.assertEqual(len(instances), 2)
        self.assertEqual(base, ["AAPL", "GOOG"])
        self.assertEqual(instances[0]["id"], "AAPL")

    def test_expand_instances_new_config(self):
        cfg = {
            "instances": [
                {"id": "AAPL-long", "symbol": "AAPL", "overrides": {"shorting": "none"}},
                {"id": "AAPL-short", "symbol": "AAPL", "overrides": {"shorting": "short_only"}},
            ]
        }
        instances, base = expand_instances(cfg, [])
        self.assertEqual(len(instances), 2)
        self.assertEqual(base, ["AAPL"])

    def test_merge_params_cascades(self):
        instances = [
            {"id": "AAPL", "symbol": "AAPL", "overrides": {"fast": 10}}
        ]
        params_cfg = {
            "defaults": {"ma_type": "SMA", "fast": 50},
            "per_symbol": {"AAPL": {"fast": 30}},
        }
        merged = merge_params_for_instances(instances, params_cfg)
        self.assertEqual(merged["AAPL"]["ma_type"], "SMA")
        self.assertEqual(merged["AAPL"]["fast"], 10)  # override wins

    def test_resolve_instance_webhook(self):
        cfg = {"webhook": {"url_env": "WEBHOOK_URL"}}
        inst = {"webhook": {"timeout": 20}}
        result = resolve_instance_webhook(cfg, inst)
        self.assertEqual(result["url_env"], "WEBHOOK_URL")
        self.assertEqual(result["timeout"], 20)


if __name__ == "__main__":
    unittest.main()
