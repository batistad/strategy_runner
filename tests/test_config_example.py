import sys
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.example.yaml"


class ConfigExampleTests(unittest.TestCase):
    def test_config_example_loads_and_has_core_keys(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIsInstance(cfg, dict)
        for key in ("market", "symbols", "params", "data", "webhook", "state"):
            self.assertIn(key, cfg)
        self.assertIsInstance(cfg.get("symbols"), list)
        self.assertIsInstance(cfg.get("params"), dict)
        self.assertIsInstance(cfg.get("data"), dict)
        self.assertIsInstance(cfg.get("state"), dict)


if __name__ == "__main__":
    unittest.main()
