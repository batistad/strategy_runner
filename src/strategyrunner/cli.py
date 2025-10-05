from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from .pipelines.daily import run_daily


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strategyrunner", description="Run strategies daily"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Execute daily pipeline")
    run.add_argument("--config", default="config.yaml", help="Path to YAML config")
    run.add_argument("--asof", help="YYYY-MM-DD override/backfill")
    run.add_argument("--dry", action="store_true", help="Dry run (no webhook)")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # loads .env if present
    args = _build_parser().parse_args(argv)

    if args.cmd == "run":
        return (
            0 if run_daily(config_path=args.config, asof=args.asof, dry=args.dry) else 1
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
