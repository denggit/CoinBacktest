#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_monitor.config import MonitorConfig, Thresholds, load_dotenv
from macro_monitor.monitor import MacroMonitor


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor CME FedWatch, US Treasury yields, and DXY repricing.")
    parser.add_argument("--once", action="store_true", help="Fetch one snapshot, print it, and exit.")
    parser.add_argument("--fedwatch-poll-seconds", type=_positive, help="FedWatch polling interval (default/env: 60).")
    parser.add_argument("--treasury-poll-seconds", type=_positive, help="Treasury polling interval (default/env: 15).")
    parser.add_argument("--dxy-poll-seconds", type=_positive, help="DXY polling interval (default/env: 15).")
    parser.add_argument(
        "--off-hours-poll-seconds",
        type=_positive,
        help="Polling interval outside New York weekdays 07:00-19:00 (default/env: 300).",
    )
    parser.add_argument(
        "--weekend-poll-seconds",
        type=_positive,
        help="Polling interval on New York Saturdays and Sundays (default/env: 3600).",
    )
    parser.add_argument("--headed", action="store_true", help="Show the reusable Chromium window.")
    parser.add_argument("--email", action="store_true", help="Enable SMTP alerts in addition to EMAIL_ENABLED.")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logging without dumping page contents.")
    parser.add_argument("--db", type=Path, help="SQLite path (default: data/macro_monitor/macro_monitor.sqlite).")
    parser.add_argument("--fedwatch-15m-alert-pct", type=_positive)
    parser.add_argument("--fedwatch-60m-alert-pct", type=_positive)
    parser.add_argument("--us2y-5m-alert-bp", type=_positive)
    parser.add_argument("--us2y-15m-alert-bp", type=_positive)
    parser.add_argument("--us2y-60m-alert-bp", type=_positive)
    parser.add_argument("--us10y-5m-alert-bp", type=_positive)
    parser.add_argument("--us10y-15m-alert-bp", type=_positive)
    parser.add_argument("--us10y-60m-alert-bp", type=_positive)
    parser.add_argument("--dxy-5m-alert-pct", type=_positive)
    parser.add_argument("--dxy-15m-alert-pct", type=_positive)
    parser.add_argument("--dxy-60m-alert-pct", type=_positive)
    parser.add_argument("--curve-15m-alert-bp", type=_positive)
    parser.add_argument("--curve-60m-alert-bp", type=_positive)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> MonitorConfig:
    load_dotenv(ROOT / ".env")
    config = MonitorConfig.from_env(ROOT, headed=args.headed, verbose=args.verbose, email=args.email)
    changes: dict[str, object] = {}
    if args.db is not None:
        changes["db_path"] = args.db if args.db.is_absolute() else ROOT / args.db
    if args.fedwatch_poll_seconds is not None:
        changes["fedwatch_poll_seconds"] = args.fedwatch_poll_seconds
    if args.treasury_poll_seconds is not None:
        changes["treasury_poll_seconds"] = args.treasury_poll_seconds
    if args.dxy_poll_seconds is not None:
        changes["dxy_poll_seconds"] = args.dxy_poll_seconds
    if args.off_hours_poll_seconds is not None:
        changes["off_hours_poll_seconds"] = args.off_hours_poll_seconds
    if args.weekend_poll_seconds is not None:
        changes["weekend_poll_seconds"] = args.weekend_poll_seconds
    threshold_changes = {
        name: getattr(args, cli_name)
        for name, cli_name in (
            ("fedwatch_15m_pct", "fedwatch_15m_alert_pct"),
            ("fedwatch_60m_pct", "fedwatch_60m_alert_pct"),
            ("us2y_5m_bp", "us2y_5m_alert_bp"),
            ("us2y_15m_bp", "us2y_15m_alert_bp"),
            ("us2y_60m_bp", "us2y_60m_alert_bp"),
            ("us10y_5m_bp", "us10y_5m_alert_bp"),
            ("us10y_15m_bp", "us10y_15m_alert_bp"),
            ("us10y_60m_bp", "us10y_60m_alert_bp"),
            ("dxy_5m_pct", "dxy_5m_alert_pct"),
            ("dxy_15m_pct", "dxy_15m_alert_pct"),
            ("dxy_60m_pct", "dxy_60m_alert_pct"),
            ("curve_15m_bp", "curve_15m_alert_bp"),
            ("curve_60m_bp", "curve_60m_alert_bp"),
        )
        if getattr(args, cli_name) is not None
    }
    if threshold_changes:
        changes["thresholds"] = replace(config.thresholds, **threshold_changes)
    return config.with_overrides(**changes)


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("macro_monitor")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    logger = configure_logging(args.verbose)
    monitor = MacroMonitor(config, logger)
    if args.once:
        return await monitor.run_once()
    await monitor.run_forever()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"[macro] fatal: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
