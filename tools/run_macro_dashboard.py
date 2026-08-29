#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_monitor.config import MonitorConfig, load_dotenv
from macro_monitor.dashboard import DashboardDataService, MacroDashboardServer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local real-time macro monitor dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port. Default: 8765")
    parser.add_argument("--db", type=Path, help="SQLite path. Defaults to MACRO_MONITOR_DB/project data path.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the dashboard in the default browser.")
    parser.add_argument("--verbose", action="store_true", help="Log HTTP requests and diagnostics.")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("macro_dashboard")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    load_dotenv(ROOT / ".env")
    config = MonitorConfig.from_env(ROOT)
    db_path = args.db if args.db is not None else config.db_path
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    logger = configure_logging(args.verbose)
    service = DashboardDataService(db_path, config.thresholds)
    server = MacroDashboardServer(service, logger, host=args.host, port=args.port, verbose=args.verbose)
    server.start()
    logger.info("[dashboard] backend collector remains independent")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()
    try:
        server.wait()
    except KeyboardInterrupt:
        logger.info("[dashboard] stopping UI server; collector is unaffected")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
