#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy_research.eth_tournament.config import TournamentConfig  # noqa: E402
from src.strategy_research.eth_tournament.runner import run_tournament  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the frozen ETH external-strategy tournament (2023-2025; 2026 sealed).")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01 00:00:00")
    p.add_argument("--start-date", default="2023-01-01 00:00:00")
    p.add_argument("--end-date", default="2025-12-31 23:59:59")
    p.add_argument("--report-root", default="data/reports/research/eth_strategy_factory/v1")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    cfg = TournamentConfig(
        symbol=a.symbol,
        warmup_start=a.warmup_start,
        research_start=a.start_date,
        research_end=a.end_date,
        report_root=Path(a.report_root),
    )
    run_tournament(cfg, progress=not a.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
