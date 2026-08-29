from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy_research.eth_source_locked_portfolio import SourceLockedConfig, run_source_locked


def main() -> int:
    p = argparse.ArgumentParser(description="R03 Source-Locked ETH Trend Replication")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01 00:00:00")
    p.add_argument("--start-date", default="2023-01-01 00:00:00")
    p.add_argument("--end-date", default="2025-12-31 23:59:59")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    cfg = SourceLockedConfig(symbol=args.symbol, warmup_start=args.warmup_start, research_start=args.start_date, research_end=args.end_date)
    run_source_locked(cfg, progress=not args.no_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
