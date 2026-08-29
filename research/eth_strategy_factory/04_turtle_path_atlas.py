from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy_research.eth_turtle_path_atlas import TurtlePathConfig, run_turtle_path_atlas


def main() -> int:
    p = argparse.ArgumentParser(description="R04 Turtle minute-path atlas for position-management research")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01 00:00:00")
    p.add_argument("--start-date", default="2023-01-01 00:00:00")
    p.add_argument("--end-date", default="2025-12-31 23:59:59")
    args = p.parse_args()
    cfg = TurtlePathConfig(symbol=args.symbol, warmup_start=args.warmup_start, research_start=args.start_date, research_end=args.end_date)
    run_turtle_path_atlas(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
