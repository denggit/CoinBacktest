#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.architecture.import_boundaries import (
    DEFAULT_ALLOWLIST,
    scan_import_boundaries,
    unexpected_violations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check research/backtest import boundaries.")
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument("--write-current-allowlist", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.write_current_allowlist:
        rows = [item.to_dict() for item in scan_import_boundaries(REPO_ROOT)]
        target = Path(args.allowlist)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "note": "Known legacy imports. New research/backtest coupling should not be added.",
                    "allowed_legacy_imports": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"allowlist written | path={target} count={len(rows)}")
        return 0

    unexpected = unexpected_violations(REPO_ROOT, args.allowlist)
    if unexpected:
        print("Unexpected import-boundary violations:")
        for item in unexpected:
            print(f"- {item.file}: {item.module} ({item.reason})")
        return 1
    print("Import boundaries ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
