#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small dependency-light sanity test for analyze_tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tool.data_service import dataframe_to_candles  # noqa: E402
from analyze_tool.plugins import build_default_registry  # noqa: E402


def main() -> int:
    idx = pd.to_datetime([
        "2026-01-01 00:00:00",
        "2026-01-01 00:01:00",
        "2026-01-01 00:02:00",
    ])
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 99.0],
            "high": [101.0, 111.0, 100.0],
            "low": [99.5, 100.8, 90.0],
            "close": [100.6, 101.2, 98.8],
            "volume": [10, 20, 30],
            "delta_volume": [1, -2, 3],
        },
        index=idx,
    )
    payload = dataframe_to_candles(df, {"loader": "selftest"})
    assert len(payload["candles"]) == 3
    assert payload["candles"][0]["extra"]["delta_volume"] == 1

    registry = build_default_registry()
    plugin = registry.get("long_shadow")
    result = plugin.run(df, {"direction": "both", "min_wick_body_ratio": 2.0, "min_wick_pct": 0.001, "color": "#facc15"})
    assert len(result.markers) >= 2, result.summary
    print("[selftest] OK", payload["candles"][0]["timestamp"], result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
