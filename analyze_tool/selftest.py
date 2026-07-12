#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependency-light sanity test for analyze_tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tool.data_service import dataframe_to_candles  # noqa: E402
from analyze_tool.plugins import build_default_registry  # noqa: E402
from analyze_tool.server import _json_safe  # noqa: E402


def build_panic_sample() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=120, freq="1min")
    close = np.full(120, 100.0)
    for i in range(1, 70):
        close[i] = close[i - 1] * (1 + (0.0001 if i % 2 else -0.00008))
    path = [99.90, 99.75, 99.58, 99.40, 99.20, 98.95, 98.72, 98.66, 98.70, 98.86, 99.05, 99.24, 99.42, 99.55, 99.70]
    close[70 : 70 + len(path)] = path
    close[85:] = np.linspace(99.72, 100.4, 35)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0004
    low = np.minimum(open_, close) * 0.9996
    low[76] = 98.48
    volume = np.full(120, 100.0)
    volume[70:77] = [150, 170, 190, 220, 260, 320, 400]
    volume[77:85] = [300, 240, 180, 160, 150, 140, 130, 120]
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "delta_volume": np.r_[np.zeros(70), -np.linspace(20, 70, 7), np.linspace(-20, 20, 43)],
        },
        index=idx,
    )


def main() -> int:
    basic = build_panic_sample().iloc[:3].copy()
    payload = dataframe_to_candles(basic, {"loader": "selftest"})
    assert len(payload["candles"]) == 3
    assert "delta_volume" in payload["candles"][0]["extra"]

    registry = build_default_registry()
    assert {"long_shadow", "panic_selloff_recovery"}.issubset({p["id"] for p in registry.list_plugins()})

    panic = registry.get("panic_selloff_recovery").run(build_panic_sample(), {})
    assert panic.summary["signal_count"] == 1, panic.summary
    assert len(panic.regions) == 1
    labels = [marker.label for marker in panic.markers]
    for expected in ["开始观察", "卖压加速", "卖压衰减 / 拒绝", "恢复确认 · 做多观察"]:
        assert expected in labels, labels
    signal = [marker for marker in panic.markers if marker.role == "signal"]
    assert len(signal) == 1
    assert signal[0].fields["signal_is_causal"] is True
    assert signal[0].fields["outcome_return_15b"] > 0

    strict_payload = panic.as_dict()
    strict_payload["summary"]["nan_probe"] = np.nan
    strict_json = json.dumps(_json_safe(strict_payload), ensure_ascii=False, allow_nan=False)
    assert "NaN" not in strict_json and "Infinity" not in strict_json
    json.loads(strict_json)

    print("[selftest] OK", panic.summary["display"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
