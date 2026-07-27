from __future__ import annotations

import pandas as pd

from src.liquidity_map.depth_scale import (
    CausalDepthScaleConfig,
    attach_causal_depth_scale,
)


def _frame(include_future: bool) -> pd.DataFrame:
    rows = []
    values = [
        (0, "bid", 100.0, 50.0),
        (0, "ask", 200.0, 100.0),
        (5, "bid", 80.0, 40.0),
        (5, "ask", 150.0, 75.0),
    ]
    if include_future:
        values.extend([(10, "bid", 1000.0, 500.0), (10, "ask", 800.0, 400.0)])
    for seconds, side, high, low in values:
        timestamp = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000) + seconds * 1000
        side_code = 1 if side == "bid" else -1
        rows.extend([
            {"bucket_start_ms": timestamp, "side_code": side_code, "depth": high},
            {"bucket_start_ms": timestamp, "side_code": side_code, "depth": low},
        ])
    return pd.DataFrame(rows)


def test_literal_max_mode_remains_available_for_diagnostics() -> None:
    scaled = attach_causal_depth_scale(
        _frame(include_future=False),
        depth_column="depth",
        config=CausalDepthScaleConfig(window_hours=24, snapshot_reference_quantile=1.0),
    )
    bid = scaled.loc[scaled["side_code"] == 1].reset_index(drop=True)
    ask = scaled.loc[scaled["side_code"] == -1].reset_index(drop=True)
    assert bid["causal_depth_reference"].tolist() == [200.0, 200.0, 200.0, 200.0]
    assert ask["causal_depth_reference"].tolist() == [200.0, 200.0, 200.0, 200.0]
    assert bid["causal_depth_ratio"].tolist() == [0.5, 0.25, 0.4, 0.2]
    assert ask["causal_depth_ratio"].tolist() == [1.0, 0.5, 0.75, 0.375]


def test_robust_snapshot_high_ignores_one_isolated_outlier() -> None:
    rows = []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for second, outlier in [(0, 1000.0), (5, 0.0)]:
        timestamp = int((base + pd.Timedelta(seconds=second)).timestamp() * 1000)
        for index in range(100):
            depth = 10.0
            if second == 0 and index == 99:
                depth = outlier
            rows.append({
                "bucket_start_ms": timestamp,
                "side_code": 1 if index < 50 else -1,
                "depth": depth,
            })
    frame = pd.DataFrame(rows)
    scaled = attach_causal_depth_scale(
        frame,
        depth_column="depth",
        config=CausalDepthScaleConfig(window_hours=24, snapshot_reference_quantile=0.99),
    )
    first_reference = float(scaled.loc[scaled["bucket_start_ms"] == scaled["bucket_start_ms"].min(), "causal_depth_reference"].iloc[0])
    assert first_reference < 50.0
    assert float(scaled["causal_depth_reference"].max()) < 50.0


def test_future_snapshot_cannot_recolor_earlier_rows() -> None:
    config = CausalDepthScaleConfig(window_hours=24, snapshot_reference_quantile=0.99)
    base = attach_causal_depth_scale(
        _frame(include_future=False),
        depth_column="depth",
        config=config,
    )
    extended = attach_causal_depth_scale(
        _frame(include_future=True),
        depth_column="depth",
        config=config,
    ).iloc[: len(base)]
    assert extended["causal_depth_reference"].tolist() == base["causal_depth_reference"].tolist()
    assert extended["causal_depth_ratio"].tolist() == base["causal_depth_ratio"].tolist()
