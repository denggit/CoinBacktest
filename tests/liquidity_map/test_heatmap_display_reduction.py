from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze_tool.plugins.orderbook_liquidity_heatmap import _normalize_depth, _reduce_cells


def _display_frame() -> pd.DataFrame:
    rows = []
    for t in range(10):
        for price in range(100):
            intensity = 0.95 if t == 0 else 0.005 + price / 100000.0
            rows.append(
                {
                    "bar_start_ms": t * 60_000,
                    "bucket_start_ms": t * 60_000,
                    "side": "bid",
                    "price_low": float(price),
                    "display_depth": 1.0 + price,
                    "intensity": intensity,
                }
            )
    return pd.DataFrame(rows)


def test_cell_budget_preserves_every_time_column() -> None:
    reduced, _ = _reduce_cells(_display_frame(), min_intensity=0.80, max_cells=100)
    assert reduced["bar_start_ms"].nunique() == 10
    assert len(reduced) <= 100


def test_log_depth_keeps_background_visible_and_caps_outliers() -> None:
    frame = pd.DataFrame({"display_depth": [1.0, 10.0, 100.0, 10_000.0], "local_ratio": [0.01, 0.1, 1.0, 1.0]})
    intensity, cap = _normalize_depth(frame, mode="log_depth", manual_max=100.0)
    assert cap > 0
    assert 0 < float(intensity.iloc[0]) < float(intensity.iloc[1]) < float(intensity.iloc[2]) <= 1.0
    assert np.isclose(float(intensity.iloc[-1]), 1.0)


def _causal_frame(rows: list[tuple[int, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bar_start_ms": timestamp_ms,
                "bucket_start_ms": timestamp_ms,
                "side": side,
                "distance_band": band,
                "display_depth": depth,
                "local_ratio": 1.0,
            }
            for timestamp_ms, side, band, depth in rows
        ]
    )


def test_causal_color_normalization_is_invariant_to_future_extension() -> None:
    hour = 3_600_000
    prefix = _causal_frame(
        [
            (0, "bid", "0-10", 10.0),
            (0, "bid", "0-10", 100.0),
            (hour, "bid", "0-10", 10.0),
            (hour, "bid", "0-10", 50.0),
        ]
    )
    extended = pd.concat(
        [
            prefix,
            _causal_frame(
                [
                    (2 * hour, "bid", "0-10", 10_000.0),
                    (2 * hour, "bid", "0-10", 100_000.0),
                ]
            ),
        ],
        ignore_index=True,
    )

    prefix_intensity, _, prefix_caps = _normalize_depth(
        prefix,
        mode="log_depth",
        manual_max=100.0,
        rolling_window_hours=24,
        rolling_percentile=0.99,
        return_caps=True,
    )
    extended_intensity, _, extended_caps = _normalize_depth(
        extended,
        mode="log_depth",
        manual_max=100.0,
        rolling_window_hours=24,
        rolling_percentile=0.99,
        return_caps=True,
    )

    np.testing.assert_allclose(
        prefix_intensity.to_numpy(),
        extended_intensity.iloc[: len(prefix)].to_numpy(),
    )
    np.testing.assert_allclose(
        prefix_caps.to_numpy(),
        extended_caps.iloc[: len(prefix)].to_numpy(),
    )


def test_causal_color_cap_expires_observations_outside_window() -> None:
    hour = 3_600_000
    frame = _causal_frame(
        [
            (0, "bid", "0-10", 1_000.0),
            (hour, "bid", "0-10", 10.0),
            (25 * hour + 60_000, "bid", "0-10", 10.0),
        ]
    )
    _, _, caps = _normalize_depth(
        frame,
        mode="auto_window",
        manual_max=100.0,
        rolling_window_hours=24,
        rolling_percentile=0.99,
        return_caps=True,
    )
    assert float(caps.iloc[1]) == 1_000.0
    assert float(caps.iloc[2]) == 10.0


def test_causal_color_cap_is_separate_by_side_and_distance_band() -> None:
    hour = 3_600_000
    frame = _causal_frame(
        [
            (0, "bid", "0-10", 10.0),
            (0, "bid", "200-500", 1_000.0),
            (0, "ask", "0-10", 100.0),
            (hour, "bid", "0-10", 20.0),
            (hour, "bid", "200-500", 2_000.0),
            (hour, "ask", "0-10", 200.0),
        ]
    )
    _, _, caps = _normalize_depth(
        frame,
        mode="auto_window",
        manual_max=100.0,
        rolling_window_hours=24,
        rolling_percentile=0.99,
        return_caps=True,
    )
    assert float(caps.iloc[3]) == 10.0
    assert float(caps.iloc[4]) == 1_000.0
    assert float(caps.iloc[5]) == 100.0


def test_cell_budget_keeps_deep_cells_and_pale_background_per_time_column() -> None:
    rows = []
    levels = np.linspace(0.02, 1.0, 100)
    for timestamp in range(20):
        for price, intensity in enumerate(levels):
            rows.append(
                {
                    "bar_start_ms": timestamp * 60_000,
                    "bucket_start_ms": timestamp * 60_000,
                    "side": "bid",
                    "price_low": float(price),
                    "display_depth": float(price + 1),
                    "intensity": float(intensity),
                }
            )
    frame = pd.DataFrame(rows)
    reduced, _ = _reduce_cells(frame, min_intensity=0.0, max_cells=200)
    assert len(reduced) <= 200
    assert reduced["bar_start_ms"].nunique() == 20
    grouped = reduced.groupby("bar_start_ms", observed=True)["intensity"]
    assert bool((grouped.max() > 0.99).all())
    assert bool((grouped.min() < 0.15).all())


def test_causal_log_display_contrast_restores_palette_separation() -> None:
    frame = pd.DataFrame(
        {
            "display_depth": [1.0, 10.0, 100.0, 500.0, 1_000.0],
            "local_ratio": [0.01, 0.05, 0.2, 0.7, 1.0],
        }
    )
    intensity, _ = _normalize_depth(
        frame,
        mode="log_depth",
        manual_max=1_000.0,
        display_contrast_gamma=2.2,
    )
    values = intensity.to_numpy(dtype=float)
    assert np.all(np.diff(values) > 0)
    assert values[0] < 0.05
    assert values[1] < 0.20
    assert 0.25 < values[2] < 0.60
    assert values[-1] == 1.0


def test_explicit_intensity_cutoff_still_keeps_each_time_column_alive() -> None:
    frame = _display_frame()
    frame.loc[frame["bar_start_ms"] != 0, "intensity"] = 0.10
    reduced, _ = _reduce_cells(frame, min_intensity=0.80, max_cells=100)
    assert reduced["bar_start_ms"].nunique() == 10
    assert len(reduced) <= 100


def test_causal_max_ratio_uses_one_global_24h_amount_scale() -> None:
    hour = 3_600_000
    frame = pd.DataFrame(
        [
            {"bar_start_ms": 0, "bucket_start_ms": 0, "side": "bid", "display_depth": 100.0, "local_ratio": 1.0},
            {"bar_start_ms": 0, "bucket_start_ms": 0, "side": "ask", "display_depth": 200.0, "local_ratio": 1.0},
            {"bar_start_ms": hour, "bucket_start_ms": hour, "side": "bid", "display_depth": 80.0, "local_ratio": 1.0},
            {"bar_start_ms": hour, "bucket_start_ms": hour, "side": "ask", "display_depth": 50.0, "local_ratio": 1.0},
        ]
    )
    intensity, cap, caps = _normalize_depth(
        frame,
        mode="causal_max_ratio",
        manual_max=100.0,
        rolling_window_hours=24,
        return_caps=True,
    )
    assert cap == pytest.approx(199.0)
    np.testing.assert_allclose(caps.to_numpy(), [199.0, 199.0, 199.0, 199.0])
    np.testing.assert_allclose(intensity.to_numpy(), [100.0 / 199.0, 1.0, 80.0 / 199.0, 50.0 / 199.0])


def test_causal_max_ratio_is_invariant_to_future_extension() -> None:
    base = pd.DataFrame(
        [
            {"bar_start_ms": 0, "bucket_start_ms": 0, "side": "bid", "display_depth": 100.0, "local_ratio": 1.0},
            {"bar_start_ms": 0, "bucket_start_ms": 0, "side": "ask", "display_depth": 200.0, "local_ratio": 1.0},
        ]
    )
    future = pd.concat(
        [
            base,
            pd.DataFrame(
                [{"bar_start_ms": 60_000, "bucket_start_ms": 60_000, "side": "bid", "display_depth": 10_000.0, "local_ratio": 1.0}]
            ),
        ],
        ignore_index=True,
    )
    base_intensity, _, base_caps = _normalize_depth(
        base,
        mode="causal_max_ratio",
        manual_max=100.0,
        rolling_window_hours=24,
        return_caps=True,
    )
    future_intensity, _, future_caps = _normalize_depth(
        future,
        mode="causal_max_ratio",
        manual_max=100.0,
        rolling_window_hours=24,
        return_caps=True,
    )
    np.testing.assert_allclose(base_intensity.to_numpy(), future_intensity.iloc[:2].to_numpy())
    np.testing.assert_allclose(base_caps.to_numpy(), future_caps.iloc[:2].to_numpy())


def test_zero_backend_cutoff_preserves_every_positive_cell_when_under_budget() -> None:
    frame = _display_frame()
    reduced, threshold = _reduce_cells(frame, min_intensity=0.0, max_cells=len(frame) + 10)
    assert threshold == 0.0
    assert len(reduced) == len(frame)
    assert float(reduced["intensity"].min()) > 0.0
