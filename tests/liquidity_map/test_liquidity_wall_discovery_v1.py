from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.liquidity_wall_discovery import (
    CausalDepthReference,
    WallDiscoveryConfig,
    WallDiscoveryEngine,
    attach_touch_outcomes,
    build_wall_touch_events,
    discover_wall_states,
    environment_feature_uplift,
    extract_snapshot_candidates,
)


def _snapshot(
    timestamp: str,
    *,
    bid_depths: dict[int, float] | None = None,
    ask_depths: dict[int, float] | None = None,
    flow_overrides: dict[tuple[str, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(timestamp, tz="UTC")
    start_ms = int(start.timestamp() * 1000)
    end_ms = start_ms + 5_000
    bid_depths = bid_depths or {}
    ask_depths = ask_depths or {}
    flow_overrides = flow_overrides or {}
    rows = []
    for side, side_code, prices in (
        ("bid", 1, range(80, 100)),
        ("ask", -1, range(100, 120)),
    ):
        custom = bid_depths if side == "bid" else ask_depths
        for price in prices:
            flow = flow_overrides.get((side, price), {})
            depth = float(custom.get(price, 10.0))
            rows.append(
                {
                    "bucket_start_ms": start_ms,
                    "bucket_end_ms": end_ms,
                    "price_index": price,
                    "side_code": side_code,
                    "end_depth_base": depth,
                    "depth_base": depth,
                    "added_base": float(flow.get("added_base", 0.0)),
                    "removed_base": float(flow.get("removed_base", 0.0)),
                    "executed_base": float(flow.get("executed_base", 0.0)),
                    "cancelled_base": float(flow.get("cancelled_base", 0.0)),
                    "consumed_base": float(flow.get("consumed_base", 0.0)),
                    "replenished_base": float(flow.get("replenished_base", 0.0)),
                    "flow_valid": 1,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["heatmap_seconds"] = 5
    frame.attrs["price_step"] = 1.0
    return frame


def _cfg(**overrides) -> WallDiscoveryConfig:
    values = {
        "price_step": 1.0,
        "candidate_widths": (1, 3, 5, 8),
        "maximum_distance_bps": 3000.0,
        "maximum_candidates_per_side": 8,
        "minimum_track_observations": 2,
        "minimum_touch_age_seconds": 5.0,
        "minimum_bin_events": 1,
    }
    values.update(overrides)
    return WallDiscoveryConfig(**values)


def test_wide_dense_band_is_candidate_but_shallow_area_is_not() -> None:
    bids = {price: 45.0 for price in range(90, 95)}
    bids.update({price: 14.0 for price in range(81, 88)})
    snapshot = _snapshot("2026-01-01 00:00:00", bid_depths=bids)
    candidates = extract_snapshot_candidates(
        snapshot,
        _cfg(),
        CausalDepthReference(window_hours=24, quantile=0.99),
    )
    walls = [item for item in candidates if item.side == "bid" and item.price_low <= 90 < item.price_high]
    assert walls
    assert walls[0].morphology in {"BAND", "COMPOSITE"}
    assert not [item for item in candidates if item.side == "bid" and item.price_low <= 83 < item.price_high]


def test_isolated_deep_line_remains_point_morphology() -> None:
    snapshot = _snapshot("2026-01-01 00:00:00", bid_depths={94: 70.0})
    candidates = extract_snapshot_candidates(
        snapshot,
        _cfg(),
        CausalDepthReference(window_hours=24, quantile=0.99),
    )
    point = [item for item in candidates if item.side == "bid" and item.low_bin == 94]
    assert point
    assert point[0].morphology == "POINT"
    assert point[0].width_bins == 1


def test_moving_band_is_tracked_as_ghost_not_rewritten_as_fixed_wall() -> None:
    frames = []
    origin = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    for i in range(20):
        center = 88 + i
        bids = {price: 35.0 for price in range(center, center + 5) if price < 100}
        frames.append(_snapshot(str(origin + pd.Timedelta(seconds=5 * i)), bid_depths=bids))
    heatmap = pd.concat(frames, ignore_index=True)
    heatmap.attrs["heatmap_seconds"] = 5
    heatmap.attrs["price_step"] = 1.0
    _, states, tracks = discover_wall_states(
        heatmap,
        _cfg(maximum_center_drift_bins=3.0, ghost_drift_widths_per_minute=0.20),
    )
    assert not states.empty
    assert int(states["wall_is_ghost"].max()) == 1
    assert float(tracks["wall_center_span_bins"].max()) > 0


def test_disappearing_band_with_cancel_flow_is_classified_withdrawn() -> None:
    cfg = _cfg(maximum_missing_frames=0, minimum_track_observations=1)
    engine = WallDiscoveryEngine(cfg, source_seconds=5)
    first = _snapshot("2026-01-01 00:00:00", bid_depths={90: 35, 91: 35, 92: 35, 93: 35, 94: 35})
    engine.process_snapshot(first)
    second = _snapshot(
        "2026-01-01 00:00:05",
        bid_depths={90: 1, 91: 1, 92: 1, 93: 1, 94: 1},
        flow_overrides={
            ("bid", price): {"removed_base": 30, "cancelled_base": 28, "consumed_base": 2}
            for price in range(90, 95)
        },
    )
    engine.process_snapshot(second)
    closed = engine.drain_closed_summaries()
    assert closed
    assert "WITHDRAWN" in {row["death_reason"] for row in closed}


def _bars(index: pd.DatetimeIndex, values: list[tuple[float, float, float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for open_, high, low, close, buy_notional, sell_notional in values:
        rows.append(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "notional": buy_notional + sell_notional,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "large_buy_notional": 0.0,
                "large_sell_notional": sell_notional * 0.8,
                "delta_notional": buy_notional - sell_notional,
                "volume": 1.0,
            }
        )
    return pd.DataFrame(rows, index=index)


def _wall_state(available: pd.Timestamp) -> dict:
    return {
        "wall_id": 1,
        "available_time_ms": int(available.timestamp() * 1000),
        "bucket_start_ms": int((available - pd.Timedelta(seconds=5)).timestamp() * 1000),
        "bucket_end_ms": int(available.timestamp() * 1000),
        "side": "bid",
        "morphology": "BAND",
        "price_low": 100.0,
        "price_high": 101.0,
        "center_price": 100.5,
        "width_bins": 1,
        "wall_observations": 3,
        "wall_age_seconds": 10.0,
        "wall_current_retention": 1.0,
        "wall_is_ghost": 0,
    }


def test_touch_uses_state_before_bar_and_never_same_bar_future_snapshot() -> None:
    idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="5s", tz="UTC")
    bars = _bars(
        idx,
        [
            (102, 102, 100.5, 101.5, 50, 50),  # touch before state is available
            (102, 102.2, 101.5, 102, 50, 50),  # no touch after state available
            (102, 102.2, 100.8, 101.2, 50, 50),
        ],
    )
    states = pd.DataFrame([_wall_state(idx[1])])
    events = build_wall_touch_events(states, bars, _cfg())
    assert events.empty

    states = pd.DataFrame([_wall_state(idx[0])])
    events = build_wall_touch_events(states, bars, _cfg(minimum_touch_age_seconds=0))
    assert len(events) == 1
    assert int(events.iloc[0]["available_time_ms"]) <= int(events.iloc[0]["touch_time_ms"])


def test_outcomes_keep_bounce_and_volume_break_symmetric() -> None:
    idx = pd.date_range("2026-01-01 00:00:00", periods=20, freq="5s", tz="UTC")
    values = [(101, 101.05, 100.5, 100.8, 50, 50)] * 20
    values[3] = (100.8, 101.4, 100.4, 101.2, 60, 40)  # bounce first
    bars = _bars(idx, values)
    event = _wall_state(idx[1])
    event.update({"touch_time_ms": int(idx[1].timestamp() * 1000), "touch_bar_end_ms": int(idx[2].timestamp() * 1000), "touch_time": idx[1]})
    bounce = attach_touch_outcomes(pd.DataFrame([event]), bars, _cfg(bounce_bps=(20,), outcome_horizons_minutes=(1,)))
    assert bounce.iloc[0]["primary_outcome"] == "BOUNCE"

    values = [(101, 101.05, 100.5, 100.8, 50, 50)] * 20
    values[3] = (100.5, 100.6, 98.5, 99.0, 100, 900)
    values[4] = (99.0, 99.2, 97.0, 97.5, 50, 950)
    break_bars = _bars(idx, values)
    broken = attach_touch_outcomes(pd.DataFrame([event]), break_bars, _cfg(bounce_bps=(20,), outcome_horizons_minutes=(1,)))
    assert broken.iloc[0]["primary_outcome"] == "BREAK"
    assert int(broken.iloc[0]["volume_confirmed_break"]) == 1
    assert float(broken.iloc[0]["break_continuation_bps"]) > 0


def test_environment_bins_are_defined_on_train_and_reused_on_holdout() -> None:
    times = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    events = pd.DataFrame(
        {
            "touch_time_ms": (times.view("int64") // 1_000_000).astype("int64"),
            "feature_x": np.arange(80, dtype=float),
            "primary_outcome": ["BREAK"] * 40 + ["BOUNCE"] * 40,
            "volume_confirmed_break": [0] * 80,
            "close_return_15m": np.linspace(-0.01, 0.01, 80),
            "mfe_15m_bps": np.linspace(0, 100, 80),
            "mae_15m_bps": np.linspace(-100, 0, 80),
        }
    )
    uplift = environment_feature_uplift(
        events,
        features=["feature_x"],
        train_fraction=0.75,
        quantiles=4,
        minimum_bin_events=1,
    )
    train = uplift.loc[uplift["sample"] == "train"]
    holdout = uplift.loc[uplift["sample"] == "holdout"]
    assert len(train) == 4
    assert len(holdout) == 1
    assert np.isposinf(float(holdout.iloc[0]["bin_high"]))


def test_research_entrypoint_writes_wall_discovery_outputs(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    script_path = Path("research/liquidity/liquidity_wall_discovery_v1/01_liquidity_wall_discovery_research.py")
    spec = importlib.util.spec_from_file_location("liquidity_wall_discovery_entry", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    origin = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    snapshots = []
    for i in range(12):
        frame = _snapshot(
            str(origin + pd.Timedelta(seconds=5 * i)),
            bid_depths={90: 45, 91: 45, 92: 45, 93: 45, 94: 45},
        )
        snapshots.append(frame)
    heatmap = pd.concat(snapshots, ignore_index=True)
    heatmap["start_timestamp"] = pd.to_datetime(heatmap["bucket_start_ms"], unit="ms", utc=True)
    heatmap["end_timestamp"] = pd.to_datetime(heatmap["bucket_end_ms"], unit="ms", utc=True)
    heatmap.attrs["heatmap_seconds"] = 5
    heatmap.attrs["price_step"] = 1.0
    heatmap.attrs["utc_day"] = "2026-01-01"

    index = pd.date_range(origin - pd.Timedelta(minutes=1), periods=80, freq="5s")
    values = [(98, 98.2, 97.5, 98, 50, 50)] * len(index)
    touch_position = int(np.where(index == origin + pd.Timedelta(seconds=25))[0][0])
    values[touch_position] = (96, 96.2, 93.5, 95.2, 40, 160)
    values[touch_position + 1] = (95.2, 96.2, 95.0, 96.0, 100, 40)
    bars = _bars(index, values)

    class FakeLiquidityLoader:
        def __init__(self, **kwargs):
            pass

        def coverage(self):
            return [SimpleNamespace(day="2026-01-01")]

        def iter_heatmap_days(self, *args, **kwargs):
            yield heatmap

    class FakeTradeLoader:
        def __init__(self, **kwargs):
            pass

        def fetch_data_by_date_range(self, *args, **kwargs):
            return bars

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeLiquidityLoader)
    monkeypatch.setattr(module, "OKXTradeBarLoader", FakeTradeLoader)
    out_dir = tmp_path / "report"
    args = module.parse_args(
        [
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-01 00:01:00",
            "--out-dir",
            str(out_dir),
            "--minimum-bin-events",
            "1",
            "--candidate-audit-every-snapshots",
            "1",
            "--candidate-audit-top",
            "2",
            "--maximum-distance-bps",
            "3000",
        ]
    )
    result = module.run(args)
    assert result == out_dir
    assert (out_dir / "00_manifest.json").exists()
    assert (out_dir / "03_touch_events.csv").exists()
    assert (out_dir / "13_wall_overlay_segments.csv").exists()
    assert (out_dir / "gpt_review_pack.zip").exists()
    events = pd.read_csv(out_dir / "03_touch_events.csv")
    assert not events.empty
    assert set(events["primary_outcome"]).intersection({"BOUNCE", "BREAK", "NEITHER", "AMBIGUOUS"})
