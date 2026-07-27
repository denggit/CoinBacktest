from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity.liquidity_hunt_momentum_r01.core import (
    LiquidityHuntConfig,
    StrategyVariant,
    aggregate_footprint_features,
    align_book_features_to_times,
    attach_forward_time_outcomes,
    build_causal_audit,
    build_events,
    build_range_features,
    prepare_book_features,
    simulate_events,
)
from research.liquidity.liquidity_hunt_momentum_r01.selftest import _synthetic_m1_frame


def test_range_feature_builder_accepts_real_loader_index_contract() -> None:
    """OKXRangeBarLoader returns end_ts as both index and explicit column."""

    frame, cfg = _synthetic_m1_frame()
    loader_like = frame[
        [
            "bar_id",
            "start_ts",
            "end_ts",
            "open",
            "high",
            "low",
            "close",
            "direction",
            "notional",
            "buy_notional",
            "sell_notional",
            "taker_buy_ratio",
            "duration_seconds",
            "volume",
        ]
    ].copy()
    loader_like = loader_like.set_index("end_ts", drop=False)
    loader_like.index.name = "end_ts"

    rebuilt = build_range_features(loader_like, cfg)

    assert isinstance(rebuilt.index, pd.RangeIndex)
    assert rebuilt["end_ts"].is_monotonic_increasing
    assert rebuilt["bar_id"].tolist() == loader_like["bar_id"].tolist()


def test_load_range_frames_normalizes_loader_index(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    script_path = Path("research/liquidity/01_liquidity_hunt_momentum_event_study.py")
    spec = importlib.util.spec_from_file_location("liquidity_hunt_range_loader_contract", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame, cfg = _synthetic_m1_frame()
    loader_like = frame[
        [
            "bar_id",
            "start_ts",
            "end_ts",
            "open",
            "high",
            "low",
            "close",
            "direction",
            "notional",
            "buy_notional",
            "sell_notional",
            "taker_buy_ratio",
            "duration_seconds",
            "volume",
        ]
    ].copy()
    loader_like = loader_like.set_index("end_ts", drop=False)
    loader_like.index.name = "end_ts"

    class FakeRangeLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_local_data(self, start_date, end_date):
            return loader_like.copy()

    monkeypatch.setattr(module, "OKXRangeBarLoader", FakeRangeLoader)
    args = SimpleNamespace(
        symbol="ETH-USDT-SWAP",
        data_dir=None,
        range_bar_db_name="okx_range_bars.db",
        warmup_start_date="2026-01-01",
        end_date="2026-01-02",
        no_progress=True,
    )

    frames = module._load_range_frames(args, (0.002,), cfg)

    assert list(frames) == ["r0020"]
    assert isinstance(frames["r0020"].index, pd.RangeIndex)
    assert not frames["r0020"].empty


def test_book_alignment_never_uses_future_available_time() -> None:
    cfg = LiquidityHuntConfig(flow_window_seconds=2, book_reference_minutes=1)
    times = pd.date_range("2026-01-01 10:00:00", periods=8, freq="1s")
    raw = pd.DataFrame(
        {
            "available_time": times,
            "book_valid": 1,
            "trade_attribution_valid": 1,
            "bid_depth_5bps_base": np.arange(100.0, 108.0),
            "ask_depth_5bps_base": 100.0,
            "bid_depth_25bps_base": 200.0,
            "ask_depth_25bps_base": 200.0,
        }
    )
    prepared = prepare_book_features(raw, cfg)
    query = pd.to_datetime(["2026-01-01 10:00:03.500", "2026-01-01 10:00:07.500"])
    aligned = align_book_features_to_times(query, prepared, tolerance=pd.Timedelta(seconds=2))
    available = pd.to_datetime(aligned["book_available_time"])
    assert (available <= pd.Series(query)).all()
    assert available.tolist() == [pd.Timestamp("2026-01-01 10:00:03"), pd.Timestamp("2026-01-01 10:00:07")]


def test_stale_book_context_is_marked_missing() -> None:
    source = pd.DataFrame(
        {"available_time": [pd.Timestamp("2026-01-01 10:00:00")], "book_valid": [1]}
    ).set_index("available_time", drop=False)
    aligned = align_book_features_to_times(
        [pd.Timestamp("2026-01-01 10:00:20")],
        source,
        tolerance=pd.Timedelta(seconds=5),
    )
    assert bool(aligned.loc[0, "book_context_missing_flag"])
    assert pd.isna(aligned.loc[0, "book_available_time"])
    assert pd.api.types.is_datetime64_ns_dtype(aligned["book_available_time"])


def test_empty_book_alignment_keeps_datetime_dtype() -> None:
    aligned = align_book_features_to_times(
        [pd.Timestamp("2026-01-01 10:00:20")],
        pd.DataFrame(),
        tolerance=pd.Timedelta(seconds=5),
    )
    assert pd.api.types.is_datetime64_ns_dtype(aligned["book_available_time"])
    assert bool(aligned.loc[0, "book_context_missing_flag"])
    assert not bool(aligned.loc[0, "book_available_after_signal_flag"])


def test_mode1_nested_stages_and_next_range_open_execution() -> None:
    frame, cfg = _synthetic_m1_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    assert set(events["stage"]) == {
        "M1_FLOW_RECLAIM",
        "M1_FLOW_RECLAIM_OBI",
        "M1_FLOW_RECLAIM_OBI_REBUILD",
    }
    strict = events.loc[events["stage"] == "M1_FLOW_RECLAIM_OBI_REBUILD"]
    trades = simulate_events(strict, frame, cfg, StrategyVariant(name="baseline"))
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_time"] > trade["signal_time"]
    assert trade["entry_price"] == frame.loc[7, "open"]
    assert not bool(trade["entry_not_after_signal_flag"])


def _mode2_frame() -> tuple[pd.DataFrame, LiquidityHuntConfig]:
    cfg = LiquidityHuntConfig(
        support_lookback_bars=3,
        notional_median_bars=3,
        notional_min_periods=2,
        cooldown_minutes=0,
    )
    count = 10
    starts = pd.date_range("2026-01-02 10:00:01", periods=count, freq="1min")
    ends = starts + pd.Timedelta(seconds=35)
    open_ = np.arange(100.0, 110.0)
    close = open_ + 0.5
    notional = np.array([100, 100, 100, 100, 250, 260, 100, 100, 100, 100], dtype=float)
    buy_ratio = np.array([0.5, 0.5, 0.5, 0.5, 0.80, 0.82, 0.5, 0.5, 0.5, 0.5])
    raw = pd.DataFrame(
        {
            "bar_id": np.arange(count, dtype=np.int64),
            "start_ts": starts,
            "end_ts": ends,
            "open": open_,
            "high": close + 0.2,
            "low": open_ - 0.2,
            "close": close,
            "direction": 1,
            "duration_seconds": 35.0,
            "volume": notional,
            "notional": notional,
            "buy_notional": notional * buy_ratio,
            "sell_notional": notional * (1.0 - buy_ratio),
            "taker_buy_ratio": buy_ratio,
        }
    )
    frame = build_range_features(raw, cfg)
    frame["book_available_time"] = frame["signal_time"]
    frame["book_context_missing_flag"] = False
    frame["book_available_after_signal_flag"] = False
    frame["footprint_missing_flag"] = False
    for column in (
        "book_obi_5s",
        "book_obi_5s_min",
        "book_obi_5s_max",
        "book_ask_depth_25bps_ref_ratio",
        "book_bid_depth_25bps_ref_ratio",
        "book_ask_to_bid_depth_25bps",
        "book_bid_to_ask_depth_25bps",
    ):
        frame[column] = np.nan
    frame.loc[4:5, "book_obi_5s"] = 0.40
    frame.loc[4:5, "book_obi_5s_min"] = 0.35
    frame.loc[4:5, "book_obi_5s_max"] = 0.45
    frame.loc[5, "book_ask_depth_25bps_ref_ratio"] = 0.50
    frame.loc[5, "book_ask_to_bid_depth_25bps"] = 0.60
    return frame, cfg


def test_mode2_requires_two_attack_bars_obi_and_void() -> None:
    frame, cfg = _mode2_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    at_signal = events.loc[events["signal_time"] == frame.loc[5, "signal_time"]]
    assert set(at_signal["stage"]) == {
        "M2_TWO_BAR_ATTACK",
        "M2_TWO_BAR_ATTACK_OBI",
        "M2_TWO_BAR_ATTACK_OBI_VOID",
    }
    assert set(at_signal["side_name"]) == {"LONG"}


def test_same_bar_stop_and_target_is_scored_as_stop() -> None:
    frame, cfg = _synthetic_m1_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    strict = events.loc[events["stage"] == "M1_FLOW_RECLAIM_OBI_REBUILD"].copy()
    entry_pos = 7
    entry = float(frame.loc[entry_pos, "open"])
    stop = float(strict.iloc[0]["sweep_price"]) * (1.0 - cfg.mode1_stop_buffer_pct)
    frame.loc[entry_pos, "low"] = stop - 0.1
    frame.loc[entry_pos, "high"] = 104.1
    strict.loc[:, "opposite_liquidity_price"] = 104.0
    trades = simulate_events(strict, frame, cfg, StrategyVariant(name="both_hit"))
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_price"] == entry
    assert bool(trade["same_bar_both_hit_flag"])
    assert trade["exit_reason"] == "same_bar_both_stop_conservative"
    assert trade["exit_price"] == stop


def test_entry_skips_range_open_at_same_timestamp_as_signal() -> None:
    starts = pd.to_datetime(
        [
            "2026-01-03 10:00:00.000",
            "2026-01-03 10:00:30.000",
            "2026-01-03 10:01:00.000",
            "2026-01-03 10:02:00.000",
        ]
    )
    ends = pd.to_datetime(
        [
            "2026-01-03 10:00:30.000",
            "2026-01-03 10:01:00.000",
            "2026-01-03 10:02:00.000",
            "2026-01-03 10:03:00.000",
        ]
    )
    bars = pd.DataFrame(
        {
            "bar_id": range(4),
            "start_ts": starts,
            "end_ts": ends,
            "open": [100.0, 100.1, 100.2, 100.3],
            "high": [100.2, 100.3, 100.4, 100.5],
            "low": [99.8, 99.9, 100.0, 100.1],
            "close": [100.1, 100.2, 100.3, 100.4],
            "direction": [1, 1, 1, 1],
            "duration_seconds": [30.0, 30.0, 60.0, 60.0],
            "volume": 100.0,
            "notional": 100.0,
            "buy_notional": 70.0,
            "sell_notional": 30.0,
            "taker_buy_ratio": 0.70,
            "book_obi_5s": 0.4,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [1],
            "range_tag": ["r0020"],
            "mode": ["M2"],
            "stage": ["M2_TWO_BAR_ATTACK"],
            "side": [1],
            "side_name": ["LONG"],
            "signal_time": [ends[0]],
            "sweep_price": [99.0],
            "first_impulse_low": [90.0],
            "first_impulse_high": [101.0],
            "opposite_liquidity_price": [200.0],
            "book_available_after_signal_flag": [False],
        }
    )

    trades = simulate_events(
        events,
        bars,
        LiquidityHuntConfig(),
        StrategyVariant(
            name="strict_timestamp",
            use_dynamic_decay_exit=False,
            use_time_stop=False,
        ),
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_time"] == starts[2]
    assert trade["entry_price"] == bars.loc[2, "open"]
    assert trade["entry_time"] > trade["signal_time"]
    assert not bool(trade["entry_not_after_signal_flag"])


def test_next_open_exit_records_start_time_not_bar_end() -> None:
    starts = pd.date_range("2026-01-03 10:00:00", periods=5, freq="1min")
    bars = pd.DataFrame(
        {
            "bar_id": range(5),
            "start_ts": starts,
            "end_ts": starts + pd.Timedelta(seconds=30),
            "open": [100.0, 100.0, 100.2, 100.4, 100.5],
            "high": [100.2, 100.3, 100.4, 100.6, 100.7],
            "low": [99.8, 99.9, 100.0, 100.2, 100.3],
            "close": [100.0, 100.1, 100.2, 100.4, 100.5],
            "direction": [0, 1, 1, 1, 1],
            "duration_seconds": 30.0,
            "volume": 100.0,
            "notional": 100.0,
            "buy_notional": 50.0,
            "sell_notional": 50.0,
            "taker_buy_ratio": [0.5, 0.5, 0.5, 0.8, 0.8],
            "book_obi_5s": [0.0, 0.0, 0.0, 0.4, 0.4],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [1],
            "range_tag": ["r0020"],
            "mode": ["M2"],
            "stage": ["M2_TWO_BAR_ATTACK"],
            "side": [1],
            "side_name": ["LONG"],
            "signal_time": [bars.loc[0, "end_ts"]],
            "sweep_price": [99.0],
            "first_impulse_low": [90.0],
            "first_impulse_high": [101.0],
            "opposite_liquidity_price": [200.0],
            "book_available_after_signal_flag": [False],
        }
    )

    trades = simulate_events(
        events,
        bars,
        LiquidityHuntConfig(),
        StrategyVariant(name="next_open_exit", use_time_stop=False),
    )

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["exit_reason"] == "two_bar_flow_or_obi_decay_next_open"
    assert trade["exit_time"] == bars.loc[3, "start_ts"]
    assert trade["exit_price"] == bars.loc[3, "open"]


def test_short_forward_mfe_uses_arithmetic_return_convention() -> None:
    starts = pd.date_range("2026-01-03 10:00:01", periods=4, freq="1min")
    bars = pd.DataFrame(
        {
            "bar_id": range(4),
            "start_ts": starts,
            "end_ts": starts + pd.Timedelta(seconds=30),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 110.0, 100.0],
            "low": [100.0, 100.0, 90.0, 100.0],
            "close": [100.0, 100.0, 95.0, 100.0],
            "direction": [0, 0, -1, 0],
            "volume": 1.0,
            "notional": 100.0,
            "buy_notional": 50.0,
            "sell_notional": 50.0,
            "taker_buy_ratio": 0.5,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [1],
            "range_tag": ["r0020"],
            "mode": ["M2"],
            "stage": ["M2_TWO_BAR_ATTACK"],
            "side": [-1],
            "side_name": ["SHORT"],
            "signal_time": [bars.loc[0, "end_ts"]],
        }
    )
    outcome = attach_forward_time_outcomes(events, bars, horizons_minutes=[1], round_trip_cost=0.0)
    assert np.isclose(outcome.loc[0, "h1_mfe"], 0.10)
    assert np.isclose(outcome.loc[0, "h1_mae"], -0.10)


def test_footprint_aggregation_separates_low_and_high_zone_delta() -> None:
    fp = pd.DataFrame(
        {
            "bar_id": [1, 1, 1, 1],
            "price_bucket": [100.0, 101.0, 102.0, 103.0],
            "notional": [100.0, 100.0, 100.0, 100.0],
            "delta_notional": [-80.0, -20.0, 20.0, 80.0],
            "buy_notional": [10.0, 40.0, 60.0, 90.0],
            "sell_notional": [90.0, 60.0, 40.0, 10.0],
            "large_delta_notional": [0.0, 0.0, 0.0, 0.0],
            "max_trade_notional": [10.0, 10.0, 10.0, 10.0],
        }
    )
    out = aggregate_footprint_features(fp)
    assert len(out) == 1
    assert out.loc[0, "fp_low_zone_delta_ratio"] < 0
    assert out.loc[0, "fp_high_zone_delta_ratio"] > 0


def test_causal_audit_separates_missing_data_from_lookahead() -> None:
    frame, cfg = _synthetic_m1_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    event = events.loc[events["stage"] == "M1_FLOW_RECLAIM"].head(1).copy()
    event.loc[:, "book_available_time"] = pd.NaT
    event.loc[:, "book_context_missing_flag"] = True
    event.loc[:, "footprint_missing_flag"] = True

    audit = build_causal_audit(event)

    assert bool(audit.loc[0, "data_missing_flag"])
    assert not bool(audit.loc[0, "causal_fail_flag"])


def test_causal_audit_flags_future_book_row() -> None:
    frame, cfg = _synthetic_m1_frame()
    events = build_events(frame, cfg, range_tag="r0020")
    strict = events.loc[events["stage"] == "M1_FLOW_RECLAIM_OBI_REBUILD"].copy()
    strict.loc[:, "book_available_time"] = pd.to_datetime(strict["signal_time"]) + pd.Timedelta(seconds=1)
    audit = build_causal_audit(strict)
    assert bool(audit.loc[0, "book_available_after_signal_flag"])
    assert bool(audit.loc[0, "causal_fail_flag"])


def test_entrypoint_writes_complete_research_outputs(tmp_path, monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    script_path = Path("research/liquidity/01_liquidity_hunt_momentum_event_study.py")
    spec = importlib.util.spec_from_file_location("liquidity_hunt_momentum_entry", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame, _ = _synthetic_m1_frame()
    monkeypatch.setattr(module, "_load_range_frames", lambda args, range_pcts, cfg: {"r0020": frame.copy()})
    monkeypatch.setattr(
        module,
        "_attach_footprints",
        lambda args, range_pcts, frames: {"r0020": {"footprint_rows": 12, "footprint_bars": 12, "skipped": False}},
    )
    monkeypatch.setattr(
        module,
        "_attach_books_daily",
        lambda args, frames, cfg: {"coverage_days": 1, "requested_days": 1, "days_with_features": 1, "feature_rows_processed": 12},
    )
    args = module.parse_args(
        [
            "--warmup-start-date",
            "2026-01-01 10:00:00",
            "--start-date",
            "2026-01-01 10:00:00",
            "--end-date",
            "2026-01-01 10:20:00",
            "--range-pcts",
            "0.002",
            "--out-dir",
            str(tmp_path),
            "--skip-review-pack",
            "--skip-full-report",
            "--no-progress",
        ]
    )
    result = module.run(args)
    assert result == tmp_path
    expected = {
        "00_manifest.json",
        "01_data_quality.csv",
        "02_event_stage_summary.csv",
        "03_forward_path_summary.csv",
        "04_split_summary.csv",
        "05_strategy_summary.csv",
        "06_cost_stress.csv",
        "07_delay_stress.csv",
        "08_range_neighborhood.csv",
        "09_yearly.csv",
        "10_monthly.csv",
        "11_fixed_feature_uplift.csv",
        "12_causal_audit.csv",
        "13_event_sample.csv",
        "14_trade_sample.csv",
        "15_predeclared_thresholds.csv",
        "16_research_brief.md",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    audit = pd.read_csv(tmp_path / "12_causal_audit.csv")
    assert not audit["causal_fail_flag"].astype(bool).any()
    strict = pd.read_csv(tmp_path / "08_range_neighborhood.csv")
    assert int(strict["trades"].sum()) == 1


def test_datetime_search_normalizes_microseconds_to_nanoseconds() -> None:
    from research.liquidity.liquidity_hunt_momentum_r01.core import datetime_index_to_ns_int64

    values = pd.DatetimeIndex(
        np.array(
            ["2026-01-05T00:00:00.000000", "2026-01-05T00:00:01.000000"],
            dtype="datetime64[us]",
        )
    )
    encoded = datetime_index_to_ns_int64(values)

    assert values.dtype == np.dtype("datetime64[us]")
    assert encoded[0] == pd.Timestamp("2026-01-05 00:00:00").value
    assert encoded[1] == pd.Timestamp("2026-01-05 00:00:01").value


def test_attach_books_daily_handles_microsecond_range_timestamps(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    script_path = Path("research/liquidity/01_liquidity_hunt_momentum_event_study.py")
    spec = importlib.util.spec_from_file_location("liquidity_hunt_books_us_contract", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame, cfg = _synthetic_m1_frame()
    for column in ("start_ts", "end_ts", "signal_time"):
        frame[column] = pd.Series(
            frame[column].to_numpy(dtype="datetime64[us]"),
            index=frame.index,
            dtype="datetime64[us]",
        )
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class FakeBookLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def coverage(self):
            return [object()]

        def load_features(self, start, end, **kwargs):
            calls.append((pd.Timestamp(start), pd.Timestamp(end)))
            times = pd.date_range("2026-01-01 09:55:00", "2026-01-01 10:20:00", freq="1s")
            return pd.DataFrame(
                {
                    "available_time": times,
                    "book_valid": 1,
                    "trade_attribution_valid": 1,
                    "bid_depth_5bps_base": 100.0,
                    "ask_depth_5bps_base": 100.0,
                    "bid_depth_25bps_base": 200.0,
                    "ask_depth_25bps_base": 200.0,
                }
            )

    monkeypatch.setattr(module, "OKXLiquidityMapLoader", FakeBookLoader)
    args = SimpleNamespace(
        symbol="ETH-USDT-SWAP",
        books_depth=5000,
        data_dir=None,
        warmup_start_date="2026-01-01",
        start_date="2026-01-01",
        end_date="2026-01-01 23:59:59",
        book_tolerance_seconds=10,
        no_progress=True,
    )

    diagnostics = module._attach_books_daily(args, {"r0020": frame}, cfg)

    assert calls, "daily loader was skipped because datetime units were mixed"
    assert diagnostics["days_with_features"] == 1
    assert diagnostics["feature_rows_processed"] > 0
    assert (~frame["book_context_missing_flag"]).any()


def test_simulation_and_forward_paths_handle_microsecond_timestamps() -> None:
    frame, cfg = _synthetic_m1_frame()
    for column in ("start_ts", "end_ts", "signal_time", "book_available_time"):
        if column in frame.columns:
            frame[column] = pd.Series(
                frame[column].to_numpy(dtype="datetime64[us]"),
                index=frame.index,
                dtype="datetime64[us]",
            )
    events = build_events(frame, cfg, range_tag="r0020")
    strict = events.loc[events["stage"] == "M1_FLOW_RECLAIM_OBI_REBUILD"]

    trades = simulate_events(strict, frame, cfg, StrategyVariant(name="microsecond_axis"))
    outcomes = attach_forward_time_outcomes(strict, frame, horizons_minutes=[1])

    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] > trades.iloc[0]["signal_time"]
    assert outcomes["h1_gross_return"].notna().any()
