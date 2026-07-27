from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_feed.okx_event_trade_window_loader import OKXEventTradeWindowLoader
from src.research_common.post_sweep_micro import (
    PostSweepMicroConfig,
    analyze_event_range_context,
    analyze_micro_window,
    build_attempt_universe,
    causal_audit,
)


def _write_trade_zip(data_dir: Path, day: str, rows: pd.DataFrame) -> None:
    raw_dir = data_dir / "okx" / "raw" / "trades" / "ETH-USDT-SWAP"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"ETH-USDT-SWAP-trades-{day}.zip"
    csv_name = f"ETH-USDT-SWAP-trades-{day}.csv"
    payload = rows.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, payload)


def test_sparse_event_window_loader_reads_each_window(tmp_path: Path) -> None:
    utc = pd.date_range("2025-01-01 00:00:00", periods=20, freq="1s", tz="UTC")
    trades = pd.DataFrame(
        {
            "ts": (utc.astype("int64") // 1_000_000).astype("int64"),
            "px": 100.0 + np.arange(len(utc)) * 0.01,
            "sz": 1.0,
            "side": np.where(np.arange(len(utc)) % 2 == 0, "buy", "sell"),
        }
    )
    _write_trade_zip(tmp_path, "2025-01-01", trades)
    windows = pd.DataFrame(
        {
            "window_id": ["W1", "W2"],
            "start_time": [pd.Timestamp("2025-01-01 08:00:02"), pd.Timestamp("2025-01-01 08:00:10")],
            "end_time": [pd.Timestamp("2025-01-01 08:00:08"), pd.Timestamp("2025-01-01 08:00:15")],
        }
    )
    loader = OKXEventTradeWindowLoader(data_dir=tmp_path)
    batches = list(loader.iter_daily_window_bars(windows, timeframe="1s", chunksize=7))
    real = [batch for batch in batches if batch.utc_day.year != 1970]
    assert len(real) == 1
    bars = real[0].bars
    assert set(bars["window_id"]) == {"W1", "W2"}
    assert len(bars.loc[bars["window_id"] == "W1"]) == 6
    assert (bars["available_time"] == bars["timestamp"] + pd.Timedelta(seconds=1)).all()
    assert set(real[0].coverage["status"]) == {"complete"}


def _universe_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    base = pd.Timestamp("2025-01-01 00:00:00")
    for event, outcome in (("E1", "oracle"), ("E2", "control")):
        for elapsed in (1, 2, 3):
            checkpoint = f"{event}_C{elapsed}"
            rows.append(
                {
                    "checkpoint_id": checkpoint,
                    "zone_event_id": event,
                    "period": "P1",
                    "elapsed_bars": elapsed,
                    "checkpoint_time": base + pd.Timedelta(minutes=elapsed),
                    "checkpoint_available_time": base + pd.Timedelta(minutes=elapsed + 1),
                    "event_available_time": base,
                    "new_low_attempt_flag": elapsed in (1, 3),
                    "new_low_attempt_index": 1 if elapsed < 3 else 2,
                    "running_low_since_sweep": 100.0 - elapsed * 0.1,
                    "sweep_low": 99.9,
                    "checkpoint_open": 100.0,
                    "checkpoint_high": 100.1,
                    "checkpoint_low": 99.8,
                    "checkpoint_close": 99.9,
                    "event_kind": "ZONE_SWEEP",
                    "event_pos": 1,
                    "checkpoint_pos": elapsed,
                    "zone_floor_price": 100.0,
                    "zone_ceiling_price": 100.1,
                    "zone_center_price": 100.05,
                    "bars_since_new_low_attempt": 0,
                    "new_low_extension_bp": 1.0,
                    "new_low_extension_to_pre_atr_240m": 0.1,
                    "attempt_delta_notional": -1.0,
                    "attempt_sell_notional": 2.0,
                    "attempt_extension_vs_previous": 1.0,
                    "close_vs_zone_floor_bp": -10.0,
                    "close_vs_running_low_bp": 5.0,
                    "running_low_vs_zone_floor_bp": -20.0,
                    "delta_ratio_1m": -0.2,
                    "sell_share_1m": 0.6,
                    "large_delta_ratio_1m": -0.3,
                    "price_change_1m_bp": -5.0,
                    "downside_bp_per_sell_million_1m": 0.5,
                    "downside_bp_per_abs_negative_delta_million_1m": 1.0,
                    "delta_ratio_5m": -0.2,
                    "sell_share_5m": 0.6,
                    "price_change_5m_bp": -10.0,
                }
            )
            is_oracle = event == "E1" and elapsed == 3
            is_control = event == "E2" and elapsed == 3
            labels.append(
                {
                    "checkpoint_id": checkpoint,
                    "zone_event_id": event,
                    "period": "P1",
                    "elapsed_bars": elapsed,
                    "entry_reference_time": base,
                    "entry_reference_price": 100.0,
                    "future_label_complete_15m": True,
                    "future_mfe_15m": 0.01,
                    "future_mae_15m": -0.002,
                    "future_close_return_15m": 0.005,
                    "future_no_lower_low_15m": is_oracle,
                    "future_label_complete_30m": True,
                    "future_mfe_30m": 0.01,
                    "future_mae_30m": -0.002,
                    "future_close_return_30m": 0.005,
                    "future_no_lower_low_30m": is_oracle,
                    "future_label_complete_60m": True,
                    "future_mfe_60m": 0.01 if is_oracle else 0.001,
                    "future_mae_60m": -0.002,
                    "future_close_return_60m": 0.005,
                    "future_no_lower_low_60m": is_oracle,
                    "future_reversal_dominant_60m": is_oracle,
                    "future_continuation_dominant_60m": is_control,
                    "future_label_complete_180m": True,
                    "future_mfe_180m": 0.02,
                    "future_mae_180m": -0.003,
                    "future_close_return_180m": 0.01,
                    "future_no_lower_low_180m": is_oracle,
                    "future_reversal_dominant_180m": is_oracle,
                    "future_continuation_dominant_180m": is_control,
                    "future_large_mfe_0p5_180m": is_oracle,
                    "future_large_mfe_1_180m": is_oracle,
                    "future_large_mfe_2_180m": False,
                }
            )
    static = pd.DataFrame({"zone_event_id": ["E1", "E2"]})
    return pd.DataFrame(rows), pd.DataFrame(labels), static


def test_attempt_universe_physically_separates_future_labels() -> None:
    features, labels, static = _universe_source()
    cfg = PostSweepMicroConfig(post_window_seconds=660, control_multiplier=1.0).validate()
    universe, label_table, pairs = build_attempt_universe(features, labels, static, cfg)
    assert len(pairs) == 1
    assert set(universe["cohort"]) >= {"ORACLE_TURN", "PRIOR_FAILED_ATTEMPT"}
    assert not any(column.startswith("future_") for column in universe.columns)
    assert any(column.startswith("future_") for column in label_table.columns)


def _synthetic_micro(event: pd.Series) -> pd.DataFrame:
    index = pd.date_range(event["start_time"], event["end_time"], freq="1s", inclusive="left")
    n = len(index)
    anchor = int((event["checkpoint_time"] - event["start_time"]).total_seconds())
    close = np.full(n, 100.0)
    close[:anchor] = 100.0
    close[anchor:anchor + 20] = np.linspace(100.0, 99.7, 20)
    close[anchor + 20:] = np.linspace(99.7, 100.5, n - anchor - 20)
    sell = np.full(n, 1_000_000.0)
    buy = np.full(n, 500_000.0)
    buy[anchor + 20:] = 900_000.0
    return pd.DataFrame(
        {
            "window_id": event["window_id"], "timestamp": index,
            "available_time": index + pd.Timedelta(seconds=1),
            "open": close, "high": close + 0.01, "low": close - 0.01, "close": close,
            "volume": 1.0, "trades_count": 10, "buy_volume": 0.4, "sell_volume": 0.6,
            "notional": buy + sell, "buy_notional": buy, "sell_notional": sell,
            "buy_trades_count": 4, "sell_trades_count": 6,
            "delta_volume": -0.2, "delta_notional": buy - sell,
            "taker_buy_ratio": buy / (buy + sell), "large_buy_notional": 0.0,
            "large_sell_notional": 0.0, "large_delta_notional": 0.0,
            "large_buy_trades_count": 0, "large_sell_trades_count": 0,
            "large_trades_count": 0, "max_trade_notional": 10_000.0,
            "max_trade_size": 1.0, "vwap": close,
        }
    )


def test_micro_triggers_enter_next_second_open_and_audit_passes() -> None:
    anchor = pd.Timestamp("2025-01-01 12:00:00")
    event = pd.Series(
        {
            "window_id": "W", "checkpoint_id": "C", "zone_event_id": "E",
            "pair_id": "E", "cohort": "ORACLE_TURN", "period": "P",
            "checkpoint_time": anchor, "start_time": anchor - pd.Timedelta(seconds=60),
            "end_time": anchor + pd.Timedelta(seconds=660),
            "prior_running_low_before_attempt": 99.95,
        }
    )
    cfg = PostSweepMicroConfig().validate()
    feature, trigger_rows, audit = analyze_micro_window(_synthetic_micro(event), event, cfg)
    assert feature is not None
    assert audit["status"] == "complete"
    triggers = pd.DataFrame(trigger_rows)
    causal = triggers.loc[~triggers["signal_uses_future"]]
    assert not causal.empty
    assert causal["entry_is_next_bar_open"].all()
    universe = pd.DataFrame({"window_id": ["W"]})
    result = causal_audit(universe, triggers)
    assert not (result["status"] == "FAIL").any()


def test_range_context_uses_completed_end_times() -> None:
    anchor = pd.Timestamp("2025-01-01 12:00:00")
    event = pd.Series(
        {
            "window_id": "W", "checkpoint_id": "C", "zone_event_id": "E",
            "pair_id": "E", "cohort": "ORACLE_TURN", "period": "P",
            "checkpoint_time": anchor,
        }
    )
    bars = pd.DataFrame(
        {
            "bar_id": [1, 2, 3],
            "start_ts": [anchor - pd.Timedelta(seconds=30), anchor, anchor + pd.Timedelta(seconds=20)],
            "end_ts": [anchor, anchor + pd.Timedelta(seconds=20), anchor + pd.Timedelta(seconds=40)],
            "duration_seconds": [30.0, 20.0, 20.0],
            "open": [100.0, 99.8, 99.6], "close": [99.8, 99.6, 99.9],
            "direction": [-1.0, -1.0, 1.0], "notional": [2e6, 2e6, 2e6],
            "buy_notional": [0.5e6, 0.5e6, 1.5e6], "sell_notional": [1.5e6, 1.5e6, 0.5e6],
            "delta_notional": [-1e6, -1e6, 1e6], "max_trade_notional": [100e3, 100e3, 100e3],
        }
    )
    result = analyze_event_range_context(event, bars, 0.002)
    assert result["status"] == "complete"
    assert result["first_up_end_ts"] == anchor + pd.Timedelta(seconds=40)
    assert result["first_up_delay_seconds"] == 40.0


def test_direct_binance_oi_context_uses_indexed_store(tmp_path: Path) -> None:
    from src.data_feed.binance_futures_metrics_loader import (
        BinanceFuturesMetricsLoader,
        BinanceMetricsDayResult,
    )
    from src.research_common.post_sweep_micro import load_binance_oi_context

    loader = BinanceFuturesMetricsLoader(data_dir=tmp_path)
    local_times = pd.to_datetime([
        "2025-01-01 08:00:00",
        "2025-01-01 08:05:00",
        "2025-01-01 08:10:00",
    ])
    frame = pd.DataFrame(
        {
            "symbol": ["ETHUSDT"] * 3,
            "timestamp": local_times,
            "source_timestamp_utc": local_times - pd.Timedelta(hours=8),
            "period": ["5m"] * 3,
            "sum_open_interest": [100.0, 110.0, 121.0],
            "sum_open_interest_value": [1000.0, 1100.0, 1210.0],
            "count_toptrader_long_short_ratio": [1.2] * 3,
            "sum_toptrader_long_short_ratio": [1.1] * 3,
            "count_long_short_ratio": [0.9] * 3,
            "sum_taker_long_short_vol_ratio": [0.8, 1.0, 1.2],
            "source_day_utc": ["2025-01-01"] * 3,
            "source": ["test"] * 3,
        }
    )
    loader.store.save_day(
        BinanceMetricsDayResult(
            day_utc=pd.Timestamp("2025-01-01").date(),
            status="partial",
            rows=len(frame),
            frame=frame,
            source_url="test",
        )
    )
    checkpoints = pd.DataFrame(
        {
            "checkpoint_id": ["C1"],
            "checkpoint_available_time": [pd.Timestamp("2025-01-01 08:11:00")],
            "price_change_5m_bp": [-10.0],
            "delta_ratio_5m": [-0.2],
        }
    )
    context = load_binance_oi_context(checkpoints, data_dir=tmp_path)
    assert len(context) == 1
    assert bool(context.loc[0, "oi_context_present"])
    assert context.loc[0, "oi_base"] == pytest.approx(121.0)
    assert context.loc[0, "oi_base_change_5m"] == pytest.approx(0.1)
    assert bool(context.loc[0, "down_oi_up_flag"])
