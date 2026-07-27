from __future__ import annotations

import numpy as np
import pandas as pd

from src.liquidation_map import EstimatedLiquidationMapEngine, LiquidationMapConfig


def _bars(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2026-06-01", periods=rows, freq="1min")
    close = 2000.0 + np.linspace(0, 30, rows) + np.sin(np.arange(rows) / 8.0) * 3.0
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100.0,
            "buy_notional": 1_100_000.0,
            "sell_notional": 900_000.0,
            "delta_notional": 200_000.0,
        },
        index=index,
    )


def _oi(rows: int = 48) -> pd.DataFrame:
    index = pd.date_range("2026-06-01 00:04", periods=rows, freq="5min")
    oi = 1_000_000_000.0 + np.arange(rows) * 5_000_000.0
    return pd.DataFrame({"oi_usd": oi, "oi_ccy": oi / 2000.0}, index=index)


def test_positive_oi_creates_both_sides_and_rows() -> None:
    result = EstimatedLiquidationMapEngine(
        LiquidationMapConfig(snapshot_every_bars=10, minimum_oi_delta_usd=1.0)
    ).compute(_bars(), open_interest=_oi())
    assert result.diagnostics["ready"] is True
    assert result.cells
    assert {cell.side for cell in result.cells} == {"long", "short"}
    assert result.row_frame["model_confidence"].max() > 0
    assert result.row_frame["nearest_short_liq_price"].notna().any()
    assert result.row_frame["nearest_long_liq_price"].notna().any()


def test_prefix_is_unchanged_when_future_is_appended() -> None:
    config = LiquidationMapConfig(snapshot_every_bars=5, minimum_oi_delta_usd=1.0)
    bars = _bars(300)
    oi = _oi(60)
    engine = EstimatedLiquidationMapEngine(config)
    short = engine.compute(bars.iloc[:200], open_interest=oi.loc[: bars.index[199]])
    long = engine.compute(bars, open_interest=oi)
    cols = ["liquidation_balance", "model_confidence", "oi_usd", "oi_delta_usd"]
    pd.testing.assert_frame_equal(
        short.row_frame[cols],
        long.row_frame.loc[short.row_frame.index, cols],
        check_dtype=False,
    )


def test_liquidation_price_is_on_correct_side() -> None:
    engine = EstimatedLiquidationMapEngine()
    assert engine._liquidation_price(2000.0, "long", 10) < 2000.0
    assert engine._liquidation_price(2000.0, "short", 10) > 2000.0
