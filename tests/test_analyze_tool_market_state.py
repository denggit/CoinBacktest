from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.plugins import build_default_registry
from analyze_tool.plugins.market_state_map import MarketStateMapPlugin
from analyze_tool.server import _json_safe


def _sample(*, rich_orderflow: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    rows = 1300
    index = pd.date_range("2026-01-01", periods=rows, freq="1min")
    returns = np.r_[
        rng.normal(0.0, 0.00007, 350),
        np.full(450, 0.00030) + rng.normal(0.0, 0.00004, 450),
        rng.normal(0.0, 0.0010, 300),
        np.full(200, -0.00025) + rng.normal(0.0, 0.00005, 200),
    ]
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    width = np.r_[np.full(800, 0.00025), np.full(300, 0.0020), np.full(200, 0.00035)]
    high = np.maximum(open_, close) * (1.0 + width)
    low = np.minimum(open_, close) * (1.0 - width)
    volume = np.r_[rng.uniform(90, 110, 800), rng.uniform(450, 900, 300), rng.uniform(140, 220, 200)]
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    if rich_orderflow:
        notional = volume * close * 10.0
        directional = np.tanh(returns / 0.00020)
        delta = notional * np.clip(0.12 * directional + rng.normal(0.0, 0.025, rows), -0.35, 0.35)
        buy = (notional + delta) / 2.0
        sell = (notional - delta) / 2.0
        large_total = notional * 0.15
        large_delta = large_total * np.clip(0.7 * np.sign(delta) + rng.normal(0.0, 0.15, rows), -1.0, 1.0)
        df["notional"] = notional
        df["buy_notional"] = buy
        df["sell_notional"] = sell
        df["delta_notional"] = delta
        df["large_buy_notional"] = (large_total + large_delta) / 2.0
        df["large_sell_notional"] = (large_total - large_delta) / 2.0
        df["large_delta_notional"] = large_delta
        df["trades_count"] = np.maximum(10, (volume * 2).astype(int))
    return df


def test_registry_contains_market_state_map() -> None:
    rows = build_default_registry().list_plugins()
    by_id = {row["id"]: row for row in rows}
    assert "market_state_map_v0" in by_id
    assert "V3.1" in by_id["market_state_map_v0"]["name"]


def test_market_state_plugin_exposes_enhanced_state_and_aligns_availability() -> None:
    df = _sample(rich_orderflow=True)
    plugin = MarketStateMapPlugin()
    result = plugin.run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            analysis_frames={},
            request={"data_type": "trade_bar", "timeframe": "1m", "range_pct": 0.002},
            meta={},
        ),
        {
            "fast_trend_window": 12,
            "trend_window": 48,
            "slow_trend_window": 160,
            "volatility_window": 30,
            "activity_window": 12,
            "baseline_window": 240,
            "flow_fast_window": 3,
            "flow_window": 12,
            "flow_slow_window": 30,
            "location_window": 48,
            "structure_window": 160,
            "trend_confirm_bars": 3,
            "min_state_bars": 15,
        },
    )

    assert len(result.tracks) == 0
    assert len(result.bands) == 3
    assert [band.band_id for band in result.bands] == [
        "direction_permission", "market_phase", "market_process"
    ]
    assert result.bands[0].render_mode == "background"
    assert all(band.render_mode == "strip" for band in result.bands[1:])
    assert all(len(band.codes) == len(df) for band in result.bands)
    assert result.summary["ready_rows"] > 900
    assert result.summary["orderflow_ready_rows"] > 900
    assert result.summary["location_ready_rows"] > 900
    assert result.summary["timestamp_semantics"] == "bar_start"
    assert "向后移动" in result.summary["causal_availability"]
    assert result.summary["state_display_lag_bars"] == 1
    assert result.bands[0].codes[0] is None
    assert "trade_context" not in result.row_fields
    assert "sell_absorption" not in result.row_fields
    assert len(result.row_fields["brief_direction"]["values"]) == len(df)
    assert len(result.row_fields["brief_advice"]["values"]) == len(df)
    assert len(result.row_fields["brief_process"]["values"]) == len(df)
    assert len(result.row_fields["brief_process_probability"]) == len(df)
    assert result.summary["ui"]["compact"] is True
    assert result.summary["ui"]["view_mode"] == "trading"
    assert result.summary["not_trade_signal"] is True

    payload = _json_safe(result.as_dict())
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in raw and "Infinity" not in raw


def test_research_view_keeps_full_v02_diagnostics() -> None:
    df = _sample(rich_orderflow=True)
    result = MarketStateMapPlugin().run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            analysis_frames={},
            request={"data_type": "trade_bar", "timeframe": "1m", "range_pct": 0.002},
            meta={},
        ),
        {
            "view_mode": "research",
            "fast_trend_window": 12,
            "trend_window": 48,
            "slow_trend_window": 160,
            "baseline_window": 240,
            "location_window": 48,
            "structure_window": 160,
        },
    )
    assert len(result.tracks) == 12
    assert all(len(track.values) == len(df) for track in result.tracks)
    assert len(result.bands) == 6
    assert len(result.row_fields["trade_context"]["values"]) == len(df)
    assert len(result.row_fields["process_state"]["values"]) == len(df)
    assert len(result.row_fields["process_direction_probability"]) == len(df)
    assert len(result.row_fields["sell_absorption"]) == len(df)
    assert result.summary["ui"]["compact"] is False
    assert result.summary["ui"]["view_mode"] == "research"


def test_frontend_default_start_is_june_2026() -> None:
    html = (Path(__file__).resolve().parents[1] / "analyze_tool" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="start" type="datetime-local" value="2026-06-01T00:00"' in html
    assert "market-state-v3-process" in html


def test_plain_ohlcv_fails_closed_for_orderflow_but_keeps_location_map() -> None:
    df = _sample(rich_orderflow=False)
    result = MarketStateMapPlugin().run_with_context(
        PluginRunContext(
            display_df=df,
            visible_df=df,
            analysis_frames={},
            request={"data_type": "normal", "timeframe": "1m", "range_pct": 0.0},
            meta={},
        ),
        {
            "view_mode": "research",
            "fast_trend_window": 12,
            "trend_window": 48,
            "slow_trend_window": 160,
            "baseline_window": 240,
            "location_window": 48,
            "structure_window": 160,
        },
    )
    assert result.summary["orderflow_ready_rows"] == 0
    assert result.summary["location_ready_rows"] > 900
    assert "订单流不可用" in result.summary["orderflow_status"]
    flow_band = next(band for band in result.bands if band.band_id == "orderflow_regime")
    labels = {category.label for category in flow_band.categories}
    assert "订单流不可用" in labels
