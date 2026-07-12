from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import PluginRunContext
from analyze_tool.plugins.panic_low_excursion_rejection import PanicLowExcursionRejectionPlugin
from research.liquidity.panic_selloff_rejection_recovery_long.panic_episode import BaselineConfig
from research.liquidity.panic_selloff_rejection_recovery_long.panic_low_excursion import (
    LowExcursionConfig,
    build_causal_low_excursion_features,
    build_low_excursion_flags,
    enrich_low_excursion_paths,
    low_excursion_preset,
    segment_low_excursion_episodes,
)


def _bars(remove_gap: bool = False) -> tuple[pd.DataFrame, pd.Timestamp]:
    idx = pd.date_range("2026-01-01", periods=14 * 1440, freq="1min")
    rng = np.random.default_rng(31)
    ret = rng.normal(0.0, 0.00005, len(idx))
    close = 2000.0 * np.exp(np.cumsum(ret))
    open_ = np.r_[2000.0, close[:-1]]
    high = np.maximum(open_, close) * 1.00005
    low = np.minimum(open_, close) * 0.99995
    panic_pos = 12 * 1440 + 480
    # Deep intrabar selloff with almost complete recovery: Round 02 close-return
    # dislocation should be small, while Round 03 low excursion is extreme.
    open_[panic_pos] = close[panic_pos - 1]
    close[panic_pos] = open_[panic_pos] * 0.9995
    low[panic_pos] = open_[panic_pos] * 0.985
    high[panic_pos] = open_[panic_pos] * 1.0002
    close[panic_pos + 1 : panic_pos + 8] = open_[panic_pos] * np.array([1.001, 1.002, 1.003, 1.0025, 1.004, 1.005, 1.0045])
    for p in range(panic_pos + 1, panic_pos + 8):
        open_[p] = close[p - 1]
        high[p] = max(open_[p], close[p]) * 1.00005
        low[p] = min(open_[p], close[p]) * 0.99995

    notional = np.full(len(idx), 800_000.0)
    trades = np.full(len(idx), 80.0)
    buy = np.full(len(idx), 400_000.0)
    sell = np.full(len(idx), 400_000.0)
    notional[panic_pos] = 12_000_000.0
    trades[panic_pos] = 1000.0
    buy[panic_pos] = 1_500_000.0
    sell[panic_pos] = 10_500_000.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
            "trades_count": trades,
            "buy_volume": 50.0,
            "sell_volume": 50.0,
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "buy_trades_count": 40.0,
            "sell_trades_count": 40.0,
            "delta_notional": buy - sell,
            "large_buy_notional": 0.0,
            "large_sell_notional": 0.0,
        },
        index=idx,
    )
    if remove_gap:
        df = df.drop(index=idx[panic_pos - 2])
    return df, idx[panic_pos]


def _baseline() -> BaselineConfig:
    return BaselineConfig(short_hours=48, long_hours=240, min_short_hours=12, min_long_hours=48, long_floor_ratio=0.5)


def test_long_lower_wick_low_excursion_is_detected_without_large_close_drop() -> None:
    df, panic = _bars()
    built = build_causal_low_excursion_features(
        df,
        baseline=_baseline(),
        tail_lookback_hours=240,
        tail_min_hours=48,
    )
    flags = build_low_excursion_flags(built.features, low_excursion_preset("core"))
    assert abs(df.loc[panic, "close"] / df.loc[panic, "open"] - 1.0) < 0.001
    assert bool(flags.loc[panic, "parent_trigger"])
    assert built.features.loc[panic, "rejection_fraction_dominant"] >= 0.90

    episodes = segment_low_excursion_episodes(
        built.features,
        preset=low_excursion_preset("core"),
        config=LowExcursionConfig(quiet_minutes=8, max_episode_minutes=60),
        research_start=panic - pd.Timedelta(minutes=5),
        research_end=panic + pd.Timedelta(minutes=30),
    )
    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert ep["detected_bar_time"] == panic
    assert ep["detected_time"] == panic + pd.Timedelta(minutes=1)
    assert ep["same_bar_rejection_band"] == "same_bar_rejection_ge75"
    paths = enrich_low_excursion_paths(episodes, built.features)
    assert paths.iloc[0]["detection_entry_time"] == panic + pd.Timedelta(minutes=1)


def test_gap_touch_disables_low_excursion_parent() -> None:
    df, panic = _bars(remove_gap=True)
    built = build_causal_low_excursion_features(
        df,
        baseline=_baseline(),
        tail_lookback_hours=240,
        tail_min_hours=48,
    )
    flags = build_low_excursion_flags(built.features, low_excursion_preset("core"))
    assert not bool(built.features.loc[panic, "data_valid"])
    assert not bool(flags.loc[panic, "parent_trigger"])


def test_current_wick_does_not_change_previous_causal_features() -> None:
    df, panic = _bars()
    base = build_causal_low_excursion_features(df, baseline=_baseline(), tail_lookback_hours=240, tail_min_hours=48).features
    changed = df.copy()
    changed.loc[panic, "low"] *= 0.95
    modified = build_causal_low_excursion_features(changed, baseline=_baseline(), tail_lookback_hours=240, tail_min_hours=48).features
    before = panic - pd.Timedelta(minutes=1)
    assert np.isclose(base.loc[before, "low_excursion_z_1m"], modified.loc[before, "low_excursion_z_1m"], equal_nan=True)
    assert modified.loc[panic, "low_excursion_z_1m"] > base.loc[panic, "low_excursion_z_1m"]


def test_plugin_uses_1m_analysis_while_displaying_15s() -> None:
    parent, panic = _bars()
    visible_parent = parent.loc[panic - pd.Timedelta(hours=2) : panic + pd.Timedelta(hours=2)]
    display_idx = pd.date_range(visible_parent.index.min(), visible_parent.index.max() + pd.Timedelta(seconds=45), freq="15s")
    display = visible_parent.reindex(display_idx, method="ffill").copy()
    display.index.name = "timestamp"
    plugin = PanicLowExcursionRejectionPlugin()
    context = PluginRunContext(
        display_df=display,
        visible_df=display,
        analysis_frames={"parent_1m": parent},
        request={"data_type": "trade_bar", "timeframe": "15s"},
    )
    result = plugin.run_with_context(context, {"preset": "core", "candidate_view": "all", "max_episodes": 20})
    assert result.summary["analysis_timeframe"] == "1m"
    assert result.summary["display_timeframe"] == "15s"
    assert result.summary["matched"] >= 1
    assert any(m.role == "detection" for m in result.markers)
    assert any(m.kind == "span" for m in result.markers)
    display_times = {ts.strftime("%Y-%m-%d %H:%M:%S") for ts in display.index}
    assert all(m.timestamp in display_times for m in result.markers)
