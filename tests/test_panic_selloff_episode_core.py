from __future__ import annotations

import numpy as np
import pandas as pd

from research.liquidity.panic_selloff_rejection_recovery_long.panic_episode import (
    BaselineConfig,
    DEFAULT_PRESETS,
    EpisodeConfig,
    build_causal_panic_features,
    enrich_episode_paths,
    segment_panic_episodes,
)


def _synthetic_trade_bars(*, remove_gap: bool = False) -> tuple[pd.DataFrame, pd.Timestamp]:
    idx = pd.date_range("2026-01-01", periods=10 * 1440, freq="1min")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.00008, len(idx))
    panic_pos = 8 * 1440 + 600
    returns[panic_pos : panic_pos + 5] = [-0.0025, -0.0030, -0.0025, -0.0015, 0.0010]
    price = 2_000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[2_000.0, price[:-1]]
    high = np.maximum(open_, price) * 1.00005
    low = np.minimum(open_, price) * 0.99995
    notional = np.full(len(idx), 1_000_000.0)
    trades = np.full(len(idx), 100.0)
    buy = np.full(len(idx), 500_000.0)
    sell = np.full(len(idx), 500_000.0)
    notional[panic_pos : panic_pos + 5] = 8_000_000.0
    trades[panic_pos : panic_pos + 5] = 600.0
    buy[panic_pos : panic_pos + 5] = 1_000_000.0
    sell[panic_pos : panic_pos + 5] = 7_000_000.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": 100.0,
            "trades_count": trades,
            "buy_volume": 50.0,
            "sell_volume": 50.0,
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "buy_trades_count": 50.0,
            "sell_trades_count": 50.0,
            "delta_notional": buy - sell,
            "large_buy_notional": 0.0,
            "large_sell_notional": 0.0,
        },
        index=idx,
    )
    if remove_gap:
        df = df.drop(index=idx[panic_pos - 1])
    return df, idx[panic_pos]


def _baseline() -> BaselineConfig:
    return BaselineConfig(
        short_hours=24,
        long_hours=168,
        min_short_hours=8,
        min_long_hours=16,
        long_floor_ratio=0.5,
    )


def test_relative_panic_episode_is_detected_and_timing_is_causal() -> None:
    df, panic_time = _synthetic_trade_bars()
    built = build_causal_panic_features(df, baseline=_baseline())
    episodes = segment_panic_episodes(
        built.features,
        preset=DEFAULT_PRESETS["core"],
        episode_config=EpisodeConfig(quiet_minutes=10, max_episode_minutes=120),
        research_start="2026-01-07",
        research_end="2026-01-10",
    )
    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["detected_bar_time"] == panic_time
    assert episode["detected_time"] == panic_time + pd.Timedelta(minutes=1)
    assert "P1_PRICE_DISLOCATION" in episode["panic_subtypes"]
    assert "P3_AGGRESSIVE_SELL_SPEED" in episode["panic_subtypes"]
    assert episode["final_low_bar_time"] >= episode["detected_bar_time"]

    paths = enrich_episode_paths(episodes, built.features)
    assert paths.iloc[0]["entry_time"] == panic_time + pd.Timedelta(minutes=1)
    assert paths.iloc[0]["entry_is_next_open"]
    assert paths.iloc[0]["fp_up40_down20_result"] in {
        "up_first",
        "down_first",
        "ambiguous_same_bar",
        "unresolved",
    }


def test_gap_touch_disables_parent_detector() -> None:
    df, panic_time = _synthetic_trade_bars(remove_gap=True)
    built = build_causal_panic_features(df, baseline=_baseline())
    assert len(built.gap_runs) == 1
    assert not bool(built.features.loc[panic_time, "data_valid"])
    episodes = segment_panic_episodes(
        built.features,
        preset=DEFAULT_PRESETS["core"],
        episode_config=EpisodeConfig(),
        research_start=panic_time - pd.Timedelta(minutes=5),
        research_end=panic_time + pd.Timedelta(minutes=30),
    )
    assert episodes.empty


def test_current_bar_does_not_change_its_own_environment_scale() -> None:
    df, panic_time = _synthetic_trade_bars()
    base = build_causal_panic_features(df, baseline=_baseline()).features
    modified = df.copy()
    modified.loc[panic_time, "close"] *= 0.90
    modified.loc[panic_time, "low"] = min(modified.loc[panic_time, "low"], modified.loc[panic_time, "close"])
    changed = build_causal_panic_features(modified, baseline=_baseline()).features

    # The current shock changes its z numerator, but strict as-of alignment means
    # the historical scale available to the immediately following bar remains
    # unchanged until a completed hourly summary is legitimately available.
    before = panic_time - pd.Timedelta(minutes=1)
    assert np.isclose(base.loc[before, "price_shock_z_1m"], changed.loc[before, "price_shock_z_1m"], equal_nan=True)
    assert changed.loc[panic_time, "price_shock_z_1m"] > base.loc[panic_time, "price_shock_z_1m"]


def test_copy_on_write_read_only_trigger_mask_is_supported() -> None:
    """Regression: Pandas CoW exposes to_numpy() as a read-only view."""

    df, panic_time = _synthetic_trade_bars()
    with pd.option_context("mode.copy_on_write", True):
        built = build_causal_panic_features(df, baseline=_baseline())
        episodes = segment_panic_episodes(
            built.features,
            preset=DEFAULT_PRESETS["core"],
            episode_config=EpisodeConfig(quiet_minutes=10, max_episode_minutes=120),
            research_start="2026-01-07",
            research_end="2026-01-10",
        )
    assert len(episodes) == 1
    assert episodes.iloc[0]["detected_bar_time"] == panic_time


def test_analyze_plugin_emits_markers_under_copy_on_write() -> None:
    """The chart plugin must use the same fixed Edge-local detector."""

    import importlib.util
    from pathlib import Path

    plugin_path = Path(__file__).resolve().parents[1] / "analyze_tool" / "plugins" / "panic_selloff_episode.py"
    spec = importlib.util.spec_from_file_location("panic_selloff_episode_plugin_regression", plugin_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    df, _ = _synthetic_trade_bars()
    plugin = module.PanicSelloffEpisodePlugin()
    with pd.option_context("mode.copy_on_write", True):
        result = plugin.run(
            df,
            {
                "preset": "core",
                "_visible_start": "2026-01-07",
                "_visible_end": "2026-01-10",
                "max_episodes": 20,
            },
        )
    assert result.summary["episodes"] >= 1
    assert any(marker.role == "detection" for marker in result.markers)
    assert any(marker.kind == "span" for marker in result.markers)
