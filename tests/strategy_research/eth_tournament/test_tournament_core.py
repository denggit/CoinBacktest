from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy_research.eth_tournament.catalog import strategy_catalog
from src.strategy_research.eth_tournament.config import TournamentConfig
from src.strategy_research.eth_tournament.contracts import EntryEvent, PortfolioSelectionKey, StrategySignals
from src.strategy_research.eth_tournament.data import TournamentData, resample_trade_bars
from src.strategy_research.eth_tournament.engines import run_event_backtest
from src.strategy_research.eth_tournament.metrics import selection_key
from src.strategy_research.eth_tournament.runner import _validate_window


def minute_frame(start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min")
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1.0,
            "notional": price,
            "buy_notional": price / 2,
            "sell_notional": price / 2,
            "delta_notional": 0.0,
        },
        index=idx,
    )


def test_catalog_contains_eight_families_and_frozen_specs() -> None:
    specs = strategy_catalog()
    assert len({s.family_id for s in specs}) == 8
    assert len(specs) == 12
    assert {f"S{i:02d}" for i in range(1, 9)} == {s.family_id for s in specs}


def test_daily_resample_is_anchored_at_08_and_only_complete_days() -> None:
    df = minute_frame("2025-01-01 08:00:00", 1440 * 2)
    out = resample_trade_bars(df, "1D", 8)
    assert list(out.index) == [pd.Timestamp("2025-01-01 08:00:00"), pd.Timestamp("2025-01-02 08:00:00")]
    assert list(pd.to_datetime(out["available_time"])) == [pd.Timestamp("2025-01-02 08:00:00"), pd.Timestamp("2025-01-03 08:00:00")]
    assert (out["volume"] == 1440).all()


def test_event_engine_charges_exact_011pct_roundtrip_at_1x_flat_price() -> None:
    bars = minute_frame("2025-01-01 00:00:00", 5)
    cfg = TournamentConfig(
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:04:00",
        initial_capital=100_000.0,
        max_notional_leverage=1.0,
    )
    sig = StrategySignals(entries=[EntryEvent(pd.Timestamp("2025-01-01 00:00:00"), 1, max_hold_minutes=1)])
    result = run_event_backtest("cost_test", bars, sig, cfg)
    assert len(result.trades) == 1
    assert result.metrics["final_equity"] == pytest.approx(100_000.0 * (1.0 - 0.0011), abs=0.02)


def test_event_engine_stop_wins_if_stop_and_target_touch_same_minute() -> None:
    bars = minute_frame("2025-01-01 00:00:00", 5)
    bars.loc[pd.Timestamp("2025-01-01 00:02:00"), ["high", "low"]] = [102.0, 98.0]
    cfg = TournamentConfig(research_start="2025-01-01 00:00:00", research_end="2025-01-01 00:04:00", max_notional_leverage=1.0)
    sig = StrategySignals(entries=[EntryEvent(pd.Timestamp("2025-01-01 00:00:00"), 1, stop_distance=1.0, target_distance=1.0, max_hold_minutes=10)])
    result = run_event_backtest("ambiguity_test", bars, sig, cfg)
    assert result.trades.iloc[0]["exit_reason"] == "STOP"
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(99.0)


def test_entry_is_strictly_after_signal_time() -> None:
    bars = minute_frame("2025-01-01 00:00:00", 4)
    cfg = TournamentConfig(research_start="2025-01-01 00:00:00", research_end="2025-01-01 00:03:00")
    sig = StrategySignals(entries=[EntryEvent(pd.Timestamp("2025-01-01 00:01:00"), 1, max_hold_minutes=1)])
    result = run_event_backtest("causal_entry", bars, sig, cfg)
    assert pd.Timestamp(result.trades.iloc[0]["entry_time"]) == pd.Timestamp("2025-01-01 00:02:00")


def test_lexicographic_priority_matches_user_contract() -> None:
    a = {"max_flat_days": 2, "max_consecutive_losing_days": 5, "max_drawdown_pct": 19, "cagr_pct": 200, "total_return_pct": 1000}
    b = {"max_flat_days": 1, "max_consecutive_losing_days": 99, "max_drawdown_pct": 99, "cagr_pct": -10, "total_return_pct": -20}
    assert selection_key(b) < selection_key(a)
    c = dict(b, max_flat_days=1, max_consecutive_losing_days=3)
    assert selection_key(c) < selection_key(b)


def test_2026_is_hard_sealed() -> None:
    cfg = TournamentConfig(research_end="2026-01-01 00:00:00")
    with pytest.raises(ValueError, match="refuses to open sealed data"):
        _validate_window(cfg)
