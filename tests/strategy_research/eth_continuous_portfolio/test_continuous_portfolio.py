from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy_research.eth_continuous_portfolio.config import ContinuousPortfolioConfig, PortfolioSpec
from src.strategy_research.eth_continuous_portfolio.data import ContinuousPortfolioData, resample_causal
from src.strategy_research.eth_continuous_portfolio.engine import run_continuous_backtest
from src.strategy_research.eth_continuous_portfolio.runner import _validate
from src.strategy_research.eth_continuous_portfolio.signals import build_raw_target, build_sleeves


def _minute_frame(start: str, periods: int, price_step: float = 0.01) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min")
    px = 100.0 + np.arange(periods) * price_step
    return pd.DataFrame({"open": px, "high": px + 0.02, "low": px - 0.02, "close": px + 0.01}, index=idx)


def test_daily_resample_is_anchored_at_08_and_available_next_08() -> None:
    frame = _minute_frame("2025-01-01 08:00:00", 1440)
    out = resample_causal(frame, "1D", 8)
    assert list(out.index) == [pd.Timestamp("2025-01-01 08:00:00")]
    assert out.iloc[0]["available_time"] == pd.Timestamp("2025-01-02 08:00:00")


def test_4h_context_available_only_after_close() -> None:
    frame = _minute_frame("2025-01-01 00:00:00", 240)
    out = resample_causal(frame, "4h", 8)
    assert out.iloc[0]["available_time"] == pd.Timestamp("2025-01-01 04:00:00")


def test_execution_is_strictly_after_signal_time() -> None:
    bars = _minute_frame("2025-01-01 00:00:00", 10, price_step=0.0)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:09:00",
        sealed_start="2026-01-01",
    )
    spec = PortfolioSpec("X", "x", max_abs_exposure=1.5, deadband=0.0)
    schedule = pd.DataFrame({"raw_target": [1.0]}, index=[pd.Timestamp("2025-01-01 00:02:00")])
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    assert r.rebalances.iloc[0]["execution_time"] == pd.Timestamp("2025-01-01 00:03:00")
    assert r.position.loc["2025-01-01 00:02:00"] == 0.0
    assert r.position.loc["2025-01-01 00:03:00"] == 1.0


def test_round_trip_cost_semantics_on_flip() -> None:
    bars = _minute_frame("2025-01-01 00:00:00", 10, price_step=0.0)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:09:00",
        sealed_start="2026-01-01",
        round_trip_cost=0.0011,
    )
    spec = PortfolioSpec("X", "x", max_abs_exposure=2.0, deadband=0.0)
    schedule = pd.DataFrame(
        {"raw_target": [1.0, -1.0]},
        index=[pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-01-01 00:04:00")],
    )
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    rows = r.rebalances
    assert rows.iloc[0]["turnover"] == pytest.approx(1.0)
    assert rows.iloc[0]["fee_fraction"] == pytest.approx(0.00055)
    assert rows.iloc[1]["turnover"] == pytest.approx(2.0)
    assert rows.iloc[1]["fee_fraction"] == pytest.approx(0.0011)


def test_deadband_prevents_small_rebalance() -> None:
    bars = _minute_frame("2025-01-01 00:00:00", 10, price_step=0.0)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:09:00",
        sealed_start="2026-01-01",
    )
    spec = PortfolioSpec("X", "x", deadband=0.10)
    schedule = pd.DataFrame(
        {"raw_target": [0.50, 0.55]},
        index=[pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-01-01 00:04:00")],
    )
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    assert r.rebalances.iloc[1]["position_after"] == pytest.approx(0.50)
    assert r.rebalances.iloc[1]["turnover"] == pytest.approx(0.0)


def test_rebalance_step_cap_smooths_position_change() -> None:
    bars = _minute_frame("2025-01-01 00:00:00", 10, price_step=0.0)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:09:00",
        sealed_start="2026-01-01",
    )
    spec = PortfolioSpec("X", "x", deadband=0.0, max_rebalance_step=0.5)
    schedule = pd.DataFrame({"raw_target": [1.5]}, index=[pd.Timestamp("2025-01-01 00:00:00")])
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    assert r.rebalances.iloc[0]["position_after"] == pytest.approx(0.5)


def test_single_net_position_never_creates_gross_hedge_legs() -> None:
    bars = _minute_frame("2025-01-01 00:00:00", 10, price_step=0.0)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:09:00",
        sealed_start="2026-01-01",
    )
    spec = PortfolioSpec("X", "x", deadband=0.0)
    schedule = pd.DataFrame(
        {"raw_target": [0.8, -0.4]},
        index=[pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-01-01 00:04:00")],
    )
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    assert r.audit["dual_exchange_side_positions"] is False
    assert r.audit["execution_semantics"] == "single_net_eth_exposure"
    assert r.position.abs().max() <= spec.max_abs_exposure


def test_sealed_2026_window_is_rejected() -> None:
    cfg = ContinuousPortfolioConfig(research_end="2026-01-01 00:00:00", sealed_start="2026-01-01 00:00:00")
    with pytest.raises(ValueError, match="refuses to open sealed data"):
        _validate(cfg)


def test_vol_target_is_hard_capped() -> None:
    idx = pd.date_range("2025-01-01", periods=2, freq="1D")
    sleeves = pd.DataFrame(
        {
            "raw_signal": [1.0, -1.0],
            "realized_vol": [0.01, 0.01],
            "channel_family": [1, -1],
            "ma_family": [1, -1],
            "tsmom_family": [1, -1],
            "intraday_family": [1, -1],
        },
        index=idx,
    )
    spec = PortfolioSpec("X", "x", volatility_target=0.25, max_abs_exposure=1.5)
    out = build_raw_target(sleeves, spec)
    assert out["raw_target"].tolist() == [1.5, -1.5]


def test_sleeve_ensemble_stays_bounded_and_uses_four_equal_families() -> None:
    # Inject already-closed bars so this test focuses on signal construction.
    d_idx = pd.date_range("2023-01-01 08:00:00", periods=500, freq="1D")
    d_close = np.linspace(100.0, 300.0, len(d_idx))
    d = pd.DataFrame(
        {
            "open": d_close - 0.2,
            "high": d_close + 1.0,
            "low": d_close - 1.0,
            "close": d_close,
            "available_time": d_idx + pd.Timedelta(days=1),
        },
        index=d_idx,
    )
    h_idx = pd.date_range("2023-01-01 00:00:00", periods=3000, freq="4h")
    h_close = np.linspace(100.0, 250.0, len(h_idx))
    h4 = pd.DataFrame(
        {
            "open": h_close - 0.1,
            "high": h_close + 0.5,
            "low": h_close - 0.5,
            "close": h_close,
            "available_time": h_idx + pd.Timedelta(hours=4),
        },
        index=h_idx,
    )
    cfg = ContinuousPortfolioConfig()
    data = ContinuousPortfolioData(cfg=cfg, one_minute=pd.DataFrame(), cache={"1D": d, "4H": h4})
    sleeves = build_sleeves(data)
    assert {"channel_family", "ma_family", "tsmom_family", "intraday_family", "raw_signal", "realized_vol"}.issubset(sleeves.columns)
    assert sleeves["raw_signal"].abs().max() <= 1.0
    last = sleeves.iloc[-1]
    expected = np.mean([last["channel_family"], last["ma_family"], last["tsmom_family"], last["intraday_family"]])
    assert last["raw_signal"] == pytest.approx(expected)


def test_drawdown_governor_reduces_later_target_after_loss() -> None:
    idx = pd.date_range("2025-01-01 00:00:00", periods=12, freq="1min")
    opens = np.array([100, 100, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90], dtype=float)
    bars = pd.DataFrame({"open": opens, "high": opens, "low": opens, "close": opens}, index=idx)
    cfg = ContinuousPortfolioConfig(
        warmup_start="2025-01-01",
        research_start="2025-01-01 00:00:00",
        research_end="2025-01-01 00:11:00",
        sealed_start="2026-01-01",
        round_trip_cost=0.0,
    )
    spec = PortfolioSpec("X", "x", deadband=0.0, use_drawdown_governor=True)
    schedule = pd.DataFrame(
        {"raw_target": [1.0, 1.0]},
        index=[pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-01-01 00:04:00")],
    )
    r = run_continuous_backtest(bars, schedule, cfg, spec)
    assert r.rebalances.iloc[1]["drawdown_before"] >= 0.05
    assert r.rebalances.iloc[1]["governor"] < 1.0
    assert r.rebalances.iloc[1]["position_after"] < 1.0
