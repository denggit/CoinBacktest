from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy_research.eth_tournament.catalog import strategy_catalog
from src.strategy_research.eth_tournament.config import TournamentConfig
from src.strategy_research.eth_tournament.data import TournamentData
from src.strategy_research.eth_tournament.strategies import s01_donchian, s02_ma, s03_bollinger, s05_absorption, s08_quarter_hour


def _empty_base() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=3, freq="1min")
    return pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "notional": 1.0, "buy_notional": 0.5, "sell_notional": 0.5, "delta_notional": 0.0}, index=idx)


def _spec(sid: str):
    return next(s for s in strategy_catalog() if s.strategy_id == sid)


def test_s01_target_is_indexed_by_daily_available_time() -> None:
    n = 500
    idx = pd.date_range("2023-01-01 08:00", periods=n, freq="1D")
    close = np.linspace(100, 300, n)
    d = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "available_time": idx + pd.Timedelta(days=1)}, index=idx)
    data = TournamentData(TournamentConfig(), _empty_base(), {"bars:1D": d})
    target = s01_donchian.build_target(data, _spec("s01_donchian_ensemble_long"))
    assert target.index.equals(pd.DatetimeIndex(d["available_time"]))
    assert (target.dropna() >= 0).all()


def test_s02_target_is_causal_and_bounded() -> None:
    n = 120
    idx = pd.date_range("2024-01-01 08:00", periods=n, freq="1D")
    close = np.r_[np.linspace(100, 150, 60), np.linspace(150, 90, 60)]
    d = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "available_time": idx + pd.Timedelta(days=1)}, index=idx)
    data = TournamentData(TournamentConfig(), _empty_base(), {"bars:1D": d})
    target = s02_ma.build_target(data, _spec("s02_ma20_50_voltrend"))
    assert target.abs().max() <= 2.0
    assert target.index.equals(pd.DatetimeIndex(d["available_time"]))


def test_s03_bollinger_entry_uses_available_time_not_bar_start() -> None:
    idx = pd.date_range("2025-01-01", periods=30, freq="1h")
    close = np.full(30, 100.0)
    close[-1] = 90.0
    b = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "available_time": idx + pd.Timedelta(hours=1)}, index=idx)
    data = TournamentData(TournamentConfig(research_start="2025-01-01", research_end="2025-02-01"), _empty_base(), {"bars:1h": b})
    sig = s03_bollinger.build_signals(data, _spec("s03_bb_rsi_mr_1h"))
    if sig.entries:
        assert sig.entries[-1].signal_time == pd.Timestamp(b["available_time"].iloc[-1])


def test_s05_absorption_uses_past_only_threshold() -> None:
    n = 220
    t = pd.date_range("2025-01-01", periods=n, freq="5min")
    lower = np.full(n, -0.2)
    lower[-1] = -0.95
    f = pd.DataFrame({
        "bar_id": np.arange(n), "end_ts": t, "available_time": t,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.6,
        "notional": 1000.0, "delta_notional": -100.0,
        "lower_delta_ratio": lower, "upper_delta_ratio": 0.1,
        "close_pos": 0.8, "total_delta_ratio": -0.1,
    }, index=t)
    data = TournamentData(TournamentConfig(research_start="2025-01-01", research_end="2025-02-01"), _empty_base(), {"absorption_features": f})
    sig = s05_absorption.build_signals(data, _spec("s05_footprint_absorption"))
    assert sig.entries
    assert sig.entries[-1].signal_time == t[-1]
    assert sig.entries[-1].side == 1


def test_s08_signal_uses_only_first10s_available_time_and_past_zscore() -> None:
    n = 1000
    qidx = pd.date_range("2025-01-01", periods=n, freq="15min")
    imbalance = 0.05 * np.sin(np.arange(n) / 11.0)
    imbalance[-1] = 0.9
    q = pd.DataFrame({"imbalance": imbalance, "available_time": qidx + pd.Timedelta(seconds=10)}, index=qidx)
    bidx = qidx
    close = np.full(n, 100.0)
    b15 = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "available_time": bidx + pd.Timedelta(minutes=15)}, index=bidx)
    cfg = TournamentConfig(research_start="2025-01-01", research_end="2025-12-31")
    data = TournamentData(cfg, _empty_base(), {"quarter_hour_opening_imbalance": q, "bars:15min": b15})
    sig = s08_quarter_hour.build_signals(data, _spec("s08_quarter_hour_oi_8h"))
    assert sig.entries
    assert sig.entries[-1].signal_time == qidx[-1] + pd.Timedelta(seconds=10)
    assert sig.entries[-1].metadata["imbalance_z"] > 1.5


def test_s06_cvd_divergence_emits_reclaim_long() -> None:
    from src.strategy_research.eth_tournament.strategies import s06_cvd
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    delta = np.full(n, -1.0)
    low[-1] = 98.0
    close[-1] = 100.0
    delta[-1] = 10.0
    b = pd.DataFrame({"open": 100.0, "high": high, "low": low, "close": close, "delta_notional": delta, "available_time": idx + pd.Timedelta(minutes=15)}, index=idx)
    data = TournamentData(TournamentConfig(research_start="2025-01-01", research_end="2025-01-02"), _empty_base(), {"bars:15min": b})
    sig = s06_cvd.build_signals(data, _spec("s06_cvd_exhaustion"))
    assert any(e.side == 1 for e in sig.entries)


def test_s07_flow_breakout_requires_price_and_flow_confirmation() -> None:
    from src.strategy_research.eth_tournament.strategies import s07_flow_breakout
    n = 800
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    imbalance = 0.05 * np.sin(np.arange(n) / 7.0)
    close[-1] = 102.0
    high[-1] = 102.5
    imbalance[-1] = 0.9
    # Reconstruct buy/sell notionals consistent with desired imbalance.
    total = np.full(n, 1000.0)
    buy = total * (1 + imbalance) / 2
    sell = total - buy
    b = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "notional": total, "buy_notional": buy, "sell_notional": sell, "delta_notional": buy-sell, "flow_imbalance": imbalance, "available_time": idx + pd.Timedelta(minutes=15)}, index=idx)
    data = TournamentData(TournamentConfig(research_start="2025-01-01", research_end="2025-02-01"), _empty_base(), {"bars:15min": b})
    sig = s07_flow_breakout.build_signals(data, _spec("s07_flow_confirmed_breakout"))
    assert sig.entries
    assert sig.entries[-1].side == 1


def test_turtle_system2_triggers_breakout_from_prior_daily_context() -> None:
    from src.strategy_research.eth_tournament.strategies.s04_turtle import run_turtle_system2
    n = 80
    didx = pd.date_range("2025-01-01 08:00", periods=n, freq="1D")
    d = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "available_time": didx + pd.Timedelta(days=1),
    }, index=didx)
    start = didx[-1] + pd.Timedelta(days=1, minutes=1)
    midx = pd.date_range(start, periods=6, freq="1min")
    one = pd.DataFrame({
        "open": [100, 100, 101.2, 101.2, 101.2, 101.2],
        "high": [100.5, 102.0, 101.5, 101.5, 101.5, 101.5],
        "low": [99.8, 100.5, 101.0, 101.0, 101.0, 101.0],
        "close": [100.2, 101.2, 101.2, 101.2, 101.2, 101.2],
        "notional": 1.0, "buy_notional": 0.5, "sell_notional": 0.5, "delta_notional": 0.0,
    }, index=midx)
    cfg = TournamentConfig(research_start=str(midx[0]), research_end=str(midx[-1]))
    data = TournamentData(cfg, one, {"bars:1D": d})
    result = run_turtle_system2(data, _spec("s04_turtle_system2"), cfg)
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["side"] == 1
    assert result.trades.iloc[0]["entry_price"] >= 101.0
