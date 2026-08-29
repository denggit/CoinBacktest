from __future__ import annotations

import pandas as pd
import pytest

from src.strategy_research.eth_turtle_path_atlas.atlas import build_path_tables, grouped_episode_stats, parse_episodes
from src.strategy_research.eth_turtle_path_atlas.config import TurtlePathConfig
from src.strategy_research.eth_turtle_path_atlas.runner import _validate


def _context() -> pd.DataFrame:
    return pd.DataFrame({"available_time": pd.to_datetime(["2023-01-01 08:00", "2023-01-02 08:00"]), "N": [10.0, 20.0]})


def test_strict_context_uses_only_available_time_before_entry() -> None:
    events = pd.DataFrame([
        {"event": "ENTRY", "time": "2023-01-02 08:00", "side": 1, "price": 100.0, "units": 1, "reason": "55D_BREAKOUT", "pnl": None},
        {"event": "EXIT", "time": "2023-01-02 08:02", "side": 1, "price": 101.0, "units": 1, "reason": "STOP", "pnl": 100.0},
    ])
    idx = pd.date_range("2023-01-02 07:59", periods=4, freq="1min")
    equity = pd.Series([100000.0] * 4, index=idx)
    eps = parse_episodes(events, equity, _context())
    assert eps[0].n_value == 10.0


def test_long_path_mfe_mae_are_directional_and_in_n() -> None:
    idx = pd.date_range("2023-01-02 08:00", periods=3, freq="1min")
    bars = pd.DataFrame({"open": [100, 101, 102], "high": [101, 104, 103], "low": [99, 98, 101], "close": [100.5, 103, 102]}, index=idx)
    events = pd.DataFrame([
        {"event": "ENTRY", "time": idx[0], "side": 1, "price": 100.0, "units": 1, "reason": "55D_BREAKOUT", "pnl": None},
        {"event": "EXIT", "time": idx[-1], "side": 1, "price": 102.0, "units": 1, "reason": "20D_EXIT", "pnl": 200.0},
    ])
    equity = pd.Series([100000, 100100, 100200], index=idx)
    episodes, checkpoints, adds = build_path_tables(bars, events, equity, _context(), discovery_end="2024-12-31", checkpoints_minutes=(1, 2))
    row = episodes.iloc[0]
    assert row.mfe_n == pytest.approx(0.4)
    assert row.mae_n == pytest.approx(0.2)
    assert row.final_move_n == pytest.approx(0.2)
    assert row.giveback_from_mfe_n == pytest.approx(0.2)
    assert len(checkpoints) == 2
    assert adds.empty


def test_short_path_uses_low_as_favorable_and_high_as_adverse() -> None:
    idx = pd.date_range("2023-01-02 08:00", periods=2, freq="1min")
    bars = pd.DataFrame({"open": [100, 98], "high": [102, 101], "low": [97, 95], "close": [98, 96]}, index=idx)
    events = pd.DataFrame([
        {"event": "ENTRY", "time": idx[0], "side": -1, "price": 100.0, "units": 1, "reason": "55D_BREAKOUT", "pnl": None},
        {"event": "EXIT", "time": idx[-1], "side": -1, "price": 96.0, "units": 1, "reason": "20D_EXIT", "pnl": 400.0},
    ])
    equity = pd.Series([100000, 100400], index=idx)
    episodes, _, _ = build_path_tables(bars, events, equity, _context(), discovery_end="2024-12-31", checkpoints_minutes=(1,))
    row = episodes.iloc[0]
    assert row.mfe_n == pytest.approx(0.5)
    assert row.mae_n == pytest.approx(0.2)
    assert row.final_move_n == pytest.approx(0.4)


def test_add_stages_are_preserved_without_changing_trade_rules() -> None:
    idx = pd.date_range("2023-01-02 08:00", periods=4, freq="1min")
    bars = pd.DataFrame({"open": [100, 105, 110, 115], "high": [101, 106, 111, 116], "low": [99, 104, 109, 114], "close": [100, 105, 110, 115]}, index=idx)
    events = pd.DataFrame([
        {"event": "ENTRY", "time": idx[0], "side": 1, "price": 100.0, "units": 1, "reason": "55D_BREAKOUT", "pnl": None},
        {"event": "ADD", "time": idx[1], "side": 1, "price": 105.0, "units": 2, "reason": "PLUS_0.5N", "pnl": None},
        {"event": "ADD", "time": idx[2], "side": 1, "price": 110.0, "units": 3, "reason": "PLUS_0.5N", "pnl": None},
        {"event": "EXIT", "time": idx[3], "side": 1, "price": 115.0, "units": 3, "reason": "20D_EXIT", "pnl": 1500.0},
    ])
    equity = pd.Series([100000, 100500, 101000, 101500], index=idx)
    episodes, _, adds = build_path_tables(bars, events, equity, _context(), discovery_end="2024-12-31", checkpoints_minutes=(1,))
    assert int(episodes.iloc[0].max_units) == 3
    assert list(adds.unit_reached) == [2, 3]


def test_group_stats_split_discovery_validation() -> None:
    episodes = pd.DataFrame([
        {"split": "DISCOVERY_2023_2024", "side": "LONG", "max_units": 1, "pnl": -10, "pnl_pct_entry_equity": -1, "mfe_n": .2, "mae_n": 2, "giveback_from_mfe_n": .4, "duration_hours": 5},
        {"split": "VALIDATION_2025", "side": "LONG", "max_units": 4, "pnl": 20, "pnl_pct_entry_equity": 2, "mfe_n": 5, "mae_n": .5, "giveback_from_mfe_n": 2, "duration_hours": 50},
    ])
    stats = grouped_episode_stats(episodes)
    assert set(stats.group) >= {"ALL", "DISCOVERY_2023_2024", "VALIDATION_2025", "LONG", "MAX_UNIT_1", "MAX_UNIT_4"}


def test_sealed_2026_is_rejected() -> None:
    with pytest.raises(ValueError):
        _validate(TurtlePathConfig(research_end="2026-01-01 00:00:00"))


def test_runner_uses_backtest_result_minute_equity_contract(tmp_path, monkeypatch) -> None:
    import types
    import src.strategy_research.eth_turtle_path_atlas.runner as runner_mod

    idx = pd.date_range("2023-01-01 00:00", periods=3, freq="1min")
    one_minute = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0], "high": [101.0, 101.0, 101.0], "low": [99.0, 99.0, 99.0], "close": [100.0, 100.0, 100.0]},
        index=idx,
    )
    daily = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2022-12-31 08:00")]),
    )
    data = types.SimpleNamespace(one_minute=one_minute, daily=lambda: daily)
    minute_equity = pd.Series([100000.0, 100000.0, 100000.0], index=idx)
    baseline = types.SimpleNamespace(
        events=pd.DataFrame(columns=["event", "time", "side", "price", "units", "reason", "pnl"]),
        minute_equity=minute_equity,
        metrics={"total_return_pct": 0.0},
        audit={"future_visibility_violations": 0},
    )

    monkeypatch.setattr(runner_mod, "load_data", lambda cfg: data)
    monkeypatch.setattr(runner_mod, "build_turtle_context", lambda frame: pd.DataFrame({"available_time": [], "N": []}))
    monkeypatch.setattr(runner_mod, "run_turtle_system2", lambda bars, context, cfg: baseline)

    captured = {}
    def fake_build_path_tables(bars, events, equity, context, **kwargs):
        captured["equity"] = equity
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(runner_mod, "build_path_tables", fake_build_path_tables)
    monkeypatch.setattr(runner_mod, "grouped_episode_stats", lambda episodes: pd.DataFrame())
    monkeypatch.setattr(runner_mod, "checkpoint_outcome_stats", lambda checkpoints: pd.DataFrame())
    monkeypatch.setattr(runner_mod, "write_decision", lambda *args, **kwargs: None)
    fake_pack = tmp_path / "gpt_review_pack.zip"
    fake_pack.write_bytes(b"")
    monkeypatch.setattr(runner_mod, "write_review_pack", lambda root: fake_pack)

    cfg = TurtlePathConfig(report_root=tmp_path / "r04")
    runner_mod.run_turtle_path_atlas(cfg)
    assert captured["equity"] is minute_equity
