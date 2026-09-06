from __future__ import annotations

import pytest

from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore


def _open_trade(tmp_path, *, exit_reason: str, exit_price: float):
    store = ReplayStore(tmp_path / f"{exit_reason}.sqlite3")
    app = ReplayApplication(None, store)  # type: ignore[arg-type]
    episode = store.create_episode("ETH-USDT-SWAP", "2026-08-01 00:00:00")
    entry = store.add_event(episode.id, "LONG", episode.cursor_time, timeframe="1m", price=100.0, payload={"order_type": "limit"})
    config = app._trade_config(
        {"account_size": 10_000, "risk_pct": 1, "limit_fee_rate": 0.0002, "market_fee_rate": 0.0005, "market_slippage_rate": 0.0002},
        side="LONG", entry_price=100.0, stop_loss=99.0, order_type="limit",
    )
    app._trade_open_events(
        episode.id, entry, trade_id="trade-1", side="LONG", timeframe="1m", entry_price=100.0,
        order_type="limit", stop_loss=99.0, take_profit=102.0, order_id="order-1", fill_model="test", trade_config=config,
    )
    result = app._close_trade(
        episode.id, store.active_trades(episode.id)[0], exit_time="2026-08-01 00:05:00",
        exit_price=exit_price, exit_reason=exit_reason, trigger_bar=None, prior_bars=[],
    )[-1]
    return store, episode.id, result["payload"]


def test_stop_loss_uses_market_fee_and_adverse_slippage(tmp_path) -> None:
    store, episode_id, payload = _open_trade(tmp_path, exit_reason="STOP_LOSS", exit_price=99.0)
    assert payload["exit_order_type"] == "market"
    assert payload["exit_fee_rate"] == 0.0005
    assert payload["exit_price"] == 99.0 * (1 - 0.0002)
    assert payload["quantity"] == pytest.approx(100.0)
    assert payload["planned_risk_amount"] == pytest.approx(100.0)
    assert payload["risk_per_unit"] == pytest.approx(1.0)
    assert payload["planned_stop_net_loss"] == pytest.approx(108.92901)
    assert payload["raw_gross_pnl"] == pytest.approx(-100.0)
    assert payload["slippage_cost"] == pytest.approx(1.98)
    assert payload["total_fees"] == pytest.approx(6.94901)
    assert payload["net_pnl"] == pytest.approx(-108.92901)
    assert payload["risk_overrun_amount"] == pytest.approx(8.92901)
    assert payload["r_multiple"] == pytest.approx(-1.0892901)
    assert store.trade_summary(episode_id)["total_net_pnl"] == pytest.approx(-108.92901)


def test_take_profit_uses_limit_fee_without_slippage(tmp_path) -> None:
    _store, _episode_id, payload = _open_trade(tmp_path, exit_reason="TAKE_PROFIT", exit_price=102.0)
    assert payload["exit_order_type"] == "limit"
    assert payload["exit_fee_rate"] == 0.0002
    assert payload["exit_price"] == 102.0
    assert payload["quantity"] == pytest.approx(100.0)
    assert payload["net_pnl"] == pytest.approx(195.96)
    assert payload["r_multiple"] == pytest.approx(1.9596)


def test_market_entry_applies_default_adverse_slippage(tmp_path) -> None:
    app = ReplayApplication(None, ReplayStore(tmp_path / "market.sqlite3"))  # type: ignore[arg-type]
    assert app._market_execution_price("LONG", 100.0, 0.0002, phase="entry") == 100.02
    assert app._market_execution_price("SHORT", 100.0, 0.0002, phase="entry") == 99.98


def test_price_risk_sizing_replaces_a_legacy_cost_adjusted_quantity(tmp_path) -> None:
    app = ReplayApplication(None, ReplayStore(tmp_path / "legacy-quantity.sqlite3"))  # type: ignore[arg-type]
    config = app._trade_config(
        {
            "account_size": 10_000,
            "risk_pct": 1,
            "quantity": 7.0,
            "limit_fee_rate": 0.0002,
            "market_fee_rate": 0.0005,
            "market_slippage_rate": 0.0002,
        },
        side="LONG",
        entry_price=100.0,
        stop_loss=99.0,
        order_type="limit",
    )
    assert config["risk_per_unit"] == pytest.approx(1.0)
    assert config["quantity"] == pytest.approx(100.0)
    assert config["planned_stop_net_loss"] == pytest.approx(108.92901)
