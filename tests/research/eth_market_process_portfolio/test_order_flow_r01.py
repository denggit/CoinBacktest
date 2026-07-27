from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "eth_market_process_portfolio" / "order_flow" / "01_order_flow_process_event_study.py"
spec = importlib.util.spec_from_file_location("order_flow_r01", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _frame(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = np.full(n, 2000.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "notional_ratio_base": 1.0,
            "buy_notional_ratio_base": 1.0,
            "sell_notional_ratio_base": 1.0,
            "trades_ratio_base": 1.0,
            "delta_ratio": 0.0,
            "delta_ratio_3": 0.0,
            "large_delta_ratio_3": 0.0,
            "price_return_3": 0.0,
            "close_pos": 0.5,
            "lower_wick_frac": 0.0,
            "absorption_score": 0.0,
            "large_trade_share": 0.0,
            "large_sell_share_of_sell": 0.0,
            "buy_notional": 100.0,
            "large_buy_notional": 0.0,
            "up_move_norm": 0.0,
            "down_move_norm": 0.0,
        },
        index=idx,
    )
    return frame


def test_cooldown_keeps_first_event_only_inside_window() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1min")
    mask = pd.Series([False, True, True, False, True, False, False, True, False, False], index=idx)
    kept = module._apply_cooldown(mask, 3)
    assert list(np.flatnonzero(kept.to_numpy())) == [1, 4, 7]


def test_next_open_outcome_deducts_cost() -> None:
    bars = _frame(20)
    bars.iloc[2, bars.columns.get_loc("open")] = 100.0
    bars.iloc[6, bars.columns.get_loc("close")] = 101.0
    events = pd.DataFrame({"signal_time": [bars.index[1]], "process": ["x"], "side": [1]})
    out = module._attach_outcomes(events, bars, (5,), 0.0011)
    assert out.loc[0, "entry_time"] == bars.index[2]
    assert abs(out.loc[0, "ret_h5_gross"] - 0.01) < 1e-12
    assert abs(out.loc[0, "ret_h5_net"] - 0.0089) < 1e-12
    assert bool(out.loc[0, "causal_entry_flag"])


def test_process_masks_detect_declared_buy_continuation() -> None:
    frame = _frame(300)
    pos = 260
    frame.iloc[pos, frame.columns.get_loc("buy_notional")] = 500.0
    frame.iloc[pos, frame.columns.get_loc("large_buy_notional")] = 100.0
    frame.iloc[pos, frame.columns.get_loc("delta_ratio_3")] = 0.30
    frame.iloc[pos, frame.columns.get_loc("large_delta_ratio_3")] = 0.20
    frame.iloc[pos, frame.columns.get_loc("price_return_3")] = 0.002
    frame.iloc[pos, frame.columns.get_loc("close_pos")] = 0.80
    frame.iloc[pos, frame.columns.get_loc("notional_ratio_base")] = 2.0
    masks = module.build_process_masks(frame)
    assert bool(masks["buy_continuation_long"].iloc[pos])
