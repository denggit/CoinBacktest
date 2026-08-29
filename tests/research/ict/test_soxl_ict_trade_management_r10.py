from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import make_synthetic_ict_day
from src.research_common.ict.trade_management import (
    TradeManagementConfig,
    _build_intermediate_pivots,
    build_management_structure_catalog,
    replay_trade_management,
    select_known_internal_target,
)


def _load_r10():
    path = Path(__file__).resolve().parents[3] / "research" / "ict" / "soxl_premarket_mss_fvg" / "10_trade_management_atlas.py"
    spec = importlib.util.spec_from_file_location("soxl_r10", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_intermediate_pivot_availability_waits_for_right_swing_confirmation():
    tz = "America/New_York"
    st = pd.DataFrame({
        "pivot_side": ["high", "high", "high"],
        "pivot_time": pd.to_datetime(["2026-06-02 09:30", "2026-06-02 09:40", "2026-06-02 09:50"]).tz_localize(tz),
        "pivot_price": [101.0, 103.0, 102.0],
        "confirmation_available_time": pd.to_datetime(["2026-06-02 09:32", "2026-06-02 09:42", "2026-06-02 09:52"]).tz_localize(tz),
        "hierarchy": ["ST", "ST", "ST"],
        "structure_tf": [1, 1, 1],
    })
    it = _build_intermediate_pivots(st)
    assert len(it) == 1
    assert it.iloc[0]["pivot_price"] == 103.0
    assert pd.Timestamp(it.iloc[0]["confirmation_available_time"]) == pd.Timestamp("2026-06-02 09:52", tz=tz)


def test_internal_target_must_be_known_at_fill_and_between_entry_and_main_target():
    tz = "America/New_York"
    structures = pd.DataFrame({
        "pivot_side": ["high", "high", "high"],
        "pivot_time": pd.to_datetime(["2026-06-02 09:10", "2026-06-02 09:20", "2026-06-02 09:25"]).tz_localize(tz),
        "pivot_price": [102.0, 104.0, 109.0],
        "confirmation_available_time": pd.to_datetime(["2026-06-02 09:12", "2026-06-02 09:35", "2026-06-02 09:27"]).tz_localize(tz),
        "hierarchy": ["ST", "ST", "ST"],
        "structure_tf": [5, 5, 5],
    })
    target = select_known_internal_target(structures, fill_time=pd.Timestamp("2026-06-02 09:30", tz=tz), entry=100.0, main_target=108.0, is_long=True, tf=5, hierarchy="ST")
    assert target is not None
    assert float(target["pivot_price"]) == 102.0  # 104 is not yet known; 109 is beyond main target.


def test_cost_cover_partial_formula_is_mechanistic_not_pnl_tuned():
    tz = "America/New_York"
    idx = pd.date_range(pd.Timestamp("2026-06-02 09:30", tz=tz), periods=20, freq="1min")
    price = np.linspace(100.0, 106.0, len(idx))
    bars = pd.DataFrame({"open": price, "high": price + 0.2, "low": price - 0.2, "close": price, "volume": 1.0}, index=idx)
    structures = pd.DataFrame({
        "pivot_side": ["high"], "pivot_time": [idx[0]], "pivot_price": [102.0],
        "confirmation_available_time": [idx[0]], "hierarchy": ["IT"], "structure_tf": [5],
    })
    trade = {
        "filled": True, "ny_date": "2026-06-02", "fill_time": idx[0], "entry_price": 100.0,
        "stop_price": 99.0, "target_price": 108.0, "trade_side": "LONG", "notional_multiple": 1.0,
        "exit_price": 108.0, "exit_reason": "opposite_premarket_extreme_target", "gross_return": 0.08,
        "net_return": 0.0789, "gross_r": 8.0, "net_r": 7.89, "account_return": 0.0789, "mfe_r": 8.0, "mae_r": -0.1,
    }
    out = replay_trade_management(bars, trade, structures, round_trip_cost=0.0011, scenario_name="internal_ith_cost_cover")
    # risk_pct=1%; cost_r=0.11R; target=+2R => fraction=(1+0.11)/(2+1)=0.37
    assert abs(float(out["internal_partial_fraction"]) - 0.37) < 1e-9


def test_r10_self_test_preserves_entries(tmp_path):
    mod = _load_r10()
    args = mod.parse_args(["--self-test", "--data-source", "alpaca", "--local-only", "--skip-review-pack", "--skip-platform-reports", "--no-progress"])
    bars = make_synthetic_ict_day()
    args.start_date = args.end_date = "2026-06-02"
    args.out_dir = str(tmp_path)
    args.include_us_equity_holidays = True
    args.required_day_coverage = 1.0
    result = mod.run_research(bars, args)
    pres = pd.read_csv(Path(result["report_dir"]) / "43_management_entry_preservation_audit.csv")
    assert not pres.empty
    assert pres["same_entry_universe"].all()
    managed = pd.read_csv(Path(result["report_dir"]) / "39_trade_management_lifecycle.csv")
    base = managed.loc[(managed["management_scenario"] == "baseline") & (managed["management_cost_multiple"] == 1.0)]
    lifecycle = pd.read_csv(Path(result["report_dir"]) / "10_base_trade_lifecycle.csv")
    assert set(base["attempt_id"].astype(str)) == set(lifecycle.loc[lifecycle["filled"], "attempt_id"].astype(str))
    merged = base[["attempt_id", "management_net_return"]].merge(lifecycle.loc[lifecycle["filled"], ["attempt_id", "net_return"]], on="attempt_id", how="inner")
    assert np.allclose(merged["management_net_return"], merged["net_return"], atol=1e-12, rtol=0.0)
