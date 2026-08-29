from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r27 import (
    R27Config,
    build_sequential_state_rows,
    causal_audit,
    prepare_root_universe,
    summarize_state_progression,
)


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=180, freq="min")
    close = np.full(len(idx), 100.0)
    open_ = np.full(len(idx), 100.0)
    high = np.full(len(idx), 100.10)
    low = np.full(len(idx), 99.90)

    # SSL root at 01:10, then reclaim, impulse/pullback, MSS/displacement FVG,
    # proximal fill, protected pivot, and target delivery.
    open_[70], high[70], low[70], close[70] = 100.0, 100.1, 98.0, 99.4
    open_[71], high[71], low[71], close[71] = 99.4, 100.3, 99.3, 100.2
    close[72:75] = [100.7, 101.2, 101.6]
    open_[72:75] = [100.2, 100.7, 101.2]
    high[72:75] = [100.8, 101.3, 101.7]
    low[72:75] = [100.1, 100.6, 101.1]
    open_[75], high[75], low[75], close[75] = 101.6, 102.0, 101.5, 101.8
    open_[76], high[76], low[76], close[76] = 101.8, 101.9, 101.3, 101.5
    open_[77], high[77], low[77], close[77] = 101.5, 101.7, 101.1, 101.3
    open_[78], high[78], low[78], close[78] = 101.3, 101.6, 100.8, 101.0
    open_[79], high[79], low[79], close[79] = 101.0, 101.4, 100.7, 101.1
    open_[80], high[80], low[80], close[80] = 101.1, 101.3, 100.5, 101.0
    open_[81], high[81], low[81], close[81] = 101.0, 101.85, 100.8, 101.6
    open_[82], high[82], low[82], close[82] = 101.6, 101.95, 101.3, 101.8
    open_[83], high[83], low[83], close[83] = 102.30, 102.90, 102.25, 102.80
    open_[84], high[84], low[84], close[84] = 102.5, 102.7, 102.2, 102.4
    open_[85], high[85], low[85], close[85] = 102.4, 102.6, 101.8, 102.0
    open_[86], high[86], low[86], close[86] = 102.0, 102.2, 101.0, 101.6
    open_[87], high[87], low[87], close[87] = 101.6, 102.0, 101.2, 101.8
    open_[88], high[88], low[88], close[88] = 101.8, 102.4, 101.4, 102.2
    open_[89], high[89], low[89], close[89] = 102.9, 103.4, 102.8, 103.2
    open_[90:101], high[90:101], low[90:101], close[90:101] = 103.2, 103.5, 102.5, 103.3
    high[100], close[100] = 105.2, 105.0
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _root() -> pd.DataFrame:
    t = pd.Timestamp("2024-01-01 01:10")
    return pd.DataFrame([{
        "root_event_id": "r1", "root_sweep_time": t,
        "root_sweep_available_time": t + pd.Timedelta(minutes=1), "root_side": "SSL",
        "root_zone_low": 99.7, "root_zone_high": 99.8,
        "root_bar_open": 100.0, "root_bar_high": 100.1,
        "root_bar_low": 98.0, "root_bar_close": 99.4,
        "opposite_1_touch_price": 105.0, "direct_reversal_label": 1,
        "comparison_class": "direct_reversal", "path_outcome": "direct_opposite_delivery",
    }])


def _run(bars: pd.DataFrame | None = None):
    roots = prepare_root_universe(_root(), split="discovery")
    b = _bars() if bars is None else bars
    return build_sequential_state_rows(b, roots, physical_end=b.index[-1])


def test_complete_ordered_path_reaches_s6_and_executes_causally():
    rows, diag = _run()
    reached = rows.loc[rows["state_reached"].eq(1)].sort_values("state_id")
    assert reached["state_id"].tolist() == list(range(7))
    assert pd.to_datetime(reached["available_time"]).is_monotonic_increasing
    assert (pd.to_datetime(reached["entry_time"]) >= pd.to_datetime(reached["available_time"])).all()
    s5 = reached.loc[reached["state_id"].eq(5)].iloc[0]
    assert s5["order_type"] == "limit"
    assert s5["entry_price"] == s5["fvg_high"]
    assert s5["outcome"] == "tp_first"
    s6 = reached.loc[reached["state_id"].eq(6)].iloc[0]
    assert s6["protected_stop_status"] == "filled"
    assert s6["protected_stop_outcome"] == "tp_first"
    audit = causal_audit(rows, diag)
    assert int(audit["violations"].sum()) == 0


def test_three_outside_closes_are_failed_acceptance_and_block_s1_plus_later_states():
    bars = _bars()
    bars.loc[bars.index[71:74], "close"] = 99.5
    rows, diag = _run(bars)
    assert rows.loc[rows["state_reached"].eq(1), "state_id"].tolist() == [0]
    assert int(diag.iloc[0]["s1_failed_acceptance"]) == 1


def test_later_mutation_cannot_change_states_already_available():
    rows_a, _ = _run()
    bars = _bars()
    bars.loc[bars.index[95]:, ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]
    rows_b, _ = _run(bars)
    cols = ["state_id", "state_reached", "available_time", "entry_time"]
    a = rows_a.loc[rows_a["state_id"].le(4), cols].reset_index(drop=True)
    b = rows_b.loc[rows_b["state_id"].le(4), cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_limit_fill_bar_has_no_target_credit_and_is_stop_first():
    bars = _bars()
    # The fill bar spans both target and the common buffered sweep stop.  Price
    # must cross the limit on its way to the stop, so it is a fill + SL, not a
    # cancelled order or target winner.
    bars.loc[bars.index[84], ["high", "low"]] = [106.0, 97.0]
    rows, _ = _run(bars)
    s5 = rows.loc[rows["state_id"].eq(5)].iloc[0]
    assert s5["state_reached"] == 1
    assert s5["entry_status"] == "filled"
    assert s5["outcome"] == "sl_first"


def test_summary_keeps_state_denominators_and_cost_stress():
    rows, _ = _run()
    summary = summarize_state_progression(rows)
    overall = summary.loc[summary["grain"].eq("overall")].sort_values("state_id")
    assert overall["eligible_roots"].eq(1).all()
    assert overall["reached"].eq(1).all()
    assert overall["expectancy_cost2x"].notna().all()


def test_validation_roots_are_physically_selected_without_holdout():
    base = _root()
    validation = base.copy()
    validation["root_event_id"] = "v1"
    validation["root_sweep_time"] = pd.Timestamp("2025-02-01")
    validation["root_sweep_available_time"] = pd.Timestamp("2025-02-01 00:01")
    holdout = validation.copy()
    holdout["root_event_id"] = "h1"
    holdout["root_sweep_time"] = pd.Timestamp("2025-08-01")
    holdout["root_sweep_available_time"] = pd.Timestamp("2025-08-01 00:01")
    selected = prepare_root_universe(pd.concat([base, validation, holdout]), split="validation", config=R27Config())
    assert selected["root_event_id"].tolist() == ["v1"]
