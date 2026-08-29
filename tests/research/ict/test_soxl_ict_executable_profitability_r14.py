import pandas as pd

from src.research_common.ict.executable_profitability import (
    ExecutableLeg,
    ExecutablePolicy,
    select_one_setup_per_sweep,
    select_policy_leg_entries,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ


def _row(**kw):
    base = {
        "ny_date": "2026-08-05", "event_id": "e1", "execution_tf": "1m",
        "structure_visibility_tier_r13": "visible_p50_p80",
        "entry_available_time": pd.Timestamp("2026-08-05 10:00", tz=NY_TZ),
        "break_available_time": pd.Timestamp("2026-08-05 10:00", tz=NY_TZ),
        "mss_reference_time": pd.Timestamp("2026-08-05 09:55", tz=NY_TZ),
        "mss_reference_price": 100.0, "trade_side": "LONG",
        "entry_price": 101.0, "stop_price": 99.0,
        "target_price": 110.0, "nearest_internal_target_price": 104.0,
        "target_liquidity_state": "shallow_probe_equal_like",
        "entry_model_r13": "break_middle_near",
    }
    base.update(kw)
    return base


def test_profit_core_does_not_require_equal_like_target():
    catalog = pd.DataFrame([
        _row(event_id="shallow", execution_tf="1m", target_liquidity_state="shallow_probe_equal_like", entry_model_r13="break_middle_near"),
        _row(event_id="partial", execution_tf="2m", target_liquidity_state="partial_consumed", entry_model_r13="break_middle_ce"),
    ])
    policy = ExecutablePolicy(
        "core",
        (
            ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),
            ExecutableLeg("2m", ("partial_consumed",), ("break_middle_ce",)),
        ),
    )
    out = select_policy_leg_entries(catalog, policy)
    assert set(out.event_id) == {"shallow", "partial"}


def test_fresh_target_is_deferred_from_narrow_r14_profit_core():
    catalog = pd.DataFrame([_row(target_liquidity_state="fresh")])
    policy = ExecutablePolicy(
        "core",
        (ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),),
    )
    assert select_policy_leg_entries(catalog, policy).empty


def test_one_setup_per_sweep_uses_earliest_signal_not_future_tf_preference():
    rows = pd.DataFrame([
        _row(execution_tf="2m", target_liquidity_state="partial_consumed", entry_model_r13="break_middle_ce", entry_available_time=pd.Timestamp("2026-08-05 10:00", tz=NY_TZ), _leg_rank_r14=1),
        _row(execution_tf="1m", entry_available_time=pd.Timestamp("2026-08-05 10:03", tz=NY_TZ), _leg_rank_r14=0),
    ])
    policy = ExecutablePolicy(
        "core",
        (
            ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),
            ExecutableLeg("2m", ("partial_consumed",), ("break_middle_ce",)),
        ),
    )
    chosen, rejected = select_one_setup_per_sweep(rows, policy=policy)
    assert len(chosen) == 1
    assert chosen.iloc[0].execution_tf == "2m"
    assert len(rejected) == 1
    assert rejected.iloc[0].r14_rejection_reason == "later_or_duplicate_setup_same_physical_sweep"


def test_micro_structure_is_not_selected_by_policy_leg_catalog():
    catalog = pd.DataFrame([_row(structure_visibility_tier_r13="micro_lt_p50")])
    policy = ExecutablePolicy(
        "core",
        (ExecutableLeg("1m", ("shallow_probe_equal_like",), ("break_middle_near",)),),
    )
    assert select_policy_leg_entries(catalog, policy).empty
