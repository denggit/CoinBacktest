from __future__ import annotations

import numpy as np
import pandas as pd

import src.research_common.ict_mss2.r03 as r03mod
from src.research_common.ict_mss2 import (
    R03Config,
    attach_causal_ict_swing_hierarchy,
    attach_causal_pool_hierarchy_to_episode_stages,
    build_core_reclaim_execution_overlays,
    build_hierarchy_stage_cohorts,
    build_tradebar_microstructure_features,
    hierarchy_causal_audit,
)


def _lifecycle_side(side: str, prices: list[float], tf: str = "15m", tf_min: int = 15) -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01 00:00")
    rows = []
    for i, price in enumerate(prices):
        pivot = base + pd.Timedelta(minutes=15 * i)
        avail = pivot + pd.Timedelta(minutes=30)
        rows.append({
            "level_id": i + 1,
            "pivot_side": side,
            "source_timeframe": tf,
            "source_timeframe_min": tf_min,
            "pivot_time": pivot,
            "level_price": price,
            "initial_available_time": avail,
            "sweep_bar_time_1m": base + pd.Timedelta(hours=8),
            "sweep_pos_1m": 500 + i,
            "trade_direction": -1 if side == "high" else 1,
            "external_50_flag": 0,
            "clean_sweep_no_prior_touch_flag": 1,
        })
    return pd.DataFrame(rows)


def test_ict_swing_hierarchy_is_recursive_and_causal() -> None:
    # ST highs: 100, 110, 102 -> 110 is IT.  Add another IT structure so the
    # center IT can itself become LT.
    highs = _lifecycle_side("high", [90, 100, 95, 120, 96, 105, 94])
    out = attach_causal_ict_swing_hierarchy(highs)
    # IT highs at 100, 120, 105; therefore 120 is LT once right IT=105 exists.
    center = out.loc[out["level_price"].eq(120)].iloc[0]
    assert pd.notna(center["ict_it_available_time"])
    assert pd.notna(center["ict_lt_available_time"])
    assert int(center["ict_swing_rank_at_sweep"]) == 3
    audit = hierarchy_causal_audit(out)
    assert int(audit["violations"].sum()) == 0


def test_ict_it_not_backfilled_before_right_st_confirmation() -> None:
    base = pd.Timestamp("2025-01-01 00:00")
    frame = _lifecycle_side("low", [110, 100, 108])
    # The center would eventually be IT, but sweep it before the right ST is
    # confirmed.  It must still be only ST at that sweep.
    frame.loc[1, "sweep_bar_time_1m"] = base + pd.Timedelta(minutes=50)
    out = attach_causal_ict_swing_hierarchy(frame)
    center = out.loc[out["level_price"].eq(100)].iloc[0]
    assert pd.notna(center["ict_it_available_time"])
    assert pd.Timestamp(center["ict_it_available_time"]) > pd.Timestamp(center["sweep_bar_time_1m"])
    assert int(center["ict_it_known_at_sweep_flag"]) == 0
    assert str(center["ict_swing_class_at_sweep"]) == "ST"


def test_pool_hierarchy_distinguishes_single_key_pool_from_st_only_count() -> None:
    lifecycle = pd.DataFrame([
        {"level_id": 1, "sweep_pos_1m": 10, "trade_direction": 1, "level_price": 100.0, "ict_swing_rank_at_sweep": 1, "source_timeframe_min": 15, "external_50_flag": 0, "clean_sweep_no_prior_touch_flag": 1},
        {"level_id": 2, "sweep_pos_1m": 10, "trade_direction": 1, "level_price": 100.05, "ict_swing_rank_at_sweep": 1, "source_timeframe_min": 30, "external_50_flag": 0, "clean_sweep_no_prior_touch_flag": 1},
        {"level_id": 3, "sweep_pos_1m": 20, "trade_direction": 1, "level_price": 98.0, "ict_swing_rank_at_sweep": 2, "source_timeframe_min": 60, "external_50_flag": 1, "clean_sweep_no_prior_touch_flag": 1},
    ])
    stages = pd.DataFrame([
        {"stage_id": "S1", "episode_id": "E1", "episode_stage_no": 1, "sweep_pos_1m": 10, "trade_direction": 1, "episode_elapsed_minutes": 0},
        {"stage_id": "S2", "episode_id": "E1", "episode_stage_no": 2, "sweep_pos_1m": 20, "trade_direction": 1, "episode_elapsed_minutes": 10},
    ])
    enriched, pools = attach_causal_pool_hierarchy_to_episode_stages(lifecycle, stages, tolerance_bps=10.0)
    s1 = enriched.loc[enriched["stage_id"].eq("S1")].iloc[0]
    s2 = enriched.loc[enriched["stage_id"].eq("S2")].iloc[0]
    assert int(s1["ict_price_pools_cum"]) == 1
    assert int(s1["ict_structural_key_pools_cum"]) == 1  # multi-TF pool despite ST-only swings
    assert int(s2["ict_it_plus_pools_cum"]) == 1
    cohorts = build_hierarchy_stage_cohorts(enriched)
    assert "first_it_plus_pool" in set(cohorts["hierarchy_cohort"])
    assert not pools.empty


def test_core_execution_overlay_preserves_missing_target_opportunity() -> None:
    idx = pd.date_range("2025-01-01", periods=20, freq="1min")
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx)
    core = pd.DataFrame([{
        "trade_event_id": "T1", "episode_id": "E1", "stage_id": "S1", "trade_direction": 1,
        "signal_available_time": idx[5], "entry_pos_1m": 5, "entry_time": idx[5], "entry_price": 100.0,
        "stop_price": 95.0, "target_htf240_price": np.nan, "target_htf240_outcome": "no_target",
        "target_htf240_gross_return": np.nan, "year": 2025, "quarter": "2025Q1",
    }])
    overlay, tie = build_core_reclaim_execution_overlays(
        bars, core, fvg_minutes=1, config=R03Config(fvg_signal_wait_minutes=5, execution_censor_minutes=10), show_progress=False
    )
    assert len(tie) == 1 and int(tie.iloc[0]["outcome_match"]) == 1
    assert len(overlay) == 4
    assert set(overlay["execution_variant"]) == {
        "reclaim_market", "post_reclaim_fvg_market", "post_reclaim_fvg_limit", "hybrid_reclaim_market_fvg_limit"
    }


def test_tradebar_cvd_divergence_uses_only_completed_bars(monkeypatch) -> None:
    idx = pd.date_range("2025-01-01 00:00", periods=12, freq="1min")
    lows = [100, 99, 98, 97, 96, 95, 97, 96, 94, 93, 92, 80]
    delta = [-100, -100, -100, -100, -100, -100, 300, 200, 100, 100, 100, -999999]
    fake = pd.DataFrame({
        "open": [101.0]*12, "high": [102.0]*12, "low": lows, "close": [100.0]*12,
        "notional": [1000.0]*12, "buy_notional": [500.0]*12, "sell_notional": [500.0]*12,
        "delta_notional": delta, "trades_count": [10.0]*12,
        "large_buy_notional": [0.0]*12, "large_sell_notional": [0.0]*12, "max_trade_notional": [100.0]*12,
    }, index=idx)

    class FakeLoader:
        def __init__(self, *a, **k):
            pass
        def load_local_data(self, *a, **k):
            return fake.copy()

    monkeypatch.setattr(r03mod, "OKXTradeBarLoader", FakeLoader)
    checkpoints = pd.DataFrame({
        "checkpoint_id": ["T1"], "decision_time": [idx[11]], "episode_start_time": [idx[0]],
    })
    feat, audit = build_tradebar_microstructure_features(checkpoints, config=R03Config(), show_progress=False)
    row = feat.iloc[0]
    # Bar 11 starts at the decision and must not be used; the giant negative delta
    # therefore cannot contaminate episode CVD.
    assert row["tb_last_source_time"] == idx[10]
    assert float(row["tb_episode_cvd_end"]) > -999999
    assert int(audit["causal_bad"].sum()) == 0


def _post_sweep_mss_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2025-01-01 00:00", periods=30, freq="1min")
    open_ = np.full(30, 100.0)
    close = np.full(30, 100.0)
    high = np.full(30, 100.8)
    low = np.full(30, 99.2)
    # Pre-sweep high is deliberately too high to be broken by the later rebound.
    high[1:5] = [101.0, 105.0, 102.0, 101.0]
    close[2] = 104.0
    # Liquidity sweep / extreme.
    low[5] = 95.0
    close[5] = 96.0
    # A NEW small STH forms only after the sweep: pos7 high=102, confirmed when
    # right bar pos8 closes, hence usable at pos9 start.  It is broken at pos10.
    high[6:11] = [100.0, 102.0, 101.0, 101.2, 103.5]
    low[6:11] = [96.0, 97.0, 96.8, 97.2, 99.0]
    close[6:11] = [98.0, 100.5, 99.0, 100.0, 103.0]
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    stages = pd.DataFrame([{
        "stage_id": "S1", "episode_id": "E1", "episode_stage_no": 1,
        "sweep_pos_1m": 5, "sweep_bar_time_1m": idx[5], "trade_direction": 1,
        "episode_start_pos_1m": 5, "episode_start_time_1m": idx[5],
        "max_consumed_level_price_stage": 96.0, "min_consumed_level_price_stage": 96.0,
        "max_consumed_level_price_cum": 96.0, "min_consumed_level_price_cum": 96.0,
    }])
    return bars, stages


def test_post_sweep_new_sth_can_be_valid_bullish_mss() -> None:
    from src.research_common.ict_mss2 import build_stack_execution_triggers

    bars, stages = _post_sweep_mss_bars()
    out = build_stack_execution_triggers(
        bars, stages, execution_minutes=1, reference_modes=("post_sweep_st",),
        include_reclaims=False, include_mss_market=True, include_mss_fvg=False,
        show_progress=False,
    )
    assert not out.empty
    row = out.loc[out["trigger_type"].eq("mss_post_sweep_st_market")].iloc[0]
    assert int(row["mss_reference_pivot_pos"]) > int(row["sweep_exec_pos"])
    assert pd.Timestamp(row["mss_reference_available_time"]) <= pd.Timestamp(row["signal_bar_time"])
    assert int(row["signal_exec_pos"]) > int(row["mss_reference_pivot_pos"])
    assert float(row["mss_reference_price"]) == 102.0


def test_pre_sweep_reference_modes_do_not_steal_post_sweep_sth() -> None:
    from src.research_common.ict_mss2 import build_stack_execution_triggers

    bars, stages = _post_sweep_mss_bars()
    pre = build_stack_execution_triggers(
        bars, stages, execution_minutes=1, reference_modes=("recent", "structural"),
        include_reclaims=False, include_mss_market=True, include_mss_fvg=False,
        show_progress=False,
    )
    # The rebound only reaches 103, while the known pre-sweep swing is ~105.
    assert pre.empty or not pre["trigger_type"].astype(str).str.contains("post_sweep_st").any()


def test_post_sweep_mss_audit_and_displacement_features_are_nonvacuous() -> None:
    from src.research_common.ict_mss2 import build_stack_execution_triggers, mss_reference_causal_audit

    bars, stages = _post_sweep_mss_bars()
    out = build_stack_execution_triggers(
        bars, stages, execution_minutes=1, reference_modes=("post_sweep_st",),
        include_reclaims=False, include_mss_market=True, include_mss_fvg=False,
        show_progress=False,
    )
    audit = mss_reference_causal_audit(out)
    assert int(audit["violations"].sum()) == 0
    row = out.iloc[0]
    for col in (
        "displacement_atr", "displacement_speed_atr_per_min", "max_directional_body_atr",
        "directional_body_share", "attack_displacement_atr", "reversal_attack_distance_ratio",
    ):
        assert col in out.columns
    # The ratio is a feature only; a weaker-than-attack reversal is not rejected.
    assert "reversal_weaker_than_attack_flag" in out.columns


def test_displacement_atlas_keeps_non_monotonic_quartiles_and_weaker_reversals() -> None:
    from src.research_common.ict_mss2 import build_displacement_payoff_atlas

    n = 160
    t = pd.date_range("2023-01-01", periods=n, freq="10D")
    x = np.linspace(0.1, 2.0, n)
    # Deliberately make the middle range better than the strongest tail.
    ret = np.where((x >= 0.6) & (x <= 1.2), 0.01, -0.003)
    frame = pd.DataFrame({
        "entry_time": t,
        "trade_direction": 1,
        "execution_minutes": 2,
        "reference_mode": "post_sweep_st",
        "trigger_type": "mss_post_sweep_st_market",
        "displacement_atr": x,
        "displacement_speed_atr_per_min": x / 10.0,
        "displacement_leg_range_atr": x + 0.2,
        "max_directional_body_atr": x / 2.0,
        "directional_body_share": np.clip(x / 2.0, 0, 1),
        "path_efficiency": np.clip(x / 2.0, 0, 1),
        "break_distance_atr": x / 4.0,
        "mss_body_atr": x / 3.0,
        "mss_body_ratio": np.clip(x / 2.0, 0, 1),
        "fvg_count_in_leg": np.floor(x * 2),
        "fvg_density_per_bar": x / 10.0,
        "largest_fvg_width_atr": x / 5.0,
        "attack_displacement_atr": np.ones(n),
        "attack_path_efficiency": np.full(n, 0.7),
        "attack_speed_atr_per_min": np.full(n, 0.1),
        "reversal_attack_distance_ratio": x,
        "reversal_attack_speed_ratio": x,
        "target_htf240_net_return_cost2x": ret,
    })
    summary, thresholds, relative = build_displacement_payoff_atlas(frame, min_train_rows=20)
    assert not summary.empty and not thresholds.empty and not relative.empty
    disp = summary.loc[summary["displacement_feature"].eq("displacement_atr") & summary["split"].eq("all")]
    assert {"Q1", "Q2", "Q3", "Q4"}.issubset(set(disp["displacement_bucket"].astype(str)))
    assert "<0.5" in set(relative["relative_strength_bucket"].astype(str))
