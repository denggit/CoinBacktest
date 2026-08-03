from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_failure_overlay.config import FailureOverlayConfig
from src.ai_research.long_tail_failure_overlay.policy import (
    OverlayThresholds,
    enforce_non_overlap,
    score_tier,
    simulate_overlay_event,
    stable_candidates,
    structural_gate_count,
)


def _thresholds() -> OverlayThresholds:
    return OverlayThresholds(
        global_warning=0.70,
        global_confirm=0.80,
        ultra_confirm=0.95,
        tier_warning={"q70_to_q80": 0.65, "q80_to_q90": 0.75, "q90_plus": 0.85},
        tier_confirm={"q70_to_q80": 0.75, "q80_to_q90": 0.85, "q90_plus": 0.92},
    )


def _event_row(percentile: float = 0.75) -> pd.Series:
    values: dict[str, object] = {
        "event_id": "event",
        "fold_id": "WF_2024",
        "decision_time": pd.Timestamp("2024-01-01 00:00"),
        "event_score_percentile": percentile,
        "label_persistent_failure": 1,
        "mfe_360m": 0.003,
        "mae_360m": 0.025,
        "p_failure_60": 0.90,
        "p_failure_180": 0.95,
        "x60_current_below_entry": 1.0,
        "x180_current_below_entry": 1.0,
        "x180_last60_return": -0.01,
        "x180_broke_prior_low_60": 1.0,
        "x180_distance_to_prior_low_60": -0.005,
        "x180_bar15_lower_low_share": 0.8,
        "x180_recovery_from_trough": 0.001,
        "x180_underwater_fraction": 0.9,
        "score_upgrade_by_180": False,
        "score_upgrade_by_360": False,
    }
    for delay in (1, 3, 5):
        entry = pd.Timestamp("2024-01-01 00:00") + pd.Timedelta(minutes=delay)
        values[f"entry_time_delay_{delay}m"] = entry
        values[f"entry_price_delay_{delay}m"] = 100.0
        values[f"fixed_exit_time_delay_{delay}m"] = entry + pd.Timedelta(minutes=359)
        values[f"fixed_exit_price_delay_{delay}m"] = 96.0
        values[f"overlay_exit_time_delay_{delay}m"] = entry + pd.Timedelta(minutes=180)
        values[f"overlay_exit_price_delay_{delay}m"] = 98.0
        values[f"disaster_exit_time_delay_{delay}m"] = pd.NaT
        values[f"disaster_exit_price_delay_{delay}m"] = np.nan
    return pd.Series(values)


def test_score_tiers_are_preserved() -> None:
    assert score_tier(0.70) == "q70_to_q80"
    assert score_tier(0.85) == "q80_to_q90"
    assert score_tier(0.95) == "q90_plus"


def test_structural_gate_requires_multiple_failures() -> None:
    count, flags = structural_gate_count(_event_row(), FailureOverlayConfig())
    assert count >= 5
    assert flags["below_entry"]
    assert flags["prior_low_break"]


def test_t60_warning_and_t180_confirmation_are_both_required() -> None:
    row = _event_row()
    trade = simulate_overlay_event(
        row,
        policy="tiered_failure_overlay",
        delay_minutes=1,
        thresholds=_thresholds(),
        config=FailureOverlayConfig(),
    )
    assert trade is not None
    assert trade["exit_reason"] == "confirmed_persistent_failure_t180"
    no_warning = row.copy()
    no_warning["p_failure_60"] = 0.10
    trade2 = simulate_overlay_event(
        no_warning,
        policy="tiered_failure_overlay",
        delay_minutes=1,
        thresholds=_thresholds(),
        config=FailureOverlayConfig(),
    )
    assert trade2 is not None
    assert trade2["exit_reason"] == "fixed_6h_diagnostic"


def test_higher_score_tier_requires_stricter_failure_probability() -> None:
    row = _event_row(0.95)
    row["p_failure_60"] = 0.80
    row["p_failure_180"] = 0.90
    trade = simulate_overlay_event(
        row,
        policy="tiered_failure_overlay",
        delay_minutes=1,
        thresholds=_thresholds(),
        config=FailureOverlayConfig(),
    )
    assert trade is not None
    assert trade["exit_reason"] == "fixed_6h_diagnostic"


def test_disaster_stop_has_priority_and_uses_recorded_next_open() -> None:
    row = _event_row()
    row["disaster_exit_time_delay_1m"] = pd.Timestamp("2024-01-01 01:01")
    row["disaster_exit_price_delay_1m"] = 96.5
    trade = simulate_overlay_event(
        row,
        policy="tiered_failure_overlay",
        delay_minutes=1,
        thresholds=_thresholds(),
        config=FailureOverlayConfig(),
    )
    assert trade is not None
    assert trade["exit_reason"] == "disaster_stop_next_open"
    assert trade["exit_price"] == 96.5


def test_non_overlap_uses_actual_policy_exit() -> None:
    frame = pd.DataFrame(
        [
            {"entry_time": pd.Timestamp("2024-01-01"), "exit_time": pd.Timestamp("2024-01-01 03:00"), "decision_time": pd.Timestamp("2024-01-01"), "event_score_percentile": 0.8},
            {"entry_time": pd.Timestamp("2024-01-01 02:00"), "exit_time": pd.Timestamp("2024-01-01 05:00"), "decision_time": pd.Timestamp("2024-01-01 02:00"), "event_score_percentile": 0.9},
            {"entry_time": pd.Timestamp("2024-01-01 04:00"), "exit_time": pd.Timestamp("2024-01-01 10:00"), "decision_time": pd.Timestamp("2024-01-01 04:00"), "event_score_percentile": 0.7},
        ]
    )
    kept, skipped = enforce_non_overlap(frame)
    assert len(kept) == 2
    assert skipped == 1


def test_stable_overlay_requires_both_years_and_baseline_uplift() -> None:
    rows = []
    for fold, overlay_total, baseline_total in (("WF_2024", 0.50, 0.40), ("WF_2025", 0.80, 0.70)):
        for policy, total, overlay_share, uplift in (
            ("fixed_6h", baseline_total, 0.0, np.nan),
            ("fixed_6h_disaster_stop", baseline_total + 0.02, 0.0, np.nan),
            ("tiered_failure_overlay", overlay_total, 0.05, 0.002),
        ):
            rows.append(
                {
                    "fold_id": fold,
                    "policy": policy,
                    "delay_minutes": 1,
                    "cost_multiplier": 2.0,
                    "trades": 200,
                    "mean_net_return": 0.003,
                    "profit_factor": 1.6,
                    "total_compounded_return": total,
                    "max_drawdown": -0.12,
                    "top10_profit_share": 0.4,
                    "mean_net_without_top10": 0.001,
                    "overlay_exit_share": overlay_share,
                    "overlay_mean_uplift_gross": uplift,
                    "overlay_false_exit_share": 0.2,
                }
            )
    periods = pd.DataFrame(
        [
            {"policy": policy, "delay_minutes": 1, "cost_multiplier": 2.0, "quarter": f"{year}Q{quarter}", "mean_net_return": 0.001}
            for policy in ("fixed_6h", "fixed_6h_disaster_stop", "tiered_failure_overlay")
            for year in (2024, 2025)
            for quarter in range(1, 5)
        ]
    )
    stable = stable_candidates(pd.DataFrame(rows), periods, FailureOverlayConfig())
    row = stable.loc[stable["policy"] == "tiered_failure_overlay"].iloc[0]
    assert bool(row["stable_overlay_upgrade"])


def test_config_does_not_restore_abandoned_state_model() -> None:
    text = str(FailureOverlayConfig().to_dict()).lower()
    for token in ("strategic_state", "tactical_state", "entry_state", "activity_state"):
        assert token not in text


def test_earlier_overlay_beats_later_disaster_breach() -> None:
    row = _event_row()
    row["disaster_exit_time_delay_1m"] = pd.Timestamp("2024-01-01 04:01")
    row["disaster_exit_price_delay_1m"] = 95.0
    trade = simulate_overlay_event(
        row,
        policy="tiered_failure_overlay",
        delay_minutes=1,
        thresholds=_thresholds(),
        config=FailureOverlayConfig(),
    )
    assert trade is not None
    assert trade["exit_reason"] == "confirmed_persistent_failure_t180"
    assert trade["exit_price"] == 98.0
