from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_incremental_hold.config import IncrementalHoldConfig
from src.ai_research.long_tail_incremental_hold.features import (
    _incremental_target,
    assert_no_future_features,
    eligible_endpoints,
    score_tier,
)
from src.ai_research.long_tail_incremental_hold.modeling import (
    causal_oof,
    choose_feature_set,
    stable_candidates,
)
from src.ai_research.long_tail_multistage_decision.features import FeatureSet


def _points(rows: int = 7200) -> pd.DataFrame:
    close = np.zeros(rows, dtype=float)
    high = np.zeros(rows, dtype=float)
    low = np.zeros(rows, dtype=float)
    close[:360] = np.linspace(0.0, 0.01, 360)
    close[360:720] = np.linspace(0.01, 0.02, 360)
    close[720:] = 0.02
    high[:] = close + 0.001
    low[:] = close - 0.002
    return pd.DataFrame({"close_return": close, "high_return": high, "low_return": low})


def test_score_tiers_are_preserved() -> None:
    assert score_tier(0.70) == "q70_to_q80"
    assert score_tier(0.85) == "q80_to_q90"
    assert score_tier(0.95) == "q90_plus"


def test_checkpoints_are_decisions_not_exit_contracts() -> None:
    config = IncrementalHoldConfig()
    assert eligible_endpoints(360, config) == (720, 1440, 2880, 7200)
    text = str(config.to_dict()).lower()
    assert "fixed_6h" not in text
    for token in ("strategic_state", "tactical_state", "entry_state", "activity_state"):
        assert token not in text


def test_incremental_utility_uses_only_window_after_checkpoint() -> None:
    points = _points()
    result = _incremental_target(points, checkpoint_minutes=360, endpoint_minutes=720, risk_penalty=1.25)
    assert result["incremental_close_return"] > 0.009
    assert result["additional_drawdown"] >= 0.0
    expected = result["incremental_close_return"] - 1.25 * result["additional_drawdown"]
    assert np.isclose(result["incremental_utility"], expected)


def test_future_after_endpoint_cannot_change_shorter_target() -> None:
    points = _points()
    before = _incremental_target(points, checkpoint_minutes=360, endpoint_minutes=720, risk_penalty=1.25)
    changed = points.copy()
    changed.loc[1000:, ["close_return", "high_return", "low_return"]] = -0.50
    after = _incremental_target(changed, checkpoint_minutes=360, endpoint_minutes=720, risk_penalty=1.25)
    assert before == after


def test_future_columns_are_rejected_as_features() -> None:
    assert_no_future_features(("x_path__current_return", "x_score__entry_percentile"))
    try:
        assert_no_future_features(("x_path__current_return", "label_best_incremental_utility"))
    except ValueError as exc:
        assert "future columns" in str(exc)
    else:
        raise AssertionError("future label leakage was not rejected")


def test_causal_oof_ridge_uses_multiple_time_folds() -> None:
    rows = 420
    frame = pd.DataFrame(
        {
            "checkpoint_time": pd.date_range("2023-01-01", periods=rows, freq="12h"),
            "x_path__current_return": np.linspace(-0.03, 0.03, rows),
            "x_path__current_mae": np.linspace(0.03, 0.0, rows),
        }
    )
    target = 0.7 * frame["x_path__current_return"].to_numpy() - 0.2 * frame["x_path__current_mae"].to_numpy()
    config = IncrementalHoldConfig(
        holding_oof_splits=4,
        holding_oof_embargo_hours=120,
        minimum_train_rows=60,
        minimum_test_rows=20,
        minimum_oof_folds=2,
    )
    feature_set = FeatureSet("mechanical_ridge", ("x_path__current_return", "x_path__current_mae"))
    result = causal_oof(frame, target, feature_set=feature_set, config=config)
    assert result.folds_used >= 2
    assert np.isfinite(result.predictions).sum() > 40


def test_choose_feature_set_prefers_stable_simpler_candidate_when_scores_match() -> None:
    rows = 100
    target = np.linspace(-1.0, 1.0, rows)
    from src.ai_research.long_tail_incremental_hold.modeling import OOFRegressionResult

    simple = FeatureSet("mechanical_ridge", ("x",))
    complex_set = FeatureSet("path_plus_score_lightgbm", ("x", "z"))
    pred = target.copy()
    result = OOFRegressionResult(predictions=pred, folds_used=3, fold_audit=pd.DataFrame())
    selected, _, audit = choose_feature_set([(complex_set, result), (simple, result)], target)
    assert selected.name == "mechanical_ridge"
    assert bool(audit.loc[audit["feature_set"] == "mechanical_ridge", "selected"].iloc[0])


def test_stable_signal_requires_both_years() -> None:
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        rows.append(
            {
                "fold_id": fold,
                "checkpoint_minutes": 360,
                "target": "next_incremental_utility",
                "feature_set": "path_structure_lightgbm",
                "scope": "broad_q70",
                "rows": 200,
                "rank_ic": 0.20,
                "mae_skill": 0.05,
                "sign_auc": 0.65,
                "sign_accuracy": 0.60,
                "top_quintile_actual_mean": 0.010,
                "bottom_quintile_actual_mean": -0.005,
                "top_bottom_spread": 0.015,
                "decile_monotonicity": 0.70,
            }
        )
    stable = stable_candidates(pd.DataFrame(rows), IncrementalHoldConfig())
    assert bool(stable.iloc[0]["passes_cross_year"])
    one_year = stable_candidates(pd.DataFrame(rows[:1]), IncrementalHoldConfig())
    assert not bool(one_year.iloc[0]["passes_cross_year"])
