from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_absorption_model.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_absorption_model.evaluation import (
    attach_trade_stress,
    calibration_thresholds,
    select_first_snapshot,
    trade_summary,
)
from src.ai_research.latent_liquidity_absorption_model.modeling import (
    fit_models,
    metric_table,
    predict,
    prepare_design,
)
from src.ai_research.latent_liquidity_absorption_model.replay import snapshot_rows_for_event
from src.ai_research.latent_liquidity_absorption_model.reports import causal_audit


def _path() -> tuple[pd.DataFrame, pd.Timestamp]:
    config = DEFAULT_CONFIG
    event_time = pd.Timestamp("2025-01-02 12:00:00")
    index = pd.date_range(
        event_time - pd.Timedelta(seconds=config.pre_replay_seconds),
        event_time + pd.Timedelta(seconds=config.post_replay_seconds),
        freq="1s",
    )
    n = len(index)
    close = np.full(n, 100.0)
    event_pos = config.pre_replay_seconds
    # Downward liquidity release followed by stabilization and reversal.
    close[event_pos + 1 : event_pos + 31] = np.linspace(100.0, 99.0, 30)
    close[event_pos + 31 : event_pos + 91] = np.linspace(99.0, 99.7, 60)
    close[event_pos + 91 :] = 99.7
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    notional = np.full(n, 1_000_000.0)
    notional[event_pos + 1 : event_pos + 31] = 4_000_000.0
    trades = np.full(n, 100.0)
    delta = np.zeros(n)
    delta[event_pos + 1 : event_pos + 31] = -2_000_000.0
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "notional": notional,
            "trades_count": trades,
            "delta_notional": delta,
            "unsafe_gap": False,
        },
        index=index,
    )
    return frame, event_time


def test_snapshot_rows_are_causal_and_multitask() -> None:
    path, event_time = _path()
    config = replace(DEFAULT_CONFIG, decision_offsets_seconds=(15, 60), model_n_estimators=10)
    event = SimpleNamespace(
        event_id="e1",
        release_episode_id="ep1",
        event_time=event_time,
        event_side="DOWN",
        period="VALIDATION_2025Q1_Q3",
        path_cluster=10,
        cluster_distance=1.2,
        event_reference_price=100.0,
    )
    rows = snapshot_rows_for_event(path, event, config)
    assert rows["decision_offset_seconds"].tolist() == [15, 60]
    assert ((pd.to_datetime(rows["entry_time"]) - pd.to_datetime(rows["decision_time"])) == pd.Timedelta(seconds=1)).all()
    assert (pd.to_datetime(rows["feature_available_time"]) <= pd.to_datetime(rows["decision_time"])).all()
    assert {"absorption_complete_target", "tradeable_before_stop_target", "future_additional_extension_bp", "future_favorable_mfe_bp"} <= set(rows)
    assert bool(rows.iloc[-1]["absorption_complete_target"])
    assert "tradeable_before_stop_d3_c2x" in rows


def _model_frame(rows_per_period: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frames = []
    for period in DEFAULT_CONFIG.periods:
        n = rows_per_period
        pressure = rng.normal(size=n)
        reclaim = rng.normal(8, 5, size=n)
        cluster = rng.choice([10, 4, 5, 8], size=n)
        side = rng.choice(["DOWN", "UP"], size=n)
        latent = 0.9 * pressure + 0.05 * reclaim + (cluster == 5) * 0.5 - (cluster == 8) * 0.5
        probability = 1.0 / (1.0 + np.exp(-latent))
        target = rng.random(n) < probability
        absorb = rng.random(n) < (1.0 / (1.0 + np.exp(-(pressure + 0.2))))
        decision_offset = rng.choice(DEFAULT_CONFIG.decision_offsets_seconds, size=n)
        frame = pd.DataFrame(
            {
                "event_id": [f"{period}-{i // 3}" for i in range(n)],
                "release_episode_id": [f"ep-{period}-{i // 3}" for i in range(n)],
                "event_time": pd.date_range("2023-01-01", periods=n, freq="min"),
                "decision_time": pd.date_range("2023-01-01", periods=n, freq="min") + pd.to_timedelta(decision_offset, unit="s"),
                "entry_time": pd.date_range("2023-01-01", periods=n, freq="min") + pd.to_timedelta(decision_offset + 1, unit="s"),
                "feature_available_time": pd.date_range("2023-01-01", periods=n, freq="min") + pd.to_timedelta(decision_offset, unit="s"),
                "event_side": side,
                "period": period,
                "path_cluster": cluster,
                "cluster_distance": rng.uniform(0, 3, size=n),
                "decision_offset_seconds": decision_offset,
                "extension_from_reference_bp": rng.uniform(0, 50, size=n),
                "reclaim_from_known_extreme_bp": reclaim,
                "seconds_since_known_extreme": rng.integers(0, 60, size=n),
                "pressure_no_progress_15s": pressure,
                "price_efficiency_15s": rng.uniform(0, 1, size=n),
                "notional_intensity_15s": rng.uniform(0.5, 5, size=n),
                "tradeable_before_stop_target": target,
                "absorption_complete_target": absorb,
                "future_additional_extension_bp": np.maximum(0, 12 - 4 * pressure + rng.normal(0, 2, size=n)),
                "future_favorable_mfe_bp": np.maximum(0, 20 + 6 * pressure + rng.normal(0, 3, size=n)),
                "future_adverse_mae_bp": rng.uniform(1, 20, size=n),
                "future_terminal_net_bp_1x": rng.normal(size=n),
                "known_extreme_price": 100.0,
                "current_close": 100.0,
                "event_reference_price": 100.0,
                "entry_price": 100.0,
                "barrier_result": np.where(target, "TARGET", "STOP"),
            }
        )
        for delay in DEFAULT_CONFIG.entry_delay_seconds:
            frame[f"structural_stop_distance_bp_d{delay}"] = 10.0
            frame[f"future_favorable_mfe_bp_d{delay}"] = frame["future_favorable_mfe_bp"]
            frame[f"future_adverse_mae_bp_d{delay}"] = frame["future_adverse_mae_bp"]
            for multiple in DEFAULT_CONFIG.cost_multipliers:
                suffix = f"d{delay}_c{int(multiple)}x"
                frame[f"barrier_result_{suffix}"] = np.where(target, "TARGET", "STOP")
                frame[f"tradeable_before_stop_{suffix}"] = target
                frame[f"future_terminal_net_bp_{suffix}"] = np.where(target, 15.0, -10.0 - 11.0 * multiple)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_model_fit_prediction_and_metrics() -> None:
    frame = _model_frame()
    config = replace(DEFAULT_CONFIG, model_n_estimators=30, minimum_train_rows=500, minimum_class_rows=50)
    models = fit_models(frame, config)
    predicted = predict(frame, models)
    assert predicted["trade_score"].between(0, 1).all()
    assert predicted["pred_additional_extension_bp"].ge(0).all()
    metrics = metric_table(predicted, config)
    assert set(metrics["task"]) >= {"TRADEABLE_FULL", "TRADEABLE_BASELINE", "ABSORPTION_COMPLETE"}
    assert metrics.loc[metrics["task"].eq("TRADEABLE_FULL"), "roc_auc"].notna().all()


def test_threshold_selects_first_snapshot_per_episode() -> None:
    frame = _model_frame(400)
    frame["trade_score"] = np.linspace(0, 1, len(frame))
    frame["p_tradeable_baseline"] = frame["trade_score"]
    thresholds = calibration_thresholds(frame, replace(DEFAULT_CONFIG, selection_quantile=0.8), "trade_score")
    selected = select_first_snapshot(frame, thresholds, score_column="trade_score", model_name="FULL")
    assert not selected["event_id"].duplicated().any()
    assert selected["selection_model"].eq("FULL").all()


def test_trade_stress_uses_target_stop_and_terminal_conservatively() -> None:
    selected = _model_frame(20).head(3).copy()
    selected["selection_model"] = "FULL"
    selected["selection_score"] = 0.9
    selected["threshold"] = 0.8
    selected["p_tradeable"] = 0.9
    selected["p_absorption_complete"] = 0.9
    selected["pred_additional_extension_bp"] = 1.0
    selected["pred_remaining_mfe_bp"] = 30.0
    selected["predicted_net_room_bp"] = 29.0
    trades = attach_trade_stress(selected, DEFAULT_CONFIG)
    assert len(trades) == 3 * len(DEFAULT_CONFIG.entry_delay_seconds) * len(DEFAULT_CONFIG.cost_multipliers)
    summary = trade_summary(trades)
    assert not summary.empty
    assert {"profit_factor", "top10_removed_mean_net_bp"} <= set(summary)


def test_feature_schema_excludes_future_and_raw_prices() -> None:
    frame = _model_frame(50)
    _, columns = prepare_design(frame)
    assert not any(name.startswith(("future_", "tradeable_", "barrier_result_")) for name in columns)
    assert not any("swing" in name.lower() for name in columns)
    assert "known_extreme_price" not in columns


def test_causal_audit_passes_clean_snapshot() -> None:
    frame = _model_frame(50)
    _, columns = prepare_design(frame)
    source_gate = pd.DataFrame([{"check": "source", "status": "PASS"}])
    quality = pd.DataFrame([{"requested_events": 100, "complete_events": 100}])
    threshold = pd.DataFrame([{"holdout_used_for_threshold": False}])
    audit = causal_audit(frame, columns, source_gate, quality, threshold, DEFAULT_CONFIG)
    assert not audit["status"].eq("FAIL").any()


def test_final_gate_promotes_only_when_prediction_execution_months_and_stress_pass() -> None:
    from src.ai_research.latent_liquidity_absorption_model.reports import decide

    metrics = pd.DataFrame(
        [
            {"period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "task": "TRADEABLE_FULL", "roc_auc": 0.64},
            {"period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "task": "TRADEABLE_BASELINE", "roc_auc": 0.58},
        ]
    )
    trades = pd.DataFrame(
        [
            {"selection_model": "FULL", "period": DEFAULT_CONFIG.calibration_period, "event_side": "DOWN", "entry_delay_seconds": 1, "cost_multiple": 1.0, "trades": 150, "mean_net_bp": 4.0, "profit_factor": 1.3, "top10_removed_mean_net_bp": 2.0},
            {"selection_model": "FULL", "period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "entry_delay_seconds": 1, "cost_multiple": 1.0, "trades": 140, "mean_net_bp": 3.5, "profit_factor": 1.25, "top10_removed_mean_net_bp": 1.5},
            {"selection_model": "FULL", "period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "entry_delay_seconds": 1, "cost_multiple": 2.0, "trades": 140, "mean_net_bp": 0.5, "profit_factor": 1.02, "top10_removed_mean_net_bp": 0.1},
        ]
    )
    monthly = pd.DataFrame(
        [
            {"selection_model": "FULL", "period": period, "event_side": "DOWN", "month": f"m{i}", "sum_net_bp": 1.0 if i < 4 else -1.0}
            for period in (DEFAULT_CONFIG.calibration_period, DEFAULT_CONFIG.holdout_period)
            for i in range(5)
        ]
    )
    causal = pd.DataFrame([{"check": "all", "status": "PASS"}])
    decision, _ = decide(metrics, trades, monthly, causal, DEFAULT_CONFIG)
    assert decision == "PROMOTE_DOWN_TO_R02_FORMAL_STRATEGY_BACKTEST"


def test_final_gate_stops_after_declared_last_chance_failure() -> None:
    from src.ai_research.latent_liquidity_absorption_model.reports import decide

    metrics = pd.DataFrame(
        [
            {"period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "task": "TRADEABLE_FULL", "roc_auc": 0.56},
            {"period": DEFAULT_CONFIG.holdout_period, "event_side": "DOWN", "task": "TRADEABLE_BASELINE", "roc_auc": 0.55},
        ]
    )
    decision, reasons = decide(metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"status": "PASS"}]), DEFAULT_CONFIG)
    assert decision == "STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE"
    assert any("final commercial gate" in reason for reason in reasons)
