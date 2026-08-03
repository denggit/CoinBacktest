#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict chronological supervised modeling and financial evaluation for R13."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import gc

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .config import PostSweepSupervisedConfig
from .features import FeatureModuleResult

EPS = 1e-12

META_AND_LABEL_PREFIXES: tuple[str, ...] = (
    "checkpoint_id", "zone_event_id", "decision_time", "entry_time", "event_available_time",
    "event_bar_time", "period", "split", "checkpoint_minutes", "long_", "short_",
)
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "future_", "outcome", "target_hit", "stop_hit", "target_before_stop", "gross_r",
    "net_1x", "net_2x", "exit_", "stopped", "horizon_end", "profitable_label",
)

ABLATION_STEPS: tuple[tuple[str, str | None], ...] = (
    ("A_R09_BASE", None),
    ("B_R09_R12", "dynamic"),
    ("C_PLUS_TRADE_1S", "trade_1s"),
    ("D_PLUS_RANGE_R0020", "range_r0020"),
    ("E_PLUS_FOOTPRINT", "footprint"),
    ("F_PLUS_OI", "oi"),
)
MODULE_PRESENT_COLUMNS: dict[str, str] = {
    "trade_1s": "trade1s_causal_valid",
    "range_r0020": "range_causal_valid",
    "footprint": "fp_causal_valid",
    "oi": "oi_context_present",
}


@dataclass(frozen=True)
class ModelingResult:
    model_summary: pd.DataFrame
    selection_summary: pd.DataFrame
    prediction_sample: pd.DataFrame
    score_deciles: pd.DataFrame
    feature_contract: pd.DataFrame
    decision_summary: pd.DataFrame


def _signed_log1p(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.sign(values) * np.log1p(np.abs(values))


def _prepare_feature_frame(frame: pd.DataFrame, columns: Iterable[str], train_mask: pd.Series) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    selected = []
    contract_rows: list[dict[str, object]] = []
    for name in dict.fromkeys(columns):
        if name not in frame.columns:
            continue
        low = name.lower()
        if any(token in low for token in FORBIDDEN_FEATURE_TOKENS):
            raise RuntimeError(f"forbidden label/outcome feature reached modeling: {name}")
        if pd.api.types.is_datetime64_any_dtype(frame[name]):
            continue
        train = frame.loc[train_mask, name]
        missing_share = float(train.isna().mean()) if len(train) else 1.0
        unique = int(train.nunique(dropna=True))
        status = "kept"
        if missing_share > 0.95:
            status = "drop_missing_gt95pct"
        elif unique <= 1:
            status = "drop_constant"
        contract_rows.append({
            "feature": name,
            "dtype": str(frame[name].dtype),
            "train_missing_share": missing_share,
            "train_unique": unique,
            "status": status,
        })
        if status == "kept":
            selected.append(name)
    out = frame.loc[:, selected].copy()
    numeric: list[str] = []
    categorical: list[str] = []
    for name in selected:
        if pd.api.types.is_bool_dtype(out[name]):
            out[name] = out[name].astype(float)
            numeric.append(name)
        elif pd.api.types.is_numeric_dtype(out[name]):
            out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            low = name.lower()
            if any(token in low for token in ("notional", "trades_count", "duration", "age_", "_count", "minutes")):
                out[name] = _signed_log1p(out[name])
            numeric.append(name)
        else:
            # High-cardinality identifiers should already be absent.  Treat the
            # remaining structural states/timeframes as compact categoricals.
            out[name] = out[name].astype("string").fillna("MISSING")
            if out.loc[train_mask, name].nunique(dropna=True) > 50:
                contract_rows[-1]["status"] = "drop_high_cardinality"
                out = out.drop(columns=name)
            else:
                categorical.append(name)
    numeric = [name for name in numeric if name in out.columns]
    categorical = [name for name in categorical if name in out.columns]
    return out, numeric, categorical, pd.DataFrame(contract_rows)


def _preprocessor(kind: str, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    if not numeric and not categorical:
        raise RuntimeError("no usable features")
    if kind == "LOGISTIC":
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
    else:
        numeric_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _estimator(kind: str, cfg: PostSweepSupervisedConfig):
    if kind == "LOGISTIC":
        return LogisticRegression(
            C=cfg.logistic_c, max_iter=1_000, class_weight="balanced",
            solver="lbfgs", random_state=cfg.random_state,
        )
    if kind == "HGB":
        return HistGradientBoostingClassifier(
            learning_rate=cfg.hgb_learning_rate, max_iter=cfg.hgb_max_iter,
            max_leaf_nodes=cfg.hgb_max_leaf_nodes, max_depth=cfg.hgb_max_depth,
            min_samples_leaf=cfg.hgb_min_samples_leaf, l2_regularization=cfg.hgb_l2_regularization,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
            class_weight="balanced", random_state=cfg.random_state,
        )
    raise ValueError(kind)


def _pipeline(kind: str, numeric: list[str], categorical: list[str], cfg: PostSweepSupervisedConfig) -> Pipeline:
    """Compatibility helper used by unit tests; the full run transforms once per model kind."""
    return Pipeline([("preprocess", _preprocessor(kind, numeric, categorical)), ("model", _estimator(kind, cfg))])


def _classification_metrics(y: pd.Series, score: np.ndarray) -> dict[str, float]:
    valid = y.notna() & np.isfinite(score)
    yy = y.loc[valid].astype(int).to_numpy()
    ss = np.asarray(score)[valid.to_numpy()]
    if len(yy) == 0:
        return {"auc": np.nan, "average_precision": np.nan, "brier": np.nan, "positive_rate": np.nan}
    return {
        "auc": float(roc_auc_score(yy, ss)) if len(np.unique(yy)) > 1 else np.nan,
        "average_precision": float(average_precision_score(yy, ss)) if len(np.unique(yy)) > 1 else np.nan,
        "brier": float(brier_score_loss(yy, ss)),
        "positive_rate": float(np.mean(yy)),
    }


def _profit_factor(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    positive = float(v[v > 0].sum())
    negative = float(-v[v < 0].sum())
    if negative <= 0:
        return np.inf if positive > 0 else np.nan
    return positive / negative


def _max_drawdown(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(v) == 0:
        return np.nan
    equity = np.cumsum(v)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = np.r_[0.0, equity] - peaks
    return float(np.min(drawdown))


def _financial_metrics(selected: pd.DataFrame) -> dict[str, object]:
    if selected.empty:
        return {
            "trades": 0, "long_trades": 0, "short_trades": 0,
            "mean_net_1x_r": np.nan, "median_net_1x_r": np.nan, "sum_net_1x_r": 0.0,
            "mean_net_2x_r": np.nan, "sum_net_2x_r": 0.0, "profit_factor_1x": np.nan,
            "win_rate_1x": np.nan, "positive_month_rate": np.nan, "max_drawdown_r": np.nan,
            "top10_removed_sum_net_1x_r": 0.0, "top10_removed_mean_net_1x_r": np.nan,
        }
    frame = selected.sort_values(["decision_time", "checkpoint_id"], kind="mergesort").copy()
    net1 = pd.to_numeric(frame["chosen_net_1x_r"], errors="coerce")
    net2 = pd.to_numeric(frame["chosen_net_2x_r"], errors="coerce")
    month = pd.to_datetime(frame["decision_time"], errors="coerce").dt.to_period("M")
    monthly = net1.groupby(month).sum(min_count=1)
    top_removed = net1.dropna().sort_values(ascending=False).iloc[min(10, int(net1.notna().sum())):]
    return {
        "trades": int(len(frame)),
        "long_trades": int(frame["chosen_direction"].eq("LONG").sum()),
        "short_trades": int(frame["chosen_direction"].eq("SHORT").sum()),
        "mean_net_1x_r": float(net1.mean()),
        "median_net_1x_r": float(net1.median()),
        "sum_net_1x_r": float(net1.sum()),
        "mean_net_2x_r": float(net2.mean()),
        "sum_net_2x_r": float(net2.sum()),
        "profit_factor_1x": float(_profit_factor(net1)),
        "win_rate_1x": float((net1 > 0).mean()),
        "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "max_drawdown_r": _max_drawdown(net1),
        "top10_removed_sum_net_1x_r": float(top_removed.sum()) if len(top_removed) else 0.0,
        "top10_removed_mean_net_1x_r": float(top_removed.mean()) if len(top_removed) else np.nan,
    }


def _choose_trades(predictions: pd.DataFrame, long_threshold: float, short_threshold: float) -> pd.DataFrame:
    frame = predictions.copy()
    long_score = pd.to_numeric(frame["long_score"], errors="coerce")
    short_score = pd.to_numeric(frame["short_score"], errors="coerce")
    long_ok = long_score >= long_threshold
    short_ok = short_score >= short_threshold
    long_margin = (long_score - long_threshold) / max(EPS, 1.0 - long_threshold)
    short_margin = (short_score - short_threshold) / max(EPS, 1.0 - short_threshold)
    choose_long = long_ok & (~short_ok | (long_margin >= short_margin))
    choose_short = short_ok & ~choose_long
    frame["chosen_direction"] = np.select([choose_long, choose_short], ["LONG", "SHORT"], default="SKIP")
    frame["chosen_score"] = np.where(choose_long, long_score, np.where(choose_short, short_score, np.nan))
    for metric in ("gross_r", "net_1x_r", "net_2x_r"):
        frame[f"chosen_{metric}"] = np.where(
            choose_long,
            pd.to_numeric(frame[f"long_{metric}"], errors="coerce"),
            np.where(choose_short, pd.to_numeric(frame[f"short_{metric}"], errors="coerce"), np.nan),
        )
    return frame.loc[frame["chosen_direction"].ne("SKIP")].copy()


def _score_deciles(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    local = predictions.loc[predictions["split"].eq(split)].copy()
    for direction in ("long", "short"):
        score = pd.to_numeric(local[f"{direction}_score"], errors="coerce")
        valid = score.notna()
        if valid.sum() < 20:
            continue
        ranks = pd.qcut(score.loc[valid].rank(method="first"), q=10, labels=False, duplicates="drop")
        temp = local.loc[valid].copy()
        temp["score_decile"] = ranks.to_numpy() + 1
        for decile, group in temp.groupby("score_decile", sort=True):
            net = pd.to_numeric(group[f"{direction}_net_1x_r"], errors="coerce")
            rows.append({
                "split": split,
                "direction": direction.upper(),
                "score_decile": int(decile),
                "events": len(group),
                "mean_score": float(pd.to_numeric(group[f"{direction}_score"], errors="coerce").mean()),
                "mean_net_1x_r": float(net.mean()),
                "profit_factor_1x": float(_profit_factor(net)),
                "win_rate_1x": float((net > 0).mean()),
            })
    return pd.DataFrame(rows)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _module_split_coverage(frame: pd.DataFrame, present_column: str) -> dict[str, float]:
    if present_column not in frame.columns:
        return {split: 0.0 for split in ("TRAIN", "VALIDATION", "HOLDOUT")}
    present = _coerce_bool(frame[present_column])
    return {
        split: float(present.loc[frame["split"].eq(split)].mean()) if frame["split"].eq(split).any() else 0.0
        for split in ("TRAIN", "VALIDATION", "HOLDOUT")
    }


def run_supervised_ablation(
    datasets: dict[int, pd.DataFrame],
    base_columns: dict[int, list[str]],
    dynamic_columns: dict[int, list[str]],
    modules: dict[str, FeatureModuleResult],
    config: PostSweepSupervisedConfig,
    *,
    progress_callback: Callable[[int, int, int, str, str], None] | None = None,
) -> ModelingResult:
    cfg = config.validate()
    model_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_samples: list[pd.DataFrame] = []
    decile_rows: list[pd.DataFrame] = []
    contract_rows: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    total_steps = sum(5 if minutes == 0 else 6 for minutes in datasets) * 2
    done = 0

    def advance(minutes: int, ablation: str, model_kind: str) -> None:
        nonlocal done
        done += 1
        if progress_callback:
            progress_callback(done, total_steps, minutes, ablation, model_kind)

    for minutes in sorted(datasets):
        frame = datasets[minutes].copy()
        # Attach every module once; cumulative feature contracts decide what is used.
        module_columns: dict[str, list[str]] = {}
        for module_name, module in modules.items():
            if module.features.empty or "checkpoint_id" not in module.features.columns:
                module_columns[module_name] = []
                continue
            before = set(frame.columns)
            frame = frame.merge(module.features, on="checkpoint_id", how="left", validate="one_to_one")
            module_columns[module_name] = [name for name in frame.columns if name not in before and name != "checkpoint_id"]

        current_features = list(base_columns.get(minutes, []))
        blocked_reason: str | None = None
        for ablation, addition in ABLATION_STEPS:
            if minutes == 0 and ablation == "B_R09_R12":
                continue
            if addition == "dynamic":
                current_features.extend(dynamic_columns.get(minutes, []))
            elif addition in modules:
                if blocked_reason is None:
                    present_column = MODULE_PRESENT_COLUMNS[addition]
                    coverage = _module_split_coverage(frame, present_column)
                    if min(coverage.values()) < cfg.minimum_module_coverage:
                        blocked_reason = f"{addition}_coverage_below_{cfg.minimum_module_coverage:.2f}:{coverage}"
                    else:
                        current_features.extend(module_columns.get(addition, []))
            if blocked_reason is not None:
                for model_kind in ("LOGISTIC", "HGB"):
                    model_rows.append({
                        "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                        "direction": "BOTH", "status": "SKIPPED", "reason": blocked_reason,
                    })
                    advance(minutes, ablation, model_kind)
                continue

            train_mask = frame["split"].eq("TRAIN")
            validation_mask = frame["split"].eq("VALIDATION")
            holdout_mask = frame["split"].eq("HOLDOUT")
            split_counts = (int(train_mask.sum()), int(validation_mask.sum()), int(holdout_mask.sum()))
            if split_counts[0] < cfg.minimum_train_events or split_counts[1] < cfg.minimum_validation_events or split_counts[2] < cfg.minimum_holdout_events:
                reason = f"split_events_below_gate:{split_counts}"
                for model_kind in ("LOGISTIC", "HGB"):
                    model_rows.append({
                        "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                        "direction": "BOTH", "status": "SKIPPED", "reason": reason,
                    })
                    advance(minutes, ablation, model_kind)
                continue

            x, numeric, categorical, contract = _prepare_feature_frame(frame, current_features, train_mask)
            contract["checkpoint_minutes"] = minutes
            contract["ablation"] = ablation
            contract_rows.append(contract)
            if x.shape[1] == 0:
                for model_kind in ("LOGISTIC", "HGB"):
                    model_rows.append({
                        "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                        "direction": "BOTH", "status": "SKIPPED", "reason": "no_usable_features",
                    })
                    advance(minutes, ablation, model_kind)
                continue

            # Logistic is a stable linear baseline. HGB is the primary nonlinear model.
            for model_kind in ("LOGISTIC", "HGB"):
                scores: dict[str, np.ndarray] = {}
                direction_ok = True
                # Transform once per ablation/model kind, then reuse the dense
                # matrix for Long and Short. This avoids duplicate one-hot and
                # imputation work and bounds peak memory.
                with threadpool_limits(limits=1):
                    preprocessor = _preprocessor(model_kind, numeric, categorical)
                    x_train = preprocessor.fit_transform(x.loc[train_mask])
                    x_validation = preprocessor.transform(x.loc[validation_mask])
                    x_holdout = preprocessor.transform(x.loc[holdout_mask])
                    transformed_features = int(x_train.shape[1])
                    for direction in ("long", "short"):
                        label_name = f"{direction}_profitable_label"
                        y = frame[label_name].astype("boolean")
                        y_train = y.loc[train_mask]
                        valid_train = y_train.notna().to_numpy()
                        y_train_valid = y_train.loc[y_train.notna()]
                        if y_train_valid.nunique() < 2:
                            model_rows.append({
                                "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                                "direction": direction.upper(), "status": "SKIPPED", "reason": "single_train_class",
                            })
                            direction_ok = False
                            break
                        estimator = _estimator(model_kind, cfg)
                        estimator.fit(x_train[valid_train], y_train_valid.astype(int).to_numpy())
                        score = np.full(len(frame), np.nan, dtype=float)
                        val_positions = validation_mask.to_numpy()
                        hold_positions = holdout_mask.to_numpy()
                        score[val_positions] = estimator.predict_proba(x_validation)[:, 1]
                        score[hold_positions] = estimator.predict_proba(x_holdout)[:, 1]
                        # Ambiguous or unavailable path labels may not enter either
                        # classification metrics or the financial selection cohort.
                        score[~y.notna().to_numpy()] = np.nan
                        scores[direction] = score
                        val_metrics = _classification_metrics(y.loc[validation_mask], score[val_positions])
                        hold_metrics = _classification_metrics(y.loc[holdout_mask], score[hold_positions])
                        model_rows.append({
                            "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                            "direction": direction.upper(), "status": "COMPLETE", "reason": "",
                            "features": x.shape[1], "transformed_features": transformed_features,
                            "numeric_features": len(numeric), "categorical_features": len(categorical),
                            "train_events": split_counts[0], "train_labeled_events": int(valid_train.sum()),
                            "validation_events": split_counts[1], "holdout_events": split_counts[2],
                            "train_positive_rate": float(y_train_valid.mean()),
                            **{f"validation_{k}": v for k, v in val_metrics.items()},
                            **{f"holdout_{k}": v for k, v in hold_metrics.items()},
                        })
                        del estimator
                advance(minutes, ablation, model_kind)
                del preprocessor, x_train, x_validation, x_holdout
                gc.collect()
                if not direction_ok:
                    continue

                predictions = frame[[
                    "checkpoint_id", "zone_event_id", "decision_time", "split",
                    "long_gross_r", "long_net_1x_r", "long_net_2x_r",
                    "short_gross_r", "short_net_1x_r", "short_net_2x_r",
                ]].copy()
                predictions["long_score"] = scores["long"]
                predictions["short_score"] = scores["short"]
                predictions["checkpoint_minutes"] = minutes
                predictions["ablation"] = ablation
                predictions["model"] = model_kind
                prediction_samples.append(predictions.loc[predictions["split"].isin(["VALIDATION", "HOLDOUT"])].head(2_000))
                deciles = pd.concat([
                    _score_deciles(predictions, "VALIDATION"),
                    _score_deciles(predictions, "HOLDOUT"),
                ], ignore_index=True)
                if not deciles.empty:
                    deciles["checkpoint_minutes"] = minutes
                    deciles["ablation"] = ablation
                    deciles["model"] = model_kind
                    decile_rows.append(deciles)

                validation = predictions.loc[predictions["split"].eq("VALIDATION")]
                holdout = predictions.loc[predictions["split"].eq("HOLDOUT")]
                for quantile in cfg.score_quantiles:
                    long_threshold = float(pd.to_numeric(validation["long_score"], errors="coerce").quantile(quantile))
                    short_threshold = float(pd.to_numeric(validation["short_score"], errors="coerce").quantile(quantile))
                    selected_validation = _choose_trades(validation, long_threshold, short_threshold)
                    selected_holdout = _choose_trades(holdout, long_threshold, short_threshold)
                    for split_name, selected in (("VALIDATION", selected_validation), ("HOLDOUT", selected_holdout)):
                        metrics = _financial_metrics(selected)
                        selection_rows.append({
                            "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                            "score_quantile": quantile, "split": split_name,
                            "long_score_threshold": long_threshold, "short_score_threshold": short_threshold,
                            **metrics,
                        })
                    if quantile == cfg.primary_score_quantile:
                        val_metrics = _financial_metrics(selected_validation)
                        hold_metrics = _financial_metrics(selected_holdout)
                        reasons = []
                        if model_kind != "HGB": reasons.append("non_primary_model")
                        if hold_metrics["trades"] < cfg.minimum_holdout_trades: reasons.append("holdout_trades")
                        if not np.isfinite(hold_metrics["mean_net_1x_r"]) or hold_metrics["mean_net_1x_r"] <= 0: reasons.append("holdout_mean_1x")
                        if not np.isfinite(hold_metrics["profit_factor_1x"]) or hold_metrics["profit_factor_1x"] < cfg.minimum_pf: reasons.append("holdout_pf")
                        if not np.isfinite(hold_metrics["mean_net_2x_r"]) or hold_metrics["mean_net_2x_r"] <= 0: reasons.append("holdout_2x")
                        if not np.isfinite(hold_metrics["positive_month_rate"]) or hold_metrics["positive_month_rate"] < cfg.minimum_positive_month_rate: reasons.append("positive_month_rate")
                        if hold_metrics["top10_removed_sum_net_1x_r"] <= 0: reasons.append("top10_removed")
                        if not np.isfinite(val_metrics["mean_net_1x_r"]) or val_metrics["mean_net_1x_r"] <= 0: reasons.append("validation_mean")
                        decision = "promote_to_backtest" if not reasons else "rejected"
                        decision_rows.append({
                            "checkpoint_minutes": minutes, "ablation": ablation, "model": model_kind,
                            "score_quantile": quantile, "decision": decision,
                            "failed_gates": ",".join(reasons),
                            "validation_trades": val_metrics["trades"],
                            "validation_mean_net_1x_r": val_metrics["mean_net_1x_r"],
                            "holdout_trades": hold_metrics["trades"],
                            "holdout_mean_net_1x_r": hold_metrics["mean_net_1x_r"],
                            "holdout_mean_net_2x_r": hold_metrics["mean_net_2x_r"],
                            "holdout_pf": hold_metrics["profit_factor_1x"],
                            "holdout_positive_month_rate": hold_metrics["positive_month_rate"],
                            "holdout_top10_removed_sum": hold_metrics["top10_removed_sum_net_1x_r"],
                        })

    return ModelingResult(
        model_summary=pd.DataFrame(model_rows),
        selection_summary=pd.DataFrame(selection_rows),
        prediction_sample=pd.concat(prediction_samples, ignore_index=True, sort=False) if prediction_samples else pd.DataFrame(),
        score_deciles=pd.concat(decile_rows, ignore_index=True, sort=False) if decile_rows else pd.DataFrame(),
        feature_contract=pd.concat(contract_rows, ignore_index=True, sort=False) if contract_rows else pd.DataFrame(),
        decision_summary=pd.DataFrame(decision_rows),
    )
