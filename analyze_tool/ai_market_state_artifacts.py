#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read/write visualization artifacts for the R03.3.3.1 market-state model.

The Analyze Tool must stay a visualization surface.  It reads the frozen state
cache directly and reads full out-of-sample activity-persistence predictions
from a small derivative artifact built by ``tools/prebuild_ai_market_state_timeline.py``.
It never refits a model during an HTTP request.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ai_research.market_state_continuity.config import (
    DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
    MarketStateContinuityConfig,
)
from src.ai_research.market_state_continuity.state_cache import (
    list_state_caches,
    load_state_year_shard,
    ns_to_datetime,
)

ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_DIR = Path("data/cache/analyze_tool/ai_market_state_r03_3_3_1")
ACTIVITY_TARGET = "activity_persist_h3"
ACTIVITY_ARCHITECTURE = "universal_ohlcv_lightgbm"


@dataclass(frozen=True)
class ActivityPredictionShard:
    year: int
    fold_id: str
    decision_times_ns: np.ndarray
    prediction: np.ndarray
    actual_persist: np.ndarray
    path: Path

    @property
    def index(self) -> pd.DatetimeIndex:
        return ns_to_datetime(self.decision_times_ns)


def _artifact_path(artifact_dir: Path, year: int) -> Path:
    return artifact_dir / f"activity_persist_oos_{int(year)}"


def _write_prediction_shard(
    *,
    artifact_dir: Path,
    year: int,
    fold_id: str,
    decision_times_ns: np.ndarray,
    prediction: np.ndarray,
    actual_persist: np.ndarray,
    config: MarketStateContinuityConfig,
) -> Path:
    target = _artifact_path(artifact_dir, year)
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)

    timestamps = np.asarray(decision_times_ns, dtype=np.int64)
    probability = np.asarray(prediction, dtype=np.float32)
    actual = np.asarray(actual_persist, dtype=np.int8)
    if not (len(timestamps) == len(probability) == len(actual)):
        raise ValueError("activity prediction artifact arrays must have identical lengths")
    decoded = ns_to_datetime(timestamps)
    if len(decoded) and not np.all(np.asarray(decoded.year, dtype=np.int16) == int(year)):
        raise RuntimeError(f"activity prediction artifact year mismatch for {year}")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0.0) | (probability > 1.0)):
        raise RuntimeError("activity persistence predictions must be finite probabilities")

    np.save(temp / "decision_times_ns.npy", timestamps, allow_pickle=False)
    np.save(temp / "prediction.npy", probability, allow_pickle=False)
    np.save(temp / "actual_persist.npy", actual, allow_pickle=False)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "year": int(year),
        "fold_id": str(fold_id),
        "target": ACTIVITY_TARGET,
        "architecture": ACTIVITY_ARCHITECTURE,
        "timestamp_unit": "ns",
        "rows": int(len(timestamps)),
        "timestamp_min": str(decoded.min()) if len(decoded) else None,
        "timestamp_max": str(decoded.max()) if len(decoded) else None,
        "source_state_cache": str(config.cache_path),
        "sealed_2026": True,
        "not_trade_signal": True,
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    temp.replace(target)
    return target


def load_activity_prediction_shard(path: str | Path) -> ActivityPredictionShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported Analyze Tool state artifact: {target}")
    if manifest.get("target") != ACTIVITY_TARGET:
        raise RuntimeError(f"unexpected state artifact target: {target}")
    if manifest.get("architecture") != ACTIVITY_ARCHITECTURE:
        raise RuntimeError(f"unexpected state artifact architecture: {target}")
    year = int(manifest["year"])
    times = np.load(target / "decision_times_ns.npy", mmap_mode="r")
    decoded = ns_to_datetime(times)
    if len(decoded) and not np.all(np.asarray(decoded.year, dtype=np.int16) == year):
        raise RuntimeError(f"Analyze Tool state artifact timestamp/year mismatch: {target}")
    return ActivityPredictionShard(
        year=year,
        fold_id=str(manifest["fold_id"]),
        decision_times_ns=times,
        prediction=np.load(target / "prediction.npy", mmap_mode="r"),
        actual_persist=np.load(target / "actual_persist.npy", mmap_mode="r"),
        path=target,
    )


def list_activity_prediction_shards(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> list[Path]:
    root = Path(artifact_dir)
    return sorted(
        path
        for path in root.glob("activity_persist_oos_????")
        if (path / "manifest.json").exists()
    )


def build_activity_prediction_artifacts(
    *,
    config: MarketStateContinuityConfig = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    force_rebuild: bool = False,
) -> list[Path]:
    """Fit only the validated activity-continuity model and persist full OOS predictions."""
    # Keep LightGBM/modeling imports out of Analyze Tool server startup.
    from src.ai_research.market_state_continuity.modeling import (
        collect_continuity_period_data,
        default_continuity_folds,
        fit_continuity_model,
        validate_continuity_dependencies,
    )

    config.validate()
    validate_continuity_dependencies()
    state_paths = list_state_caches(config)
    if not state_paths:
        raise RuntimeError(
            "R03.3.3.1 state cache is missing. Run "
            "python research/eth_ai_trading/03_3_3_1_market_state_continuity_audit.py first."
        )
    available_years = {load_state_year_shard(path).year for path in state_paths}
    required_years = set(range(2021, 2026))
    missing = sorted(required_years - available_years)
    if missing:
        raise RuntimeError(f"R03.3.3.1 state cache missing years: {missing}")

    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for fold in default_continuity_folds(config):
        year = int(fold.test_start.year)
        target = _artifact_path(root, year)
        if target.exists() and (target / "manifest.json").exists() and not force_rebuild:
            try:
                outputs.append(load_activity_prediction_shard(target).path)
                print(f"[skip] {fold.fold_id} activity timeline already exists rows={load_activity_prediction_shard(target).prediction.shape[0]:,}")
                continue
            except Exception:
                pass

        print(f"[fit] {fold.fold_id} {ACTIVITY_TARGET} {ACTIVITY_ARCHITECTURE}")
        fit = collect_continuity_period_data(
            state_paths,
            [],
            start=fold.fit_start,
            end=fold.fit_end,
            target=ACTIVITY_TARGET,
            architecture=ACTIVITY_ARCHITECTURE,
        )
        test = collect_continuity_period_data(
            state_paths,
            [],
            start=fold.test_start,
            end=fold.test_end,
            target=ACTIVITY_TARGET,
            architecture=ACTIVITY_ARCHITECTURE,
        )
        model = fit_continuity_model(fit, config)
        prediction = np.asarray(model.predict_proba(test.x)[:, 1], dtype=float)
        output = _write_prediction_shard(
            artifact_dir=root,
            year=year,
            fold_id=fold.fold_id,
            decision_times_ns=test.timestamps_ns,
            prediction=prediction,
            actual_persist=test.y,
            config=config,
        )
        outputs.append(output)
        print(
            f"[done] {fold.fold_id} rows={len(prediction):,} "
            f"range={test.index.min()} -> {test.index.max()} path={output}"
        )

    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage_id": "R03.3.3.1",
        "target": ACTIVITY_TARGET,
        "architecture": ACTIVITY_ARCHITECTURE,
        "years": [int(load_activity_prediction_shard(path).year) for path in outputs],
        "state_cache": str(config.cache_path),
        "not_trade_signal": True,
        "sealed_2026": True,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def load_state_frame_for_range(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: MarketStateContinuityConfig = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
) -> pd.DataFrame:
    """Load full 15m state rows covering ``start`` through ``end`` from mmap caches."""
    columns_needed = (
        "strategic_score",
        "tactical_score",
        "entry_score",
        "activity_score",
        "strategic_state",
        "tactical_state",
        "entry_state",
        "activity_state",
        "strategic_boundary_margin",
        "tactical_boundary_margin",
        "entry_boundary_margin",
        "activity_boundary_margin",
        "strategic_age_bars",
        "tactical_age_bars",
        "entry_age_bars",
        "activity_age_bars",
        "strategic_flip_rate_6h",
        "tactical_flip_rate_6h",
        "entry_flip_rate_6h",
        "activity_flip_rate_6h",
        "strategic_tactical_alignment",
        "tactical_entry_alignment",
        "all_direction_alignment",
        "long_pullback_setup",
        "short_pullback_setup",
        "trend_momentum_long",
        "trend_momentum_short",
    )
    parts: list[pd.DataFrame] = []
    for path in list_state_caches(config):
        shard = load_state_year_shard(path)
        if shard.year < start.year or shard.year > end.year:
            continue
        times = np.asarray(shard.decision_times_ns, dtype=np.int64)
        left = int(np.searchsorted(times, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(times, int(pd.Timestamp(end).value), side="right"))
        if right <= left:
            continue
        missing = [column for column in columns_needed if column not in shard.state_index]
        if missing:
            raise RuntimeError(f"R03.3.3.1 state cache missing visualization columns: {missing}")
        positions = [shard.state_index[column] for column in columns_needed]
        matrix = np.asarray(shard.states[left:right, positions], dtype=np.float32)
        part = pd.DataFrame(matrix, index=ns_to_datetime(times[left:right]), columns=columns_needed)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=columns_needed)
    frame = pd.concat(parts).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def load_activity_prediction_frame(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in list_activity_prediction_shards(artifact_dir):
        shard = load_activity_prediction_shard(path)
        if shard.year < start.year or shard.year > end.year:
            continue
        times = np.asarray(shard.decision_times_ns, dtype=np.int64)
        left = int(np.searchsorted(times, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(times, int(pd.Timestamp(end).value), side="right"))
        if right <= left:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "activity_persist_h3_probability": np.asarray(shard.prediction[left:right], dtype=float),
                    "activity_transition_risk_h3": 1.0 - np.asarray(shard.prediction[left:right], dtype=float),
                    "activity_actual_persist_h3": np.asarray(shard.actual_persist[left:right], dtype=float),
                    "prediction_fold": shard.fold_id,
                },
                index=ns_to_datetime(times[left:right]),
            )
        )
    if not parts:
        return pd.DataFrame(
            columns=(
                "activity_persist_h3_probability",
                "activity_transition_risk_h3",
                "activity_actual_persist_h3",
                "prediction_fold",
            )
        )
    frame = pd.concat(parts).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]
