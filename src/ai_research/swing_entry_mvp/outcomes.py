#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact first-hit labels over the existing R03 1m cache.

The original R03 label used future extrema. R03.1 instead records whether the
3%/5% target was reached before the adverse boundary. If both are touched in
one minute, the adverse boundary wins conservatively.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None  # type: ignore[assignment]
    prange = range  # type: ignore[assignment]

from src.ai_research.swing_baseline.config import SwingTargetSpec
from src.ai_research.swing_baseline.dataset import SwingYearShard, load_year_shard
from src.ai_research.swing_baseline.modeling import PeriodData, collect_period_data
from src.research_common.progress import ProgressReporter

from .config import SwingEntryMvpConfig


OUTCOME_SCHEMA_VERSION = 1


if njit is not None:

    @njit(cache=True, parallel=True)
    def _first_hit_kernel(
        entry_positions: np.ndarray,
        entry_prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        target_move: float,
        adverse_move: float,
        horizon_bars: int,
        direction: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(entry_positions)
        quality = np.zeros(n, dtype=np.float32)
        event_code = np.zeros(n, dtype=np.int8)  # 0 timeout, 1 target, 2 adverse, 3 same-bar adverse-first
        bars_to_event = np.full(n, -1, dtype=np.int32)
        for i in prange(n):
            start = int(entry_positions[i])
            entry = float(entry_prices[i])
            if start < 0 or start >= len(high) or not np.isfinite(entry) or entry <= 0:
                quality[i] = np.nan
                continue
            end = min(len(high), start + horizon_bars)
            if direction > 0:
                target_price = entry * (1.0 + target_move)
                adverse_price = entry * (1.0 - adverse_move)
                for j in range(start, end):
                    target_hit = high[j] >= target_price
                    adverse_hit = low[j] <= adverse_price
                    if adverse_hit:
                        event_code[i] = 3 if target_hit else 2
                        bars_to_event[i] = j - start
                        break
                    if target_hit:
                        quality[i] = 1.0
                        event_code[i] = 1
                        bars_to_event[i] = j - start
                        break
            else:
                target_price = entry * (1.0 - target_move)
                adverse_price = entry * (1.0 + adverse_move)
                for j in range(start, end):
                    target_hit = low[j] <= target_price
                    adverse_hit = high[j] >= adverse_price
                    if adverse_hit:
                        event_code[i] = 3 if target_hit else 2
                        bars_to_event[i] = j - start
                        break
                    if target_hit:
                        quality[i] = 1.0
                        event_code[i] = 1
                        bars_to_event[i] = j - start
                        break
        return quality, event_code, bars_to_event


def validate_outcome_dependency() -> dict[str, str]:
    if njit is None:
        raise RuntimeError(
            "R03.1 startup dependency check failed: numba is not installed.\n"
            "Install it before any exact-label work with:\n"
            "  python -m pip install numba\n"
            "Then rerun the same command. Existing R03 feature caches remain reusable."
        )
    return {"numba": "available"}


def _base_shard_signature(shard: SwingYearShard) -> str:
    base_manifest = json.loads((shard.path / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "path": str(shard.path.name),
        "cache_signature": base_manifest.get("cache_signature"),
        "rows": int(len(shard.decision_times_ns)),
        "minute_rows": int(len(shard.minute_times_ns)),
        "start": int(shard.decision_times_ns[0]),
        "end": int(shard.decision_times_ns[-1]),
        "minute_start": int(shard.minute_times_ns[0]),
        "minute_end": int(shard.minute_times_ns[-1]),
        "feature_columns": list(shard.full_feature_columns),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def outcome_signature(shard: SwingYearShard, config: SwingEntryMvpConfig) -> str:
    payload = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "base_shard": _base_shard_signature(shard),
        "targets": [target.to_dict() for target in config.base.target_specs],
        "execution_delay_minutes": config.base.execution_delay_minutes,
        "same_bar_policy": config.same_bar_policy,
        "label_definition": "target_before_fixed_adverse_first_hit",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def outcome_columns(config: SwingEntryMvpConfig) -> tuple[str, ...]:
    columns: list[str] = []
    for target in config.base.target_specs:
        for direction in ("long", "short"):
            columns.extend(
                [
                    f"{target.target_id}_{direction}_quality",
                    f"{target.target_id}_{direction}_event_code",
                    f"{target.target_id}_{direction}_bars_to_event",
                ]
            )
    return tuple(columns)


def overlay_path(root: Path, shard: SwingYearShard) -> Path:
    year = pd.Timestamp(int(shard.decision_times_ns[0])).year
    return root / f"exact_outcomes_{year}"


def _entry_positions(shard: SwingYearShard) -> np.ndarray:
    positions = np.searchsorted(shard.minute_times_ns, shard.entry_times_ns, side="left").astype(np.int64)
    invalid = (positions >= len(shard.minute_times_ns))
    positions[invalid] = -1
    return positions


def build_outcome_overlay(
    shard_path: Path,
    config: SwingEntryMvpConfig,
    *,
    force_rebuild: bool = False,
) -> Path:
    validate_outcome_dependency()
    shard = load_year_shard(shard_path)
    output = overlay_path(config.exact_label_cache_path, shard)
    manifest_path = output / "manifest.json"
    signature = outcome_signature(shard, config)
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature:
                return output
        except (OSError, json.JSONDecodeError):
            pass

    temp = output.with_name(output.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    positions = _entry_positions(shard)
    high = np.asarray(shard.minute_ohlc[:, 1], dtype=np.float64)
    low = np.asarray(shard.minute_ohlc[:, 2], dtype=np.float64)
    entry_prices = np.asarray(shard.entry_prices, dtype=np.float64)
    arrays: list[np.ndarray] = []
    columns: list[str] = []
    diagnostics: list[dict[str, object]] = []
    for target in config.base.target_specs:
        for direction_name, direction in (("long", 1), ("short", -1)):
            quality, event_code, bars_to_event = _first_hit_kernel(
                positions,
                entry_prices,
                high,
                low,
                float(target.target_move),
                float(target.max_adverse_move),
                int(target.horizon_hours * 60),
                direction,
            )
            arrays.extend([quality.astype(np.float32), event_code.astype(np.float32), bars_to_event.astype(np.float32)])
            columns.extend(
                [
                    f"{target.target_id}_{direction_name}_quality",
                    f"{target.target_id}_{direction_name}_event_code",
                    f"{target.target_id}_{direction_name}_bars_to_event",
                ]
            )
            valid = np.isfinite(quality)
            diagnostics.append(
                {
                    "year": int(pd.Timestamp(int(shard.decision_times_ns[0])).year),
                    "base_shard": str(shard.path.name),
                    "target_id": target.target_id,
                    "direction": direction_name,
                    "rows": int(valid.sum()),
                    "positive_rate": float(np.mean(quality[valid])) if valid.any() else float("nan"),
                    "target_events": int(np.sum(event_code == 1)),
                    "adverse_events": int(np.sum(event_code == 2)),
                    "ambiguous_adverse_first": int(np.sum(event_code == 3)),
                    "timeouts": int(np.sum(event_code == 0)),
                }
            )
    matrix = np.column_stack(arrays).astype(np.float32, copy=False)
    np.save(temp / "decision_times_ns.npy", np.asarray(shard.decision_times_ns, dtype=np.int64), allow_pickle=False)
    np.save(temp / "outcomes.npy", matrix, allow_pickle=False)
    manifest = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "signature": signature,
        "base_shard": str(shard.path),
        "rows": int(len(shard.decision_times_ns)),
        "columns": columns,
        "diagnostics": diagnostics,
        "same_bar_policy": config.same_bar_policy,
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    temp.replace(output)
    return output


def build_outcome_overlays(
    shard_paths: Iterable[Path],
    config: SwingEntryMvpConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.exact_label_cache_path.mkdir(parents=True, exist_ok=True)
    paths = list(shard_paths)
    reporter = ProgressReporter("[R03.1 exact labels] years", len(paths), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, path in enumerate(paths, start=1):
        outputs.append(build_outcome_overlay(path, config, force_rebuild=force_rebuild))
        reporter.update(index)
    reporter.close()
    return outputs


@dataclass(frozen=True)
class OutcomeOverlay:
    path: Path
    decision_times_ns: np.ndarray
    outcomes: np.ndarray
    columns: tuple[str, ...]

    @property
    def column_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.columns)}

    def positions(self, start: pd.Timestamp, end: pd.Timestamp) -> slice:
        start_ns = int(pd.Timestamp(start).value)
        end_ns = int(pd.Timestamp(end).value)
        left = int(np.searchsorted(self.decision_times_ns, start_ns, side="left"))
        right = int(np.searchsorted(self.decision_times_ns, end_ns, side="right"))
        return slice(left, right)


def load_outcome_overlay(path: Path) -> OutcomeOverlay:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return OutcomeOverlay(
        path=path,
        decision_times_ns=np.load(path / "decision_times_ns.npy", mmap_mode="r"),
        outcomes=np.load(path / "outcomes.npy", mmap_mode="r"),
        columns=tuple(manifest["columns"]),
    )


def collect_exact_period_data(
    shard_paths: Iterable[Path],
    overlay_paths: Iterable[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    target: SwingTargetSpec,
) -> PeriodData:
    shard_paths = list(shard_paths)
    overlays = [load_outcome_overlay(path) for path in overlay_paths]
    base = collect_period_data(
        shard_paths,
        start,
        end,
        label_names=(
            f"{target.target_id}_long_quality",
            f"{target.target_id}_short_quality",
        ),
    )
    exact_parts: dict[str, list[np.ndarray]] = {
        f"{target.target_id}_long_quality": [],
        f"{target.target_id}_short_quality": [],
    }
    time_parts: list[np.ndarray] = []
    for overlay in overlays:
        positions = overlay.positions(start, end)
        if int(positions.stop or 0) <= int(positions.start or 0):
            continue
        time_parts.append(np.asarray(overlay.decision_times_ns[positions], dtype=np.int64))
        index = overlay.column_index
        for name in exact_parts:
            exact_parts[name].append(np.asarray(overlay.outcomes[positions, index[name]], dtype=np.float32))
    if not time_parts:
        raise RuntimeError(f"no R03.1 exact outcomes for {start} -> {end}")
    exact_times = np.concatenate(time_parts)
    order = np.argsort(exact_times, kind="stable")
    exact_times = exact_times[order]
    if not np.array_equal(exact_times, base.timestamps_ns):
        raise RuntimeError("R03.1 exact-outcome timestamps do not match the frozen R03 feature cache")
    exact_labels = {name: np.concatenate(parts)[order] for name, parts in exact_parts.items()}
    return PeriodData(
        timestamps_ns=base.timestamps_ns,
        high_x=base.high_x,
        full_x=base.full_x,
        labels=exact_labels,
        context=base.context,
        context_columns=base.context_columns,
        entry_times_ns=base.entry_times_ns,
        entry_prices=base.entry_prices,
    )


def overlay_manifest_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        rows.extend(manifest.get("diagnostics", []))
    return rows
