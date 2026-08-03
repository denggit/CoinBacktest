#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load R09/R12 report caches and construct leakage-safe R13 datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PostSweepSupervisedConfig

R09_FEATURE_FILE = "16_zone_feature_table.csv.gz"
R09_LABEL_FILE = "17_zone_label_table.csv.gz"
R12_FEATURE_FILE = "14_checkpoint_feature_table.csv.gz"
R12_LABEL_FILE = "15_outcome_label_table.csv.gz"

R09_RELEASE_FEATURES: tuple[str, ...] = (
    "release_event_bar_downside_bp",
    "release_event_bar_close_off_low_bp",
    "release_sell_notional_1m_vs_prior60",
    "release_trades_count_1m_vs_prior60",
    "release_large_sell_notional_1m_vs_prior60",
    "release_large_sell_count_1m_vs_prior60",
    "release_sell_share_1m",
    "release_negative_delta_ratio_1m",
    "release_price_downside_1m_bp",
    "release_terminal_return_1m_bp",
    "release_sell_impact_bp_per_million_1m",
    "release_sell_notional_5m_vs_prior60",
    "release_trades_count_5m_vs_prior60",
    "release_large_sell_notional_5m_vs_prior60",
    "release_large_sell_count_5m_vs_prior60",
    "release_sell_share_5m",
    "release_negative_delta_ratio_5m",
    "release_price_downside_5m_bp",
    "release_terminal_return_5m_bp",
    "release_sell_impact_bp_per_million_5m",
    "release_sell_notional_15m_vs_prior60",
    "release_trades_count_15m_vs_prior60",
    "release_large_sell_notional_15m_vs_prior60",
    "release_large_sell_count_15m_vs_prior60",
    "release_sell_share_15m",
    "release_negative_delta_ratio_15m",
    "release_price_downside_15m_bp",
    "release_terminal_return_15m_bp",
    "release_sell_impact_bp_per_million_15m",
    "release_max_trade_notional_1m_vs_prior60",
    "release_baseline_available",
    "stop_release_score",
    "high_stop_release_label",
)

META_COLUMNS: tuple[str, ...] = (
    "checkpoint_id", "zone_event_id", "checkpoint_minutes", "decision_time",
    "entry_time", "event_available_time", "event_bar_time", "period", "split",
)

ABSOLUTE_OR_IDENTIFIER_TOKENS: tuple[str, ...] = ("_id", "_pos", "_time", "_ts")
LABEL_TOKENS: tuple[str, ...] = (
    "future_", "outcome", "target_", "stop_hit", "target_hit", "gross_",
    "net_", "exit_", "stopped", "horizon_end", "label",
)
ABSOLUTE_PRICE_SUFFIXES: tuple[str, ...] = ("_price", "_open", "_high", "_low", "_close", "_abs")
ABSOLUTE_PRICE_NAMES: frozenset[str] = frozenset({
    "checkpoint_close", "path_low_visible", "path_high_visible",
    "pre_reclaim_low_visible", "after_reclaim_low_visible",
})


def _is_absolute_price_level(name: str) -> bool:
    low = str(name).lower()
    return low in {"open", "high", "low", "close", "price", *ABSOLUTE_PRICE_NAMES} or low.endswith(ABSOLUTE_PRICE_SUFFIXES)


def _release_feature_available(name: str, checkpoint_minutes: int) -> bool:
    """Return whether an R09 release field is visible at the decision time.

    R09's event-bar/1m release measurements are known at the first executable
    minute after the sweep.  Its frozen release score and 5m fields require five
    completed bars.  R13 has no 15m checkpoint, so 15m release fields are never
    admitted.
    """
    low = str(name).lower()
    if "release_" not in low and low not in {"stop_release_score", "high_stop_release_feature", "high_release"}:
        return True
    if "_15m" in low:
        return False
    if "_5m" in low or low in {"stop_release_score", "high_stop_release_feature", "high_release"}:
        return int(checkpoint_minutes) >= 5
    return True


def _profitable_label(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    valid = numeric.notna()
    out.loc[valid] = numeric.loc[valid].ge(float(threshold)).to_numpy()
    return out


def _resolved_structural_label(
    values: pd.Series,
    outcomes: pd.Series,
    threshold: float,
    *,
    target_token: str = "TARGET",
) -> pd.Series:
    """Label only structurally resolved target/stop paths.

    TIME/INVALID rows are censored instead of being learned as forced time exits.
    A positive row must hit the frozen structural target and still clear the
    real-cost net-R threshold; a stop row is negative.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = outcomes.astype("string").str.upper().str.strip()
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    target = normalized.eq(target_token) & numeric.notna()
    stopped = normalized.str.startswith("STOP", na=False) & numeric.notna()
    out.loc[target] = numeric.loc[target].ge(float(threshold)).to_numpy()
    out.loc[stopped] = False
    return out


def _safe_base_feature(name: str) -> bool:
    low = name.lower()
    if name in {"zone_primary_timeframe", "zone_timeframes"}:
        return True
    if name in {"event_kind", "period", "matched_zone_event_id"}:
        return False
    if any(token in low for token in LABEL_TOKENS):
        return False
    if _is_absolute_price_level(name):
        return False
    if any(token in low for token in ABSOLUTE_OR_IDENTIFIER_TOKENS):
        # Timeframe is a structural scale, not an observation timestamp.
        return "timeframe" in low and not low.endswith(("_time", "_ts"))
    return True


def _safe_dynamic_feature(name: str) -> bool:
    low = name.lower()
    if name in {"state", "state_direction"}:
        return True
    if name in {"high_release", "high_timeframe_zone", "multitimeframe_zone"}:
        return False
    if any(token in low for token in LABEL_TOKENS):
        return False
    if _is_absolute_price_level(name):
        return False
    if any(token in low for token in ABSOLUTE_OR_IDENTIFIER_TOKENS):
        return False
    return True


@dataclass(frozen=True)
class R13DataBundle:
    datasets: dict[int, pd.DataFrame]
    base_columns: dict[int, list[str]]
    dynamic_columns: dict[int, list[str]]
    audit: pd.DataFrame


def resolve_report_dir(path: str | Path, alternatives: Iterable[str] = ()) -> Path:
    requested = Path(path)
    candidates = [requested, *(requested.parent / name for name in alternatives)]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"report directory not found; checked={[str(x) for x in candidates]}")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _parse_times(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in out.columns:
        if name.endswith("_time") or name.endswith("_ts") or name in {
            "event_bar_time", "event_available_time", "checkpoint_available_time", "entry_time",
        }:
            out[name] = pd.to_datetime(out[name], errors="coerce")
    return out


def _split(decision_time: pd.Series, cfg: PostSweepSupervisedConfig) -> pd.Series:
    ts = pd.to_datetime(decision_time, errors="coerce")
    train_end = pd.Timestamp(cfg.train_end_exclusive)
    validation_end = pd.Timestamp(cfg.validation_end_exclusive)
    return pd.Series(
        np.select(
            [ts < train_end, (ts >= train_end) & (ts < validation_end), ts >= validation_end],
            ["TRAIN", "VALIDATION", "HOLDOUT"],
            default="INVALID",
        ),
        index=decision_time.index,
        dtype="object",
    )



def _derive_m0_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "zone_event_id", "r09_entry_time", "tp15_sl15_outcome", "tp15_sl15_same_bar_both_flag",
        "tp15_sl15_gross_return", "tp15_sl15_net_return_1x_cost", "tp15_sl15_net_return_2x_cost",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"R09 label table missing M0 fields: {missing}")
    out = labels.loc[:, sorted(required)].copy()
    out["decision_time"] = pd.to_datetime(out["r09_entry_time"], errors="coerce")
    risk = 0.0015
    long_gross = pd.to_numeric(out["tp15_sl15_gross_return"], errors="coerce").copy()
    long_net_1x = pd.to_numeric(out["tp15_sl15_net_return_1x_cost"], errors="coerce").copy()
    long_net_2x = pd.to_numeric(out["tp15_sl15_net_return_2x_cost"], errors="coerce").copy()
    inferred_cost_1x = long_gross - long_net_1x
    inferred_cost_2x = long_gross - long_net_2x
    outcome = out["tp15_sl15_outcome"].astype(str)
    same_bar = out["tp15_sl15_same_bar_both_flag"].fillna(False).astype(bool)
    # A 1m OHLC bar cannot reveal whether TP or SL occurred first.  Exclude the
    # ambiguous row from both directions rather than turning it into a negative
    # example or imposing a direction-specific ordering assumption.
    long_gross.loc[same_bar] = np.nan
    long_net_1x.loc[same_bar] = np.nan
    long_net_2x.loc[same_bar] = np.nan
    short_gross = np.select(
        [outcome.eq("TP"), outcome.eq("SL"), outcome.eq("TIME")],
        [-risk, risk, -long_gross],
        default=np.nan,
    ).astype(float)
    short_gross[same_bar.to_numpy()] = np.nan
    result = pd.DataFrame({
        "zone_event_id": out["zone_event_id"].astype(str),
        "decision_time": out["decision_time"],
        "long_gross_r": long_gross / risk,
        "long_net_1x_r": long_net_1x / risk,
        "long_net_2x_r": long_net_2x / risk,
        "long_target_before_stop": outcome.eq("TP") & ~same_bar,
        "long_path_outcome": outcome.where(~same_bar, pd.NA),
        "short_gross_r": short_gross / risk,
        "short_net_1x_r": (short_gross - inferred_cost_1x) / risk,
        "short_net_2x_r": (short_gross - inferred_cost_2x) / risk,
        "short_target_before_stop": outcome.eq("SL") & ~same_bar,
        "short_path_outcome": outcome.map({"TP": "STOP", "SL": "TARGET", "TIME": "TIME"}).where(~same_bar, pd.NA),
        "long_natural_stop_distance_bp": 15.0,
        "short_natural_stop_distance_bp": 15.0,
    })
    return result


def _pivot_r12_labels(labels: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "zone_event_id", "checkpoint_minutes", "checkpoint_available_time", "entry_time",
        "trade_direction", "natural_stop_distance_bp", "r2p0_outcome", "r2p0_target_before_stop",
        "r2p0_gross_r", "r2p0_net_1x_r", "r2p0_net_2x_r",
    ]
    missing = sorted(set(wanted) - set(labels.columns))
    if missing:
        raise RuntimeError(f"R12 label table missing fields: {missing}")
    source = labels.loc[:, wanted].copy()
    source["trade_direction"] = source["trade_direction"].astype(str).str.upper()
    if source.duplicated(["zone_event_id", "checkpoint_minutes", "trade_direction"]).any():
        raise RuntimeError("R12 labels contain duplicate direction rows")
    keys = ["zone_event_id", "checkpoint_minutes", "checkpoint_available_time", "entry_time"]
    parts: list[pd.DataFrame] = []
    for direction in ("LONG", "SHORT"):
        part = source.loc[source["trade_direction"].eq(direction)].drop(columns="trade_direction")
        rename = {
            name: f"{direction.lower()}_{name}"
            for name in part.columns
            if name not in keys
        }
        parts.append(part.rename(columns=rename))
    out = parts[0].merge(parts[1], on=keys, how="inner", validate="one_to_one")
    out = out.rename(columns={"checkpoint_available_time": "decision_time"})
    return out


def _assert_no_label_columns(feature_columns: Iterable[str]) -> None:
    leaked = [name for name in feature_columns if any(token in name.lower() for token in LABEL_TOKENS)]
    if leaked:
        raise RuntimeError(f"future/outcome columns leaked into feature contract: {leaked[:20]}")


def load_r13_data(
    r09_dir: str | Path,
    r12_dir: str | Path,
    config: PostSweepSupervisedConfig,
    *,
    max_events: int = 0,
) -> R13DataBundle:
    cfg = config.validate()
    r09_root = resolve_report_dir(
        r09_dir,
        alternatives=("09_structured_swing_stop_pool_hypotheses_r09", "structured_swing_stop_pool_hypotheses_r09"),
    )
    r12_root = resolve_report_dir(
        r12_dir,
        alternatives=("12_post_sweep_rejection_acceptance_r12",),
    )
    r09_features = _parse_times(_read(r09_root / R09_FEATURE_FILE))
    r09_labels = _parse_times(_read(r09_root / R09_LABEL_FILE))
    r12_features = _parse_times(_read(r12_root / R12_FEATURE_FILE))
    r12_labels = _parse_times(_read(r12_root / R12_LABEL_FILE))
    if "event_kind" in r09_features.columns:
        r09_features = r09_features.loc[r09_features["event_kind"].astype(str).eq("swing_zone_sweep")].copy()
    if "event_kind" in r09_labels.columns:
        r09_labels = r09_labels.loc[r09_labels["event_kind"].astype(str).eq("swing_zone_sweep")].copy()

    if r09_features["zone_event_id"].duplicated().any():
        raise RuntimeError("R09 zone feature table must be unique by zone_event_id")
    if max_events > 0:
        ordered = r09_features.sort_values(["event_available_time", "zone_event_id"], kind="mergesort").copy()
        ordered["_split"] = _split(ordered["event_available_time"], cfg)
        per_split = max(1, int(np.ceil(int(max_events) / 3)))
        sampled = []
        for split_name in ("TRAIN", "VALIDATION", "HOLDOUT"):
            group = ordered.loc[ordered["_split"].eq(split_name)]
            if group.empty:
                continue
            positions = np.linspace(0, len(group) - 1, min(per_split, len(group)), dtype=int)
            sampled.append(group.iloc[np.unique(positions)])
        chosen = pd.concat(sampled, ignore_index=True) if sampled else ordered.head(int(max_events))
        chosen = chosen.sort_values(["event_available_time", "zone_event_id"], kind="mergesort").head(int(max_events))
        wanted_ids = set(chosen["zone_event_id"].astype(str))
        r09_features = r09_features.loc[r09_features["zone_event_id"].astype(str).isin(wanted_ids)].copy()
        r09_labels = r09_labels.loc[r09_labels["zone_event_id"].astype(str).isin(wanted_ids)].copy()
        r12_features = r12_features.loc[r12_features["zone_event_id"].astype(str).isin(wanted_ids)].copy()
        r12_labels = r12_labels.loc[r12_labels["zone_event_id"].astype(str).isin(wanted_ids)].copy()

    release_cols = [name for name in R09_RELEASE_FEATURES if name in r09_labels.columns]
    release = r09_labels[["zone_event_id", *release_cols]].drop_duplicates("zone_event_id")
    release = release.rename(columns={"high_stop_release_label": "high_stop_release_feature"})
    r09_features = r09_features.rename(columns={"high_stop_release_label": "high_stop_release_feature"})
    r12_features = r12_features.rename(columns={"high_stop_release_label": "high_stop_release_feature"})
    r09_full = r09_features.merge(release, on="zone_event_id", how="left", validate="one_to_one", suffixes=("", "_release"))
    if "high_stop_release_feature_release" in r09_full.columns:
        r09_full["high_stop_release_feature"] = r09_full["high_stop_release_feature"].fillna(r09_full["high_stop_release_feature_release"])
        r09_full = r09_full.drop(columns="high_stop_release_feature_release")
    base_cols = [name for name in r09_full.columns if _safe_base_feature(name)]
    base_cols = [name for name in base_cols if name not in META_COLUMNS]
    _assert_no_label_columns(base_cols)

    datasets: dict[int, pd.DataFrame] = {}
    base_contract: dict[int, list[str]] = {}
    dynamic_contract: dict[int, list[str]] = {}
    audit_rows: list[dict[str, object]] = []

    # M0: sweep instant. The symmetric 15bp reference is kept separate from the
    # natural-stop 2R checkpoints and is never compared as if the labels were identical.
    if 0 in cfg.checkpoints_minutes:
        m0_labels = _derive_m0_labels(r09_labels)
        m0 = r09_full.merge(m0_labels, on="zone_event_id", how="inner", validate="one_to_one")
        m0["checkpoint_minutes"] = 0
        m0["checkpoint_id"] = m0["zone_event_id"].astype(str) + "__M0"
        m0["entry_time"] = m0["decision_time"]
        m0["split"] = _split(m0["decision_time"], cfg)
        m0["long_profitable_label"] = _resolved_structural_label(
            m0["long_net_1x_r"], m0["long_path_outcome"], cfg.profitable_net_r_threshold, target_token="TP",
        )
        m0["short_profitable_label"] = _resolved_structural_label(
            m0["short_net_1x_r"], m0["short_path_outcome"], cfg.profitable_net_r_threshold,
        )
        datasets[0] = m0.reset_index(drop=True)
        base_contract[0] = [name for name in base_cols if _release_feature_available(name, 0)]
        dynamic_contract[0] = []

    r12_label_wide = _pivot_r12_labels(r12_labels)
    r12_feature_cols_all = [name for name in r12_features.columns if name not in {"zone_event_id"}]
    dynamic_cols = [name for name in r12_feature_cols_all if name not in set(r09_full.columns) and _safe_dynamic_feature(name)]
    # Explicitly rename causal pre-entry path metrics so generic leakage checks do
    # not mistake them for future labels.
    dynamic_rename = {
        "pre_entry_mfe_long_bp": "visible_pre_entry_upside_bp",
        "pre_entry_mae_long_bp": "visible_pre_entry_downside_bp",
    }
    r12_features = r12_features.rename(columns=dynamic_rename)
    dynamic_cols = [dynamic_rename.get(name, name) for name in dynamic_cols]
    dynamic_cols = [name for name in dynamic_cols if name in r12_features.columns]
    _assert_no_label_columns([name for name in dynamic_cols if not name.startswith("visible_pre_entry_")])

    for minutes in cfg.checkpoints_minutes:
        if minutes == 0:
            continue
        features = r12_features.loc[pd.to_numeric(r12_features["checkpoint_minutes"], errors="coerce").eq(minutes)].copy()
        labels = r12_label_wide.loc[pd.to_numeric(r12_label_wide["checkpoint_minutes"], errors="coerce").eq(minutes)].copy()
        # R09 is the canonical static/event-time table.  R12 contributes only
        # checkpoint metadata and genuinely post-sweep dynamic fields, avoiding
        # inconsistent copies of the same base feature across M0/M3/M5/M10.
        dynamic_payload = [
            name for name in dynamic_cols
            if name in features.columns and _release_feature_available(name, int(minutes))
        ]
        feature_keys = ["zone_event_id", "checkpoint_minutes", "checkpoint_available_time", "entry_time"]
        feature_keys = [name for name in feature_keys if name in features.columns]
        checkpoint_features = features.loc[:, list(dict.fromkeys([*feature_keys, *dynamic_payload]))].copy()
        merged = r09_full.merge(checkpoint_features, on="zone_event_id", how="inner", validate="one_to_one")
        merged = merged.merge(
            labels,
            on=["zone_event_id", "checkpoint_minutes"],
            how="inner",
            suffixes=("", "_label"),
            validate="one_to_one",
        )
        if "decision_time_label" in merged.columns:
            mismatch = pd.to_datetime(merged["checkpoint_available_time"], errors="coerce") != pd.to_datetime(merged["decision_time_label"], errors="coerce")
            if mismatch.fillna(True).any():
                raise RuntimeError(f"R12 feature/label decision time mismatch at M{minutes}")
        merged["decision_time"] = pd.to_datetime(merged["checkpoint_available_time"], errors="coerce")
        merged["entry_time"] = pd.to_datetime(merged.get("entry_time_label", merged.get("entry_time")), errors="coerce")
        for direction in ("long", "short"):
            rename_map = {
                f"{direction}_r2p0_outcome": f"{direction}_path_outcome",
                f"{direction}_r2p0_gross_r": f"{direction}_gross_r",
                f"{direction}_r2p0_net_1x_r": f"{direction}_net_1x_r",
                f"{direction}_r2p0_net_2x_r": f"{direction}_net_2x_r",
                f"{direction}_r2p0_target_before_stop": f"{direction}_target_before_stop",
            }
            merged = merged.rename(columns={k: v for k, v in rename_map.items() if k in merged.columns})
        merged["checkpoint_id"] = merged["zone_event_id"].astype(str) + f"__M{minutes}"
        merged["split"] = _split(merged["decision_time"], cfg)
        merged["long_profitable_label"] = _resolved_structural_label(
            merged["long_net_1x_r"], merged["long_path_outcome"], cfg.profitable_net_r_threshold,
        )
        merged["short_profitable_label"] = _resolved_structural_label(
            merged["short_net_1x_r"], merged["short_path_outcome"], cfg.profitable_net_r_threshold,
        )
        datasets[int(minutes)] = merged.reset_index(drop=True)
        base_contract[int(minutes)] = [
            name for name in base_cols
            if name in merged.columns and _release_feature_available(name, int(minutes))
        ]
        dynamic_contract[int(minutes)] = [
            name for name in dynamic_cols
            if name in merged.columns and _release_feature_available(name, int(minutes))
        ]

    for minutes, frame in datasets.items():
        duplicate = int(frame["checkpoint_id"].duplicated().sum())
        split_counts = frame["split"].value_counts().to_dict()
        audit_rows.append({
            "checkpoint_minutes": minutes,
            "events": len(frame),
            "unique_sweeps": frame["zone_event_id"].nunique(),
            "duplicate_checkpoint_ids": duplicate,
            "train_events": int(split_counts.get("TRAIN", 0)),
            "validation_events": int(split_counts.get("VALIDATION", 0)),
            "holdout_events": int(split_counts.get("HOLDOUT", 0)),
            "base_feature_count": len(base_contract[minutes]),
            "dynamic_feature_count": len(dynamic_contract[minutes]),
            "release_availability_violations": int(sum(
                not _release_feature_available(name, int(minutes))
                for name in [*base_contract[minutes], *dynamic_contract[minutes]]
            )),
            "entry_at_decision_time_violations": int((pd.to_datetime(frame["entry_time"], errors="coerce") != pd.to_datetime(frame["decision_time"], errors="coerce")).fillna(True).sum()),
        })
    audit = pd.DataFrame(audit_rows)
    failures = audit.loc[
        audit["duplicate_checkpoint_ids"].gt(0)
        | audit["release_availability_violations"].gt(0)
        | audit["entry_at_decision_time_violations"].gt(0)
    ]
    if not failures.empty:
        raise RuntimeError(f"R13 source-data causal gate failed:\n{failures.to_string(index=False)}")
    return R13DataBundle(datasets=datasets, base_columns=base_contract, dynamic_columns=dynamic_contract, audit=audit)
