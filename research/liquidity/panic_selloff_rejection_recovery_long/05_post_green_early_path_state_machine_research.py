#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""05 causal post-green early-path and multi-stage entry research.

Green is no longer treated as one universal entry time. Fixed, interpretable
state machines either confirm immediate continuation, wait for a controlled
pullback/reclaim, or reject the episode after structural failure / renewed sell
pressure. All decisions execute at the following bar open.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.trade_bar_orderflow import (  # noqa: E402
    attach_orderflow_to_stage_events,
    build_trade_bar_orderflow_features,
    summarize_episode_orderflow,
    validate_trade_bar_orderflow,
)
from research.liquidity.panic_selloff_rejection_recovery_long.common.orange_to_green_path import (  # noqa: E402
    attach_post_green_path_diagnostics,
    build_orange_to_green_path_features,
)
from research.liquidity.panic_selloff_rejection_recovery_long.common.post_green_entry_state import (  # noqa: E402
    ENTRY_MODELS,
    build_post_green_diagnostics_and_decisions,
    entry_model_dictionary,
    post_class_capture,
    summarize_funnel,
)

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


V1 = _load_sibling("01_environment_and_cluster_scale_in_research.py", "panic_recovery_01_shared_for_05")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="05 post-green early path and causal multi-stage entry state machines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--data-source", choices=["trade_bar"], default="trade_bar")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--train-end-date", default="2024-12-31 23:59:59")
    p.add_argument(
        "--out-dir",
        default="data/reports/research/liquidity/panic_selloff_rejection_recovery_long/05_post_green_early_path_state_machine",
    )

    # Detector parameters remain aligned with 01-04.
    p.add_argument("--baseline-window", type=int, default=60)
    p.add_argument("--selloff-window", type=int, default=5)
    p.add_argument("--min-red-bars", type=int, default=3)
    p.add_argument("--observe-drop-pct", type=float, default=0.0045)
    p.add_argument("--observe-drop-vol-mult", type=float, default=2.5)
    p.add_argument("--observe-volume-ratio", type=float, default=1.10)
    p.add_argument("--panic-drop-pct", type=float, default=0.0075)
    p.add_argument("--panic-volume-ratio", type=float, default=1.35)
    p.add_argument("--stabilization-bars", type=int, default=2)
    p.add_argument("--min-rebound-from-low-pct", type=float, default=0.0020)
    p.add_argument("--pressure-decay-ratio", type=float, default=0.68)
    p.add_argument("--reclaim-fraction", type=float, default=0.35)
    p.add_argument("--breakout-lookback", type=int, default=2)
    p.add_argument("--max-episode-bars", type=int, default=30)
    p.add_argument("--cooldown-bars", type=int, default=8)

    p.add_argument("--orderflow-baseline-window", type=int, default=240)
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--candidate-horizon", type=int, default=60)
    p.add_argument("--post-path-horizon", type=int, default=240)
    p.add_argument("--early-max-wait-bars", type=int, default=10)
    p.add_argument("--low-retest-tolerance-pct", type=float, default=0.0008)

    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0005)
    p.add_argument("--target-r-list", default="0.75,1.0,1.5")
    p.add_argument("--cost-multipliers", default="1.0,2.0")
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--save-trade-sample", type=int, default=30000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_list(text: str, *, cast: Callable[[str], Any], name: str) -> list[Any]:
    out: list[Any] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = cast(token)
            if float(value) <= 0:
                raise ValueError(f"{name} must contain positive values")
            out.append(value)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(out))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def build_train_frozen_path_gates(signals: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["path_orange_to_low_return", "path_sell_delta_ratio", "path_low_close_position"]
    missing = [c for c in required if c not in signals.columns]
    if missing:
        raise RuntimeError(f"05 path gates missing 04 fields: {missing}")
    train = signals[pd.to_datetime(signals["event_time"]) <= train_end]
    if len(train) < 100:
        raise RuntimeError("not enough train signals for frozen path gates")
    thresholds = {
        "deep_sell_q20": float(pd.to_numeric(train["path_orange_to_low_return"], errors="coerce").quantile(0.20)),
        "sell_delta_q80": float(pd.to_numeric(train["path_sell_delta_ratio"], errors="coerce").quantile(0.80)),
        "low_close_q20": float(pd.to_numeric(train["path_low_close_position"], errors="coerce").quantile(0.20)),
    }
    deep = pd.to_numeric(signals["path_orange_to_low_return"], errors="coerce") <= thresholds["deep_sell_q20"]
    non_extreme_delta = pd.to_numeric(signals["path_sell_delta_ratio"], errors="coerce") >= thresholds["sell_delta_q80"]
    low_close = pd.to_numeric(signals["path_low_close_position"], errors="coerce") <= thresholds["low_close_q20"]
    masks = pd.DataFrame(
        {
            "GATE_ALL": True,
            "GATE_DEEP_SELL": deep.fillna(False),
            "GATE_DEEP_SELL_NONEXTREME_DELTA": (deep & non_extreme_delta).fillna(False),
            "GATE_DEEP_SELL_LOW_CLOSE": (deep & low_close).fillna(False),
        },
        index=signals.index,
    )
    definitions = pd.DataFrame(
        [
            {"gate_name": "GATE_ALL", "description": "No orange-to-green path gate", "train_only_thresholds": "none"},
            {
                "gate_name": "GATE_DEEP_SELL",
                "description": "Orange-to-low decline is in train deepest 20%",
                "train_only_thresholds": f"path_orange_to_low_return <= {thresholds['deep_sell_q20']:.8f}",
            },
            {
                "gate_name": "GATE_DEEP_SELL_NONEXTREME_DELTA",
                "description": "Deep decline while aggregate sell-path delta is not in the most negative tail",
                "train_only_thresholds": (
                    f"path_orange_to_low_return <= {thresholds['deep_sell_q20']:.8f}; "
                    f"path_sell_delta_ratio >= {thresholds['sell_delta_q80']:.8f}"
                ),
            },
            {
                "gate_name": "GATE_DEEP_SELL_LOW_CLOSE",
                "description": "Deep decline and purple-low bar closes in train lowest close-position tail",
                "train_only_thresholds": (
                    f"path_orange_to_low_return <= {thresholds['deep_sell_q20']:.8f}; "
                    f"path_low_close_position <= {thresholds['low_close_q20']:.8f}"
                ),
            },
        ]
    )
    return definitions, masks


def attach_candidate_masks(
    decisions: pd.DataFrame,
    green_signals: pd.DataFrame,
    gate_masks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_by_episode = pd.concat(
        [green_signals[["episode_id"]].reset_index(drop=True), gate_masks.reset_index(drop=True)], axis=1
    ).drop_duplicates("episode_id")
    out = decisions.merge(gate_by_episode, on="episode_id", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for model in [m.name for m in ENTRY_MODELS]:
        for gate in gate_masks.columns:
            candidate_name = f"{model}__{gate}"
            col = f"filter__{candidate_name}"
            out[col] = (out["entry_model"] == model) & out[gate].fillna(False).astype(bool)
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "entry_model": model,
                    "gate_name": gate,
                    "source": "fixed_state_machine+train_frozen_path_gate",
                    "filter_expression": candidate_name,
                    "decision_rows": int(out[col].sum()),
                }
            )
    return out, pd.DataFrame(rows)


def _candidate_mask(decisions: pd.DataFrame, candidate_name: str) -> np.ndarray:
    col = f"filter__{candidate_name}"
    if col not in decisions.columns:
        raise KeyError(col)
    return decisions[col].fillna(False).to_numpy(dtype=bool)


def simulate_single_entry_candidates(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if decisions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    decisions = decisions.sort_values(["event_time", "entry_model"]).reset_index(drop=True)
    arrays = V1._build_signal_arrays(decisions, bars)
    valid = arrays["signal_pos"] >= 0
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    targets = V1.target_specs(args)
    costs = _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers")
    total = len(candidates) * len(targets) * len(costs)
    progress = ProgressReporter(
        "[simulate] post-green state entries",
        total,
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress),
    )
    done = 0
    parts: list[pd.DataFrame] = []
    reason_map = {1: "target", 2: "structural_stop", 3: "end_of_data"}
    for candidate in candidates.itertuples(index=False):
        eligible = _candidate_mask(decisions, str(candidate.candidate_name)) & valid
        for target in targets:
            target_mode = 0 if target.mode == "r" else 1
            for cost_mult in costs:
                result = V1._simulate_cluster_fast(
                    opens,
                    highs,
                    lows,
                    closes,
                    arrays["signal_pos"],
                    arrays["episode_low"],
                    arrays["reference"],
                    eligible.astype(np.int8),
                    np.asarray([1.0], dtype=float),
                    1,
                    0,
                    int(args.entry_delay_bars),
                    float(args.stop_buffer_pct),
                    int(target_mode),
                    float(target.value),
                    0,
                    float(args.entry_fee_rate) * float(cost_mult),
                    float(args.exit_fee_rate) * float(cost_mult),
                    float(args.entry_slippage_pct) * float(cost_mult),
                    float(args.exit_slippage_pct) * float(cost_mult),
                )
                if len(result[0]):
                    part = pd.DataFrame(
                        {
                            "entry_signal_idx": result[0],
                            "entry_bar_pos": result[1],
                            "exit_bar_pos": result[2],
                            "entry_count": result[3],
                            "filled_weight": result[4],
                            "avg_entry_raw": result[5],
                            "stop_price": result[6],
                            "target_price": result[7],
                            "net_return_on_max_capital": result[8],
                            "net_return_on_deployed_capital": result[9],
                            "hold_bars": result[10],
                            "exit_reason_code": result[11],
                        }
                    )
                    source = decisions.iloc[part["entry_signal_idx"].to_numpy(dtype=int)].reset_index(drop=True)
                    part["entry_time"] = bars.index[part["entry_bar_pos"].to_numpy(dtype=int)].to_numpy()
                    part["exit_time"] = bars.index[part["exit_bar_pos"].to_numpy(dtype=int)].to_numpy()
                    part["entry_episode_id"] = pd.to_numeric(source["episode_id"], errors="coerce").to_numpy()
                    part["green_time"] = pd.to_datetime(source["green_time"]).to_numpy()
                    part["decision_time"] = pd.to_datetime(source["decision_time"]).to_numpy()
                    part["decision_bar_offset"] = pd.to_numeric(source["decision_bar_offset"], errors="coerce").to_numpy()
                    part["entry_model"] = str(candidate.entry_model)
                    part["gate_name"] = str(candidate.gate_name)
                    part["candidate_name"] = str(candidate.candidate_name)
                    part["candidate_source"] = str(candidate.source)
                    part["scheme"] = "single_full"
                    part["max_entries"] = 1
                    part["add_only_below_avg"] = False
                    part["cluster_gap_bars"] = 0
                    part["target_name"] = target.name
                    part["target_mode"] = target.mode
                    part["target_value"] = float(target.value)
                    part["cost_mult"] = float(cost_mult)
                    part["exit_reason"] = part["exit_reason_code"].map(reason_map)
                    part["year"] = pd.to_datetime(part["entry_time"]).dt.year
                    part["split"] = np.where(
                        pd.to_datetime(part["green_time"]) <= pd.Timestamp(args.train_end_date),
                        "train",
                        "holdout",
                    )
                    parts.append(part)
                done += 1
                if done < total:
                    progress.update(done)
    progress.close()
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if trades.empty:
        return trades, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    base_cols = [
        "candidate_name", "candidate_source", "entry_model", "gate_name", "scheme",
        "max_entries", "add_only_below_avg", "cluster_gap_bars", "target_name",
        "target_mode", "target_value", "cost_mult",
    ]
    summary = V1.summarize_cluster_trades(trades, base_cols)
    yearly = V1.summarize_cluster_trades(trades, [*base_cols, "year"])
    split = V1.summarize_cluster_trades(trades, [*base_cols, "split"])
    return trades, summary, yearly, split


def build_candidate_validation(split_summary: pd.DataFrame) -> pd.DataFrame:
    if split_summary.empty:
        return pd.DataFrame()
    keys = [
        "candidate_name", "entry_model", "gate_name", "target_name", "target_mode",
        "target_value", "cost_mult",
    ]
    metrics = ["trades", "mean_net_on_max", "profit_factor_on_max", "max_drawdown_on_max", "win_rate_on_max"]
    wide = split_summary.pivot_table(index=keys, columns="split", values=metrics, aggfunc="first")
    wide.columns = [f"{split_name}_{metric}" for metric, split_name in wide.columns]
    out = wide.reset_index()
    for split_name in ("train", "holdout"):
        for metric in metrics:
            col = f"{split_name}_{metric}"
            if col not in out:
                out[col] = np.nan
    out["holdout_pass"] = (
        (pd.to_numeric(out["train_trades"], errors="coerce") >= 40)
        & (pd.to_numeric(out["holdout_trades"], errors="coerce") >= 30)
        & (pd.to_numeric(out["train_mean_net_on_max"], errors="coerce") > 0)
        & (pd.to_numeric(out["holdout_mean_net_on_max"], errors="coerce") > 0)
        & (pd.to_numeric(out["train_profit_factor_on_max"], errors="coerce") > 1.0)
        & (pd.to_numeric(out["holdout_profit_factor_on_max"], errors="coerce") > 1.0)
    )
    out["train_selection_score"] = (
        pd.to_numeric(out["train_mean_net_on_max"], errors="coerce")
        * np.sqrt(pd.to_numeric(out["train_trades"], errors="coerce").clip(lower=0))
        + 0.0005
        * (pd.to_numeric(out["train_profit_factor_on_max"], errors="coerce") - 1.0).clip(-1.0, 2.0)
        * np.sqrt(pd.to_numeric(out["train_trades"], errors="coerce").clip(lower=0))
    )
    return out.sort_values(
        ["holdout_pass", "train_selection_score", "holdout_mean_net_on_max"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_regime_diagnostics(
    diagnostics: pd.DataFrame,
    signals: pd.DataFrame,
    funnel_yearly: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = diagnostics.merge(
        signals[["episode_id", "post_outcome_class"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    merged["year"] = pd.to_datetime(merged["green_time"]).dt.year
    merged["split"] = np.where(
        pd.to_datetime(merged["green_time"]) <= pd.Timestamp(args.train_end_date), "train", "holdout"
    )
    metric_cols = [
        c for c in merged.columns
        if c.startswith("diag_post_") and c.endswith(("_close_r", "_mfe_r", "_mae_r", "_delta_ratio", "_large_delta_ratio", "_sell_intensity_mean"))
    ]
    rows: list[dict[str, Any]] = []
    for (year, split_name), part in merged.groupby(["year", "split"], sort=True):
        for metric in metric_cols:
            values = pd.to_numeric(part[metric], errors="coerce")
            rows.append(
                {
                    "year": int(year),
                    "split": split_name,
                    "metric": metric,
                    "count": int(values.notna().sum()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )
    classes: list[dict[str, Any]] = []
    for (year, outcome), part in merged.groupby(["year", "post_outcome_class"], dropna=False):
        denom = int((merged["year"] == year).sum())
        classes.append(
            {
                "year": int(year),
                "post_outcome_class": outcome,
                "count": len(part),
                "share": len(part) / max(1, denom),
            }
        )
    class_df = pd.DataFrame(classes)
    if not funnel_yearly.empty:
        class_df.attrs["funnel_yearly_rows"] = len(funnel_yearly)
    return pd.DataFrame(rows), class_df


def causal_audit(decisions: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    event_pos = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(decisions["decision_time"])))
    entry_pos = event_pos + int(args.entry_delay_bars)
    valid = (event_pos >= 0) & (entry_pos >= 0) & (entry_pos < len(bars))
    expected_entry = pd.Series(pd.NaT, index=decisions.index, dtype="datetime64[ns]")
    if valid.any():
        expected_entry.loc[valid] = bars.index[entry_pos[valid]].to_numpy()
    return pd.DataFrame(
        {
            "episode_id": decisions["episode_id"].to_numpy(),
            "entry_model": decisions["entry_model"].to_numpy(),
            "green_time": pd.to_datetime(decisions["green_time"]).to_numpy(),
            "decision_time": pd.to_datetime(decisions["decision_time"]).to_numpy(),
            "feature_window_end": pd.to_datetime(decisions["feature_window_end"]).to_numpy(),
            "expected_entry_time": expected_entry.to_numpy(),
            "decision_not_before_green": (
                pd.to_datetime(decisions["decision_time"]) >= pd.to_datetime(decisions["green_time"])
            ).to_numpy(),
            "feature_window_ends_at_decision": (
                pd.to_datetime(decisions["feature_window_end"]) == pd.to_datetime(decisions["decision_time"])
            ).to_numpy(),
            "entry_is_next_open": valid,
            "stop_known_at_decision": decisions["decision_stop_known"].fillna(False).astype(bool).to_numpy(),
        }
    )


def write_summary(
    out_dir: Path,
    funnel_summary: pd.DataFrame,
    validation: pd.DataFrame,
    yearly: pd.DataFrame,
    capture: pd.DataFrame,
) -> None:
    lines = [
        "# 05 Post-Green Early Path State Machine Summary",
        "",
        "绿灯不再统一入场；延迟模型只使用绿灯后已经关闭的早期路径，下一根 open 执行。",
        "",
        "## Decision funnel",
    ]
    entered = funnel_summary[funnel_summary["status"] == "entered"] if not funnel_summary.empty else pd.DataFrame()
    if entered.empty:
        lines.append("- No state-machine entries.")
    else:
        for row in entered.sort_values("share", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.entry_model}: entered={int(row.count)}, rate={row.share:.1%}, "
                f"median wait={row.median_decision_bar_offset:.1f} bars"
            )
    lines.extend(["", "## Train/holdout candidates"])
    passed = validation[validation["holdout_pass"] == True] if not validation.empty else pd.DataFrame()  # noqa: E712
    if passed.empty:
        lines.append("- None passed train and holdout. Do not promote a state machine from this run.")
    else:
        for row in passed.head(12).itertuples(index=False):
            lines.append(
                f"- {row.candidate_name}/{row.target_name}/cost={row.cost_mult:.1f}x: "
                f"train n={int(row.train_trades)}, mean={row.train_mean_net_on_max:.4%}, PF={row.train_profit_factor_on_max:.3f}; "
                f"holdout n={int(row.holdout_trades)}, mean={row.holdout_mean_net_on_max:.4%}, PF={row.holdout_profit_factor_on_max:.3f}"
            )
    lines.extend(["", "## Best full-period structural results"])
    if validation.empty:
        lines.append("- No completed trades.")
    else:
        top = validation[(validation["cost_mult"] == 1.0)].sort_values(
            ["holdout_pass", "holdout_mean_net_on_max"], ascending=[False, False]
        ).head(12)
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.candidate_name}/{row.target_name}: holdout n={int(row.holdout_trades) if np.isfinite(row.holdout_trades) else 0}, "
                f"mean={row.holdout_mean_net_on_max:.4%}, PF={row.holdout_profit_factor_on_max:.3f}"
            )
    lines.extend(["", "## Outcome-class capture (diagnostic only)"])
    if capture.empty:
        lines.append("- No post-class capture table.")
    else:
        part = capture[(capture["status"] == "entered") & capture["post_outcome_class"].notna()]
        for row in part.sort_values("capture_or_reject_rate", ascending=False).head(15).itertuples(index=False):
            lines.append(
                f"- {row.entry_model} captures {row.post_outcome_class}: {row.capture_or_reject_rate:.1%} ({int(row.count)})"
            )
    lines.extend(
        [
            "",
            "## Causal and risk limits",
            "- Green-next-open is retained only as a baseline.",
            "- Every delayed decision uses bars closed after green and enters on the next open.",
            "- A purple-stop break or renewed sell-pressure failure before entry cancels the setup.",
            "- Post outcome classes and future returns are diagnostics only; no state-machine condition reads them.",
            "- Orange-to-green gate thresholds are learned on 2023-2024 only and frozen for holdout.",
            "- Trades use purple-low structural stop and R/reference target; no ordinary time exit.",
            "- Same-bar stop/target collision is stop-first.",
        ]
    )
    (out_dir / "19_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_source != "trade_bar":
        raise ValueError("05 requires --data-source trade_bar")
    horizons = tuple(int(x) for x in _parse_list(args.horizons, cast=int, name="horizons"))
    if int(args.candidate_horizon) not in horizons:
        raise ValueError("candidate_horizon must be included in horizons")
    if int(args.early_max_wait_bars) < 10:
        raise ValueError("early_max_wait_bars must be >= 10 for the fixed state machines")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = V1.load_bars(args)
    coverage = validate_trade_bar_orderflow(bars)
    print("[features] building causal trade-bar order-flow features", flush=True)
    orderflow = build_trade_bar_orderflow_features(bars, baseline_window=int(args.orderflow_baseline_window))
    context = V1.build_context_features(bars)
    stage_events, detector_features = V1.build_stage_events(bars, context, args, horizons)
    if stage_events.empty:
        raise RuntimeError("No panic episode stages detected")
    stage_events, stage_causal_audit = V1.attach_next_open_outcomes(stage_events, bars, args, horizons)
    episode_orderflow = summarize_episode_orderflow(
        stage_events,
        orderflow,
        progress_every=int(args.progress_every),
        progress_enabled=not bool(args.no_progress),
    )
    enriched = attach_orderflow_to_stage_events(stage_events, orderflow, episode_orderflow)
    print("[path] rebuilding orange-to-green causal path", flush=True)
    path_features, path_meta = build_orange_to_green_path_features(
        bars,
        orderflow,
        enriched,
        low_retest_tolerance_pct=float(args.low_retest_tolerance_pct),
        progress_enabled=not bool(args.no_progress),
        progress_every=int(args.progress_every),
    )
    signals = enriched[enriched["stage"] == "signal"].copy().sort_values("event_time").reset_index(drop=True)
    signals = signals.merge(path_features, on="episode_id", how="inner", validate="one_to_one")
    if signals.empty:
        raise RuntimeError("No green signals with complete orange-to-green path")
    signals = attach_post_green_path_diagnostics(
        signals,
        bars,
        horizon=int(args.post_path_horizon),
        entry_delay_bars=int(args.entry_delay_bars),
        stop_buffer_pct=float(args.stop_buffer_pct),
        entry_fee_rate=float(args.entry_fee_rate),
        exit_fee_rate=float(args.exit_fee_rate),
        entry_slippage_pct=float(args.entry_slippage_pct),
        exit_slippage_pct=float(args.exit_slippage_pct),
    )

    diagnostics, decisions, funnel = build_post_green_diagnostics_and_decisions(
        bars,
        orderflow,
        signals,
        stop_buffer_pct=float(args.stop_buffer_pct),
        max_wait_bars=int(args.early_max_wait_bars),
        progress_enabled=not bool(args.no_progress),
        progress_every=int(args.progress_every),
    )
    if decisions.empty:
        raise RuntimeError("No post-green decisions generated")
    decision_pos = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(decisions["event_time"])))
    valid_next_open = (decision_pos >= 0) & (decision_pos + int(args.entry_delay_bars) < len(bars))
    dropped_eod = int((~valid_next_open).sum())
    if dropped_eod:
        print(f"[causal] dropped {dropped_eod} decisions without a following entry bar", flush=True)
        decisions = decisions.loc[valid_next_open].reset_index(drop=True)
    gate_defs, gate_masks = build_train_frozen_path_gates(signals, pd.Timestamp(args.train_end_date))
    decisions, candidates = attach_candidate_masks(decisions, signals, gate_masks)
    trades, structural, yearly, split_summary = simulate_single_entry_candidates(
        bars, decisions, candidates, args
    )
    validation = build_candidate_validation(split_summary)
    funnel_summary, funnel_yearly = summarize_funnel(funnel)
    capture = post_class_capture(funnel, signals)
    regime_metrics, regime_classes = build_regime_diagnostics(
        diagnostics, signals, funnel_yearly, args
    )
    audit = causal_audit(decisions, bars, args)

    required_audit = [
        "decision_not_before_green",
        "feature_window_ends_at_decision",
        "entry_is_next_open",
        "stop_known_at_decision",
    ]
    if not audit.empty and not audit[required_audit].all(axis=None):
        bad = audit[~audit[required_audit].all(axis=1)].head(10)
        raise AssertionError(f"05 causal audit failed:\n{bad.to_string(index=False)}")
    if any(str(c).startswith("post_") for c in decisions.filter(regex="^filter__").columns):
        raise AssertionError("future post_* field leaked into candidate masks")

    trade_sample = trades.copy()
    if int(args.save_trade_sample) > 0 and len(trade_sample) > int(args.save_trade_sample):
        trade_sample = trade_sample.sort_values(["candidate_name", "entry_time"]).head(int(args.save_trade_sample))

    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")
    _write_csv(entry_model_dictionary(), out_dir / "02_entry_model_dictionary.csv")
    _write_csv(gate_defs, out_dir / "03_train_frozen_path_gate_dictionary.csv")
    _write_csv(path_meta, out_dir / "04_orange_to_green_feature_dictionary.csv")
    _write_csv(diagnostics, out_dir / "05_post_green_fixed_window_diagnostics.csv")
    _write_csv(funnel, out_dir / "06_state_machine_decision_funnel.csv")
    _write_csv(funnel_summary, out_dir / "07_state_machine_funnel_summary.csv")
    _write_csv(funnel_yearly, out_dir / "08_state_machine_funnel_yearly.csv")
    _write_csv(capture, out_dir / "09_post_outcome_class_capture_diagnostic.csv")
    _write_csv(candidates, out_dir / "10_entry_candidates.csv")
    _write_csv(structural, out_dir / "11_structural_stop_summary.csv")
    _write_csv(split_summary, out_dir / "12_structural_stop_train_holdout.csv")
    _write_csv(validation, out_dir / "13_candidate_validation.csv")
    _write_csv(yearly, out_dir / "14_structural_stop_yearly.csv")
    _write_csv(regime_metrics, out_dir / "15_early_path_regime_metric_shift.csv")
    _write_csv(regime_classes, out_dir / "16_post_path_class_yearly.csv")
    _write_csv(trade_sample, out_dir / "17_structural_trades_sample.csv")
    _write_csv(audit, out_dir / "18_causal_audit.csv")
    _write_csv(stage_causal_audit, out_dir / "18b_stage_causal_audit.csv")
    write_summary(out_dir, funnel_summary, validation, yearly, capture)

    meta = {
        "script": Path(__file__).name,
        "research_family": "liquidity/panic_selloff_rejection_recovery_long",
        "research_question": "Can causal early post-green path states choose immediate entry, delayed reclaim entry, or rejection?",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "bar_rows": int(len(bars)),
        "detector_feature_rows": int(len(detector_features)),
        "green_signal_count": int(len(signals)),
        "decision_count": int(len(decisions)),
        "candidate_count": int(len(candidates)),
        "entry_models": [m.name for m in ENTRY_MODELS],
        "path_gates": gate_defs.to_dict(orient="records"),
        "cost_convention": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "cost_multipliers": _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        },
        "causal_guards": [
            "green baseline enters next open",
            "delayed decisions use only closed bars through decision_time",
            "feature_window_end equals decision_time",
            "all entries execute decision next-bar open",
            "purple stop is known before any decision",
            "purple-stop break before entry aborts setup",
            "renewed sell-pressure failure before entry aborts setup",
            "post outcome class and future returns are diagnostics only",
            "path gate thresholds use train only and are frozen in holdout",
            "same-bar stop/target collision is stop-first",
            "no ordinary time exit",
        ],
        "params": vars(args),
    }
    _write_json(out_dir / "00_manifest.json", meta)
    finalize_research_report(out_dir, title="05 Panic Recovery Post-Green Early Path State Machines", print_log=True)
    print(f"[done] reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
