#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""02 rich trade-bar order-flow and absorption research for panic recovery.

This pass keeps the 01 multi-bar episode timing as the baseline, but replaces
simple OHLCV environment screening with trade-derived evidence:

- active buy/sell notional and trade counts;
- short-horizon normalized CVD / delta pressure;
- large-trade participation and large-flow reversal;
- transaction activity, average trade size and max-trade concentration;
- price impact per unit of selling and absorption near the episode low;
- repeated green signals whose order flow genuinely improves.

Important timing
----------------
- All stage nodes are created from closed bars.
- Entry remains next-bar open.
- Rolling baselines are shifted and use earlier bars only.
- A green-signal episode aggregate ends at the green signal bar.
- Start/orange filters use node-local fields only; they never use the eventual
  episode low or any later episode aggregate.
- Fixed-horizon outcomes and the final episode low are outputs, never inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.trade_bar_orderflow import (  # noqa: E402
    attach_orderflow_to_stage_events,
    build_trade_bar_orderflow_features,
    summarize_episode_orderflow,
    trade_bar_field_coverage,
    validate_trade_bar_orderflow,
)


SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def _load_v1_module():
    path = Path(__file__).with_name("01_environment_and_cluster_scale_in_research.py")
    spec = importlib.util.spec_from_file_location("panic_recovery_01_shared", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load 01 shared research helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


V1 = _load_v1_module()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="02 panic recovery rich trade-bar order-flow + absorption research",
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
        default="data/reports/research/liquidity/panic_selloff_rejection_recovery_long/02_trade_bar_orderflow_absorption",
    )

    # Same episode baseline as 01 so this pass isolates order-flow information.
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
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--min-filter-train", type=int, default=80)
    p.add_argument("--min-filter-holdout", type=int, default=35)
    p.add_argument("--top-atomic-for-pairs", type=int, default=10)
    p.add_argument("--top-scale-filters", type=int, default=4)

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--cost-multipliers", default="1.0,2.0")

    p.add_argument("--cluster-gap-bars", default="15,30,60")
    p.add_argument("--stop-buffer-pct", type=float, default=0.0005)
    p.add_argument("--target-r-list", default="0.75,1.0,1.5")
    p.add_argument("--save-trade-sample", type=int, default=30000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_list(text: str, *, cast: Callable[[str], Any], name: str) -> list[Any]:
    values: list[Any] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = cast(token)
        if float(value) <= 0:
            raise ValueError(f"{name} must contain positive values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_filter_columns(events: pd.DataFrame, specs: list[Any]) -> pd.DataFrame:
    """Attach only the explicitly supplied causal masks.

    01's helper also computes a green-specific recovery fraction, which is not
    valid for orange/start rows. 02 therefore keeps mask attachment generic.
    """
    out = events.copy()
    for spec in specs:
        mask = spec.predicate(out)
        out[f"filter__{spec.name}"] = mask.fillna(False).astype(bool)
    return out


def attach_repeat_green_features(signals: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy().sort_values("event_time").reset_index(drop=True)
    out["repeat_green_gap_bars"] = V1._timestamp_gap_in_bars(out["event_time"], bars.index)
    for col in (
        "signal_delta_ratio_2",
        "signal_large_delta_ratio_2",
        "signal_taker_buy_ratio_2",
        "flow_recovery_score",
        "entry_open",
    ):
        s = _num(out, col)
        out[f"prev_{col}"] = s.shift(1)
        out[f"change_vs_prev_{col}"] = s - s.shift(1)
    out["entry_return_vs_prev_green"] = _num(out, "entry_open") / _num(out, "prev_entry_open") - 1.0
    return out


def build_green_orderflow_filters(signals: pd.DataFrame) -> list[Any]:
    F = V1.FixedFilter
    specs = [
        F("panic_delta_le_n20", "panic_flow", "episode内主动卖方delta ratio <= -0.20",
          lambda d: _num(d, "panic_min_delta_ratio") <= -0.20),
        F("panic_delta_le_n35", "panic_flow", "episode内主动卖方delta ratio <= -0.35",
          lambda d: _num(d, "panic_min_delta_ratio") <= -0.35),
        F("panic_large_delta_le_n25", "large_flow", "episode内大单delta ratio <= -0.25",
          lambda d: _num(d, "panic_min_large_delta_ratio") <= -0.25),
        F("panic_large_share_ge_10", "large_flow", "恐慌阶段大单成交额占比 >= 10%",
          lambda d: _num(d, "panic_max_large_trade_share") >= 0.10),
        F("panic_large_share_ge_20", "large_flow", "恐慌阶段大单成交额占比 >= 20%",
          lambda d: _num(d, "panic_max_large_trade_share") >= 0.20),
        F("panic_notional_burst_ge_15", "activity", "恐慌成交额 >= 历史中位数1.5倍",
          lambda d: _num(d, "panic_max_notional_ratio") >= 1.50),
        F("panic_trades_burst_ge_15", "activity", "恐慌成交笔数 >= 历史中位数1.5倍",
          lambda d: _num(d, "panic_max_trades_ratio") >= 1.50),
        F("panic_max_trade_share_ge_08", "activity", "最大单笔占bar成交额 >= 8%",
          lambda d: _num(d, "panic_max_trade_share") >= 0.08),
        F("low_absorption_ge_15", "absorption", "真实低点吸收评分 >= 1.5",
          lambda d: _num(d, "low_absorption_score") >= 1.50),
        F("low_absorption_ge_25", "absorption", "真实低点吸收评分 >= 2.5",
          lambda d: _num(d, "low_absorption_score") >= 2.50),
        F("low_sell_burst_rejected", "absorption", "低点卖出成交额放大且收盘位于bar上半部",
          lambda d: (_num(d, "low_sell_notional_ratio") >= 1.50) & (_num(d, "low_close_pos") >= 0.55)),
        F("low_large_sell_rejected", "absorption", "低点大卖单占比高且价格拒绝下跌",
          lambda d: (_num(d, "low_large_sell_share") >= 0.15) & (_num(d, "low_close_pos") >= 0.55)),
        F("low_delta_divergence_ge_10", "divergence", "创新低时delta比前段最差值改善 >= 0.10",
          lambda d: _num(d, "low_delta_divergence") >= 0.10),
        F("low_large_divergence_ge_15", "divergence", "创新低时大单delta改善 >= 0.15",
          lambda d: _num(d, "low_large_delta_divergence") >= 0.15),
        F("delta_recovery_ge_20", "recovery_flow", "绿灯delta较恐慌最差值改善 >= 0.20",
          lambda d: _num(d, "delta_recovery_from_panic") >= 0.20),
        F("delta_recovery_ge_35", "recovery_flow", "绿灯delta较恐慌最差值改善 >= 0.35",
          lambda d: _num(d, "delta_recovery_from_panic") >= 0.35),
        F("large_delta_recovery_ge_25", "recovery_flow", "绿灯大单delta改善 >= 0.25",
          lambda d: _num(d, "large_delta_recovery_from_panic") >= 0.25),
        F("signal_delta_nonnegative", "recovery_flow", "绿灯2-bar delta ratio >= 0",
          lambda d: _num(d, "signal_delta_ratio_2") >= 0.0),
        F("signal_large_delta_nonnegative", "recovery_flow", "绿灯2-bar大单delta ratio >= 0",
          lambda d: _num(d, "signal_large_delta_ratio_2") >= 0.0),
        F("signal_taker_buy_ge_55", "recovery_flow", "绿灯2-bar主动买入占比 >= 55%",
          lambda d: _num(d, "signal_taker_buy_ratio_2") >= 0.55),
        F("sell_intensity_decay_le_70", "decay", "绿灯卖出强度降至恐慌峰值70%以下",
          lambda d: _num(d, "sell_intensity_decay") <= 0.70),
        F("flow_recovery_score_ge_50", "composite", "多维订单流恢复评分 >= 0.50",
          lambda d: _num(d, "flow_recovery_score") >= 0.50),
        F("flow_recovery_score_ge_80", "composite", "多维订单流恢复评分 >= 0.80",
          lambda d: _num(d, "flow_recovery_score") >= 0.80),
        F("repeat30_delta_improved", "cluster_flow", "30 bars内重复绿灯且delta继续改善 >= 0.08",
          lambda d: (_num(d, "repeat_green_gap_bars") <= 30) & (_num(d, "change_vs_prev_signal_delta_ratio_2") >= 0.08)),
        F("repeat30_large_delta_improved", "cluster_flow", "30 bars内重复绿灯且大单delta改善 >= 0.10",
          lambda d: (_num(d, "repeat_green_gap_bars") <= 30) & (_num(d, "change_vs_prev_signal_large_delta_ratio_2") >= 0.10)),
        F("repeat60_both_flow_improved", "cluster_flow", "60 bars内重复绿灯且普通/大单流同步改善",
          lambda d: (
              (_num(d, "repeat_green_gap_bars") <= 60)
              & (_num(d, "change_vs_prev_signal_delta_ratio_2") >= 0.05)
              & (_num(d, "change_vs_prev_signal_large_delta_ratio_2") >= 0.05)
          )),
        F("repeat30_lower_price_flow_improved", "cluster_flow", "30 bars内更低价格重复绿灯且订单流改善",
          lambda d: (
              (_num(d, "repeat_green_gap_bars") <= 30)
              & (_num(d, "entry_return_vs_prev_green") <= 0.0)
              & (_num(d, "change_vs_prev_signal_delta_ratio_2") >= 0.05)
          )),
    ]
    return specs


def build_start_orderflow_filters(starts: pd.DataFrame) -> list[Any]:
    """Orange-node rules use only fields visible on that orange closed bar."""
    F = V1.FixedFilter
    return [
        F("start_delta_le_n20", "start_flow", "橙灯bar delta ratio <= -0.20",
          lambda d: _num(d, "node_delta_ratio") <= -0.20),
        F("start_delta_le_n35", "start_flow", "橙灯bar delta ratio <= -0.35",
          lambda d: _num(d, "node_delta_ratio") <= -0.35),
        F("start_large_delta_le_n25", "start_large_flow", "橙灯bar大单delta ratio <= -0.25",
          lambda d: _num(d, "node_large_delta_ratio") <= -0.25),
        F("start_sell_burst_ge_15", "start_activity", "橙灯卖出成交额 >= 历史中位数1.5倍",
          lambda d: _num(d, "node_sell_notional_ratio_base") >= 1.50),
        F("start_trades_burst_ge_15", "start_activity", "橙灯成交笔数 >= 历史中位数1.5倍",
          lambda d: _num(d, "node_trades_ratio_base") >= 1.50),
        F("start_large_share_ge_10", "start_large_flow", "橙灯大单成交额占比 >= 10%",
          lambda d: _num(d, "node_large_trade_share") >= 0.10),
        F("start_max_trade_share_ge_08", "start_activity", "橙灯最大单笔占比 >= 8%",
          lambda d: _num(d, "node_max_trade_share") >= 0.08),
        F("start_absorption_ge_15", "start_absorption", "橙灯当下吸收评分 >= 1.5",
          lambda d: _num(d, "node_absorption_score") >= 1.50),
        F("start_absorption_ge_25", "start_absorption", "橙灯当下吸收评分 >= 2.5",
          lambda d: _num(d, "node_absorption_score") >= 2.50),
        F("start_sell_rejected", "start_absorption", "橙灯卖出放大但收盘在bar上半部",
          lambda d: (_num(d, "node_sell_notional_ratio_base") >= 1.50) & (_num(d, "close_pos") >= 0.55)),
        F("start_delta_reversal_short", "start_reversal", "橙灯短周期delta开始改善 >= 0.10",
          lambda d: _num(d, "node_delta_reversal_short") >= 0.10),
        F("start_large_reversal_short", "start_reversal", "橙灯短周期大单delta开始改善 >= 0.12",
          lambda d: _num(d, "node_large_delta_reversal_short") >= 0.12),
        F("start_taker_buy_ge_50", "start_reversal", "橙灯2-bar主动买入占比 >= 50%",
          lambda d: _num(d, "node_taker_buy_ratio_2") >= 0.50),
    ]


def evaluate_quantile_slices(
    events: pd.DataFrame,
    *,
    features: list[str],
    args: argparse.Namespace,
    stage_name: str,
) -> pd.DataFrame:
    """Learn quartile boundaries on train only and apply unchanged to holdout."""
    train_end = pd.Timestamp(args.train_end_date)
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    train = events[pd.to_datetime(events["event_time"]) <= train_end]
    rows: list[dict[str, Any]] = []

    for feature in features:
        train_values = _num(train, feature).dropna()
        if train_values.nunique() < 8:
            continue
        cuts = train_values.quantile([0.0, 0.25, 0.50, 0.75, 1.0]).to_numpy(dtype=float, copy=True)
        cuts[0] = -np.inf
        cuts[-1] = np.inf
        cuts = np.unique(cuts)
        if len(cuts) < 4:
            continue
        bucket = pd.cut(_num(events, feature), bins=cuts, include_lowest=True, duplicates="drop")
        for label, part in events.assign(_bucket=bucket).dropna(subset=["_bucket"]).groupby("_bucket", observed=True):
            tr = part[pd.to_datetime(part["event_time"]) <= train_end]
            ho = part[pd.to_datetime(part["event_time"]) > train_end]
            row: dict[str, Any] = {
                "stage": stage_name,
                "feature": feature,
                "train_derived_bucket": str(label),
                "lower_bound": float(label.left),
                "upper_bound": float(label.right),
            }
            for prefix, sample in (("all", part), ("train", tr), ("holdout", ho)):
                stats = V1._summary_row(sample, return_col)
                row.update({f"{prefix}_{k}": v for k, v in stats.items()})
            row["holdout_pass"] = bool(V1._holdout_pass(row, args))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["holdout_pass", "holdout_mean_net", "train_mean_net"],
        ascending=[False, False, False],
    ).reset_index(drop=True) if rows else pd.DataFrame()


def summarize_stage_orderflow(events: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "node_delta_ratio",
        "node_large_delta_ratio",
        "node_taker_buy_ratio",
        "node_sell_notional_ratio_base",
        "node_trades_ratio_base",
        "node_large_trade_share",
        "node_max_trade_share",
        "node_absorption_score",
        "node_delta_reversal_short",
        "node_large_delta_reversal_short",
    ]
    rows: list[dict[str, Any]] = []
    for stage, part in events.groupby("stage", sort=False):
        for field in fields:
            s = _num(part, field).dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "stage": stage,
                    "feature": field,
                    "count": int(len(s)),
                    "mean": float(s.mean()),
                    "q25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _top_pass(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if df.empty or "holdout_pass" not in df.columns:
        return pd.DataFrame()
    return df[df["holdout_pass"]].head(n)


def write_summary(
    out_dir: Path,
    coverage: pd.DataFrame,
    green_atomic: pd.DataFrame,
    green_pairs: pd.DataFrame,
    start_atomic: pd.DataFrame,
    green_quantiles: pd.DataFrame,
    cluster_compare: pd.DataFrame,
) -> None:
    lines = [
        "# 02 Trade-Bar Order-Flow / Absorption Research Summary",
        "",
        "本研究强制使用真实 trade bar 字段；如果订单流字段缺失或只有常数，会直接报错，不会退化成 OHLCV。",
        "",
        "## Trade-bar field coverage",
    ]
    usable = coverage[coverage["usable"]]
    lines.append(f"- usable rich fields: {len(usable)}/{len(coverage)}")
    bad = coverage[~coverage["usable"]]
    if not bad.empty:
        lines.append("- unusable/optional fields: " + ", ".join(bad["field"].astype(str).tolist()))

    lines.extend(["", "## Green order-flow filters with holdout pass"])
    passed = pd.concat([_top_pass(green_atomic), _top_pass(green_pairs)], ignore_index=True)
    if passed.empty:
        lines.append("- None. Rich order-flow did not produce a train/holdout-stable green filter.")
    else:
        for row in passed.head(10).itertuples(index=False):
            name = getattr(row, "filter_name", None) or getattr(row, "pair_name", None)
            lines.append(
                f"- {name}: train n={int(row.train_count)}, mean={row.train_mean_net:.4%}; "
                f"holdout n={int(row.holdout_count)}, mean={row.holdout_mean_net:.4%}, "
                f"PF={row.holdout_profit_factor:.3f}"
            )

    lines.extend(["", "## Orange/start absorption filters with holdout pass"])
    passed_start = _top_pass(start_atomic)
    if passed_start.empty:
        lines.append("- None. Orange-at-low visual cases were not identifiable with a stable fixed rule.")
    else:
        for row in passed_start.head(8).itertuples(index=False):
            lines.append(
                f"- {row.filter_name}: train n={int(row.train_count)}, mean={row.train_mean_net:.4%}; "
                f"holdout n={int(row.holdout_count)}, mean={row.holdout_mean_net:.4%}"
            )

    lines.extend(["", "## Train-derived quantile buckets with holdout pass"])
    qpass = green_quantiles[green_quantiles.get("holdout_pass", False)] if not green_quantiles.empty else pd.DataFrame()
    if qpass.empty:
        lines.append("- None.")
    else:
        for row in qpass.head(8).itertuples(index=False):
            lines.append(
                f"- {row.feature} {row.train_derived_bucket}: holdout n={int(row.holdout_count)}, "
                f"mean={row.holdout_mean_net:.4%}, PF={row.holdout_profit_factor:.3f}"
            )

    lines.extend(["", "## Order-flow filtered scale-in"])
    if cluster_compare.empty:
        lines.append("- No cluster variants were simulated.")
    elif "profit_factor_on_max" not in cluster_compare.columns:
        lines.append("- No completed cluster summary rows.")
    else:
        top = cluster_compare.sort_values("profit_factor_on_max", ascending=False).head(8)
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.candidate_name} / {row.scheme} / gap={int(row.cluster_gap_bars)} / "
                f"{row.target_name} / cost={row.cost_mult:.1f}x: "
                f"n={int(row.trades)}, mean={row.mean_net_on_max:.4%}, PF={row.profit_factor_on_max:.3f}"
            )

    lines.extend(
        [
            "",
            "## Causal rules",
            "- All entries are next-bar open.",
            "- Every rolling baseline is shifted to exclude the current bar.",
            "- Green episode features stop at the green signal bar.",
            "- Orange filters use only orange-node fields; final episode low is not an input.",
            "- Same-bar target/stop collision remains stop-first in the 01 shared simulator.",
            "- This is still research, not a production strategy declaration.",
        ]
    )
    (out_dir / "16_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_source != "trade_bar":
        raise ValueError("02 requires --data-source trade_bar")
    horizons = tuple(int(x) for x in _parse_list(args.horizons, cast=int, name="horizons"))
    if int(args.candidate_horizon) not in horizons:
        raise ValueError("candidate_horizon must be included in horizons")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = V1.load_bars(args)
    coverage = validate_trade_bar_orderflow(bars)
    print("[features] building rich causal trade-bar order-flow features", flush=True)
    orderflow = build_trade_bar_orderflow_features(
        bars,
        baseline_window=int(args.orderflow_baseline_window),
    )

    context = V1.build_context_features(bars)
    stage_events, detector_features = V1.build_stage_events(bars, context, args, horizons)
    if stage_events.empty:
        raise RuntimeError("No panic episode stages detected")
    stage_events, causal_audit = V1.attach_next_open_outcomes(stage_events, bars, args, horizons)

    episode_orderflow = summarize_episode_orderflow(
        stage_events,
        orderflow,
        progress_every=int(args.progress_every),
        progress_enabled=not bool(args.no_progress),
    )
    enriched = attach_orderflow_to_stage_events(stage_events, orderflow, episode_orderflow)
    stage_profile = summarize_stage_orderflow(enriched)

    signals = enriched[enriched["stage"] == "signal"].copy().sort_values("event_time").reset_index(drop=True)
    signals = attach_repeat_green_features(signals, bars)
    green_specs = build_green_orderflow_filters(signals)
    signals = add_filter_columns(signals, green_specs)
    green_atomic, green_pairs, scale_candidates = V1.evaluate_environment_filters(signals, green_specs, args)

    starts = enriched[enriched["stage"] == "start"].copy().sort_values("event_time").reset_index(drop=True)
    start_specs = build_start_orderflow_filters(starts)
    starts = add_filter_columns(starts, start_specs)
    start_atomic, start_pairs, _ = V1.evaluate_environment_filters(starts, start_specs, args)

    green_quantiles = evaluate_quantile_slices(
        signals,
        features=[
            "panic_min_delta_ratio",
            "panic_min_large_delta_ratio",
            "panic_max_large_trade_share",
            "panic_max_trades_ratio",
            "panic_max_notional_ratio",
            "panic_max_trade_share",
            "low_absorption_score",
            "low_delta_divergence",
            "low_large_delta_divergence",
            "delta_recovery_from_panic",
            "large_delta_recovery_from_panic",
            "signal_delta_ratio_2",
            "signal_large_delta_ratio_2",
            "signal_taker_buy_ratio_2",
            "sell_intensity_decay",
            "flow_recovery_score",
        ],
        args=args,
        stage_name="signal",
    )
    start_quantiles = evaluate_quantile_slices(
        starts,
        features=[
            "node_delta_ratio",
            "node_large_delta_ratio",
            "node_sell_notional_ratio_base",
            "node_trades_ratio_base",
            "node_large_trade_share",
            "node_max_trade_share",
            "node_absorption_score",
            "node_delta_reversal_short",
            "node_large_delta_reversal_short",
        ],
        args=args,
        stage_name="start",
    )

    trades, cluster_summary, cluster_yearly = V1.simulate_cluster_variants(
        bars,
        signals,
        scale_candidates,
        args,
    )
    cluster_compare = V1.build_cluster_comparison(cluster_summary)

    write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")
    write_csv(episode_orderflow, out_dir / "02_episode_orderflow_features.csv")
    write_csv(enriched, out_dir / "03_stage_events_enriched.csv")
    write_csv(stage_profile, out_dir / "04_stage_orderflow_profile.csv")
    write_csv(green_atomic, out_dir / "05_green_orderflow_atomic_train_holdout.csv")
    write_csv(green_pairs, out_dir / "06_green_orderflow_pairs_train_holdout.csv")
    write_csv(green_quantiles, out_dir / "07_green_orderflow_quantiles_train_holdout.csv")
    write_csv(start_atomic, out_dir / "08_start_absorption_atomic_train_holdout.csv")
    write_csv(start_pairs, out_dir / "09_start_absorption_pairs_train_holdout.csv")
    write_csv(start_quantiles, out_dir / "10_start_orderflow_quantiles_train_holdout.csv")
    write_csv(scale_candidates, out_dir / "11_scale_filter_candidates.csv")
    write_csv(cluster_summary, out_dir / "12_orderflow_cluster_scale_in_summary.csv")
    write_csv(cluster_compare, out_dir / "13_orderflow_cluster_scale_in_vs_single.csv")
    write_csv(cluster_yearly, out_dir / "14_orderflow_cluster_scale_in_yearly.csv")
    trade_out = trades
    if int(args.save_trade_sample) > 0 and len(trades) > int(args.save_trade_sample):
        trade_out = trades.sort_values(["candidate_name", "scheme", "entry_time"]).head(int(args.save_trade_sample))
    write_csv(trade_out, out_dir / "15_orderflow_cluster_trades_sample.csv")
    write_csv(causal_audit, out_dir / "17_causal_audit.csv")
    write_summary(
        out_dir,
        coverage,
        green_atomic,
        green_pairs,
        start_atomic,
        green_quantiles,
        cluster_compare,
    )

    meta = {
        "script": Path(__file__).name,
        "research_family": "liquidity/panic_selloff_rejection_recovery_long",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "bar_rows": int(len(bars)),
        "detector_feature_rows": int(len(detector_features)),
        "episode_count": int(enriched["episode_id"].nunique()),
        "green_signal_count": int(len(signals)),
        "orange_start_count": int(len(starts)),
        "usable_trade_bar_fields": coverage[coverage["usable"]]["field"].tolist(),
        "green_filter_count": int(len(green_specs)),
        "start_filter_count": int(len(start_specs)),
        "scale_candidates": scale_candidates.to_dict(orient="records"),
        "cost_convention": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "cost_multipliers": _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        },
        "causal_guards": [
            "trade-bar baselines use prior closed bars via shift(1)",
            "episode order-flow windows end at signal_time",
            "orange/start filters use node-local fields only",
            "all entries execute next-bar open",
            "future returns and final-low diagnostics never enter filter masks",
            "same-bar stop/target collision is stop-first",
            "total cluster exposure is capped at 100%",
        ],
        "params": vars(args),
    }
    write_json(out_dir / "00_manifest.json", meta)
    finalize_research_report(
        out_dir,
        title="02 Panic Recovery Trade-Bar Order-Flow and Absorption",
        print_log=True,
    )
    print(f"[done] reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
