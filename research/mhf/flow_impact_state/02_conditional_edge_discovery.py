#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX Flow-Impact State Strategy — Round 02 conditional edge discovery.

R02 answers one narrow question: among the broad active-flow pressure events
from R01, do any *broad, causal, high-sample* conditions produce positive
expectancy after normal OKX costs in discovery, validation, and untouched
holdout periods?

The scan is deliberately constrained:
- the event threshold is fixed by R01 frequency calibration, not by returns;
- thresholds are fitted on 2023-2024 discovery data only;
- validation (2025-01 through 2025-09) and holdout (2025-10 through 2026-06)
  never choose features, thresholds, branches, horizons, or pairs;
- cumulative tails are used instead of arbitrary parameter grids;
- at most two causal features may be combined;
- final candidates require >=1,000 events and 40-90 events/month;
- no TP/SL, maker-fill assumption, Books/Liquidity, 4H hard gate, or ML model.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.conditional_edge import (  # noqa: E402
    DEFAULT_FEATURE_POLARITIES,
    ConditionalEdgeConfig,
    add_base_uplift,
    assign_time_splits,
    build_pair_specs,
    build_tail_specs,
    candidate_time_stability,
    clock_phase_diagnostic,
    evaluate_base_universes,
    evaluate_specs,
    feature_monotonicity,
    final_qualification,
    fit_discovery_quantiles,
    freeze_discovery_candidates,
    freeze_pair_candidates,
    pivot_split_results,
    prepare_conditional_features,
)
from src.research_common.flow_impact import (  # noqa: E402
    FlowImpactConfig,
    assign_pressure_event_clusters,
    build_flow_impact_features,
    detect_pressure_events,
    flow_field_coverage,
)
from src.research_common.flow_impact_io import (  # noqa: E402
    inclusive_end,
    load_rich_trade_bars,
    timeframe_delta,
)
from src.research_common.flow_impact_outcomes import future_path_outcomes  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "02_conditional_edge_discovery"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MHF_FLOW_IMPACT_STATE_R02"
EDGE_ID = "ETH_MHF_FLOW_IMPACT_STATE"
TITLE = "OKX Flow-Impact Causal Conditional Edge Discovery"
DEFAULT_OUT_DIR = "data/reports/research/mhf/flow_impact_state/02_conditional_edge_discovery"
DEFAULT_WINDOWS = (1, 3, 5)
DEFAULT_HORIZON_MINUTES = (1, 2, 5, 15, 30)
DEFAULT_TAIL_QUANTILES = (0.50, 0.67, 0.75, 0.80, 0.90, 0.95)
DEFAULT_TOUCH_LEVELS = (15.0, 25.0, 50.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict time-split conditional edge discovery for OKX active-flow pressure events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--discovery-end", default="2024-12-31 23:59:59")
    parser.add_argument("--validation-end", default="2025-09-30 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--pressure-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--horizon-minutes", default=",".join(map(str, DEFAULT_HORIZON_MINUTES)))
    parser.add_argument("--tail-quantiles", default=",".join(map(str, DEFAULT_TAIL_QUANTILES)))
    parser.add_argument("--first-touch-bps", default=",".join(map(str, DEFAULT_TOUCH_LEVELS)))
    parser.add_argument("--baseline-bars", type=int, default=1440)
    parser.add_argument("--baseline-min-periods", type=int, default=720)
    parser.add_argument(
        "--min-pressure-z",
        type=float,
        default=2.0,
        help="Frozen from R01 frequency calibration; do not select this from R02 returns.",
    )
    parser.add_argument("--cooldown-multiplier", type=float, default=1.0)
    parser.add_argument("--release-pressure-z", type=float, default=0.5)
    parser.add_argument("--entry-fee-rate", type=float, default=0.00055)
    parser.add_argument("--exit-fee-rate", type=float, default=0.00055)
    parser.add_argument("--entry-slippage", type=float, default=0.00020)
    parser.add_argument("--exit-slippage", type=float, default=0.00020)
    parser.add_argument("--minimum-total-events", type=int, default=1000)
    parser.add_argument("--minimum-discovery-events", type=int, default=500)
    parser.add_argument("--minimum-validation-events", type=int, default=200)
    parser.add_argument("--minimum-holdout-events", type=int, default=200)
    parser.add_argument("--minimum-year-events", type=int, default=80)
    parser.add_argument("--minimum-full-pf", type=float, default=1.20)
    parser.add_argument("--minimum-positive-month-ratio", type=float, default=0.65)
    parser.add_argument("--minimum-active-date-ratio", type=float, default=0.65)
    parser.add_argument("--target-monthly-events-low", type=float, default=40.0)
    parser.add_argument("--target-monthly-events-high", type=float, default=90.0)
    parser.add_argument("--maximum-top5-winner-share", type=float, default=0.20)
    parser.add_argument("--discovery-fdr-alpha", type=float, default=0.10)
    parser.add_argument("--max-pair-features", type=int, default=4)
    parser.add_argument("--event-sample-rows", type=int, default=20_000)
    parser.add_argument("--write-full-events", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted(set(int(part.strip()) for part in str(value).split(",") if part.strip())))
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError(f"expected positive integer CSV, got: {value!r}")
    return parsed


def _parse_float_csv(value: str) -> tuple[float, ...]:
    parsed = tuple(sorted(set(float(part.strip()) for part in str(value).split(",") if part.strip())))
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError(f"expected positive float CSV, got: {value!r}")
    return parsed


def _conditional_config(args: argparse.Namespace) -> ConditionalEdgeConfig:
    cfg = ConditionalEdgeConfig(
        discovery_end=str(args.discovery_end),
        validation_end=str(args.validation_end),
        minimum_total_events=int(args.minimum_total_events),
        minimum_discovery_events=int(args.minimum_discovery_events),
        minimum_validation_events=int(args.minimum_validation_events),
        minimum_holdout_events=int(args.minimum_holdout_events),
        minimum_year_events=int(args.minimum_year_events),
        minimum_full_profit_factor=float(args.minimum_full_pf),
        minimum_positive_month_ratio=float(args.minimum_positive_month_ratio),
        minimum_active_date_ratio=float(args.minimum_active_date_ratio),
        target_monthly_events_low=float(args.target_monthly_events_low),
        target_monthly_events_high=float(args.target_monthly_events_high),
        maximum_top5_winner_share=float(args.maximum_top5_winner_share),
        discovery_fdr_alpha=float(args.discovery_fdr_alpha),
        max_pair_features=int(args.max_pair_features),
    )
    cfg.validate()
    return cfg


def _build_event_universe(
    bars: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[int, ...], tuple[int, ...], float, float]:
    windows = _parse_int_csv(args.pressure_windows)
    horizon_minutes = _parse_int_csv(args.horizon_minutes)
    touch_levels = _parse_float_csv(args.first_touch_bps)
    bar_delta = timeframe_delta(args.timeframe)
    bar_minutes = bar_delta.total_seconds() / 60.0
    horizons_bars = tuple(int(round(minutes / bar_minutes)) for minutes in horizon_minutes)
    if any(
        abs(horizon * bar_minutes - minutes) > 1e-9
        for horizon, minutes in zip(horizons_bars, horizon_minutes, strict=True)
    ):
        raise ValueError("Every horizon minute must be an exact multiple of the selected timeframe")

    feature_cfg = FlowImpactConfig(
        pressure_windows=windows,
        baseline_bars=int(args.baseline_bars),
        baseline_min_periods=int(args.baseline_min_periods),
        min_pressure_z=float(args.min_pressure_z),
        event_cooldown_multiplier=float(args.cooldown_multiplier),
    )
    feature_cfg.validate()
    coverage = flow_field_coverage(bars)
    print(f"[features] rows={len(bars):,} windows={windows}", flush=True)
    features = build_flow_impact_features(bars, feature_cfg)
    print(f"[events] pressure onset detection threshold_z={float(args.min_pressure_z):.3f}", flush=True)
    events = detect_pressure_events(
        features,
        windows=windows,
        min_pressure_z=float(args.min_pressure_z),
        cooldown_multiplier=float(args.cooldown_multiplier),
    )
    if events.empty:
        raise RuntimeError("No pressure events detected")
    start = pd.Timestamp(args.start_date)
    end = inclusive_end(args.end_date, bar_delta)
    events = events.loc[
        (pd.to_datetime(events["signal_time"]) >= start)
        & (pd.to_datetime(events["signal_time"]) <= end)
    ].copy()
    if events.empty:
        raise RuntimeError("No pressure events inside research window")
    events = assign_pressure_event_clusters(events, cluster_gap_bars=max(windows))

    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    events, _, audit = future_path_outcomes(
        bars,
        features,
        events,
        horizons_bars=horizons_bars,
        touch_levels_bps=touch_levels,
        normal_cost=normal_cost,
        fee_only_cost=fee_only_cost,
        release_pressure_z=float(args.release_pressure_z),
        bar_delta=bar_delta,
        progress_enabled=not args.no_progress,
    )
    valid = events.loc[~audit["causal_or_data_fail_flag"].to_numpy(dtype=bool)].copy()
    if valid.empty:
        raise RuntimeError("All events failed causal/data-path audit")
    valid = prepare_conditional_features(valid)
    valid = assign_time_splits(valid, _conditional_config(args))
    print(
        "[universe] "
        + " ".join(
            f"{name}={count:,}"
            for name, count in valid["research_split"].value_counts().sort_index().items()
        ),
        flush=True,
    )
    return valid, audit, coverage, windows, horizons_bars, fee_only_cost, normal_cost


def _feature_coverage(events: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(events.get(feature), errors="coerce")
        rows.append(
            {
                "feature": feature,
                "non_null_events": int(values.notna().sum()),
                "non_null_ratio": float(values.notna().mean()),
                "unique_values": int(values.nunique(dropna=True)),
                "p01": float(values.quantile(0.01)) if values.notna().any() else np.nan,
                "p50": float(values.quantile(0.50)) if values.notna().any() else np.nan,
                "p99": float(values.quantile(0.99)) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _split_definition(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, part in events.groupby("research_split", observed=False):
        rows.append(
            {
                "research_split": str(split),
                "start": pd.to_datetime(part["signal_bar_start"]).min(),
                "end": pd.to_datetime(part["signal_bar_start"]).max(),
                "window_events": int(len(part)),
                "unique_clusters": int(part["event_cluster_id"].nunique()),
                "months": int(part["month"].nunique()),
                "years": int(part["year"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("start")


def _candidate_specs(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "spec_id",
        "spec_type",
        "pressure_window_bars",
        "branch",
        "horizon_bars",
        "feature_1",
        "polarity_1",
        "selectivity_q_1",
        "threshold_1",
        "feature_2",
        "polarity_2",
        "selectivity_q_2",
        "threshold_2",
    ]
    return frame[[column for column in columns if column in frame.columns]].drop_duplicates("spec_id")


def _event_sample(events: pd.DataFrame, frozen_specs: pd.DataFrame, limit: int) -> pd.DataFrame:
    base_columns = [
        "event_id",
        "event_cluster_id",
        "signal_bar_start",
        "entry_time",
        "side_name",
        "pressure_window_bars",
        "research_split",
        *DEFAULT_FEATURE_POLARITIES.keys(),
    ]
    base_columns = [column for column in base_columns if column in events.columns]
    if frozen_specs.empty:
        return events[base_columns].head(int(limit)).copy()
    selected_ids = set(frozen_specs["spec_id"].head(20))
    parts: list[pd.DataFrame] = []
    for spec in frozen_specs.loc[frozen_specs["spec_id"].isin(selected_ids)].to_dict(orient="records"):
        part = events.loc[events["pressure_window_bars"].eq(int(spec["pressure_window_bars"]))].copy()
        mask = pd.to_numeric(part[spec["feature_1"]], errors="coerce")
        mask = mask.ge(float(spec["threshold_1"])) if spec["polarity_1"] == "high" else mask.le(float(spec["threshold_1"]))
        if str(spec.get("feature_2", "") or ""):
            second = pd.to_numeric(part[spec["feature_2"]], errors="coerce")
            second = second.ge(float(spec["threshold_2"])) if spec["polarity_2"] == "high" else second.le(float(spec["threshold_2"]))
            mask &= second
        chosen = part.loc[mask.fillna(False), base_columns].copy()
        chosen.insert(0, "spec_id", str(spec["spec_id"]))
        parts.append(chosen.head(max(1, int(limit) // max(1, len(selected_ids)))))
    if not parts:
        return events[base_columns].head(int(limit)).copy()
    return pd.concat(parts, ignore_index=True).head(int(limit))


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        ("pressure_z", "abnormal signed-notional pressure magnitude", "high pressure"),
        ("flow_ratio_aligned", "event-direction signed notional / total notional", "directional participation"),
        ("trade_imbalance_aligned", "event-direction aggressive trade-count imbalance", "trade-count participation"),
        ("large_flow_ratio_aligned", "event-direction imbalance inside large trades", "large-flow direction"),
        ("large_notional_share", "large-trade notional / total notional", "large-flow participation"),
        ("large_trade_share", "large-trade count / total trade count", "large-trade frequency"),
        ("flow_concentration", "abs net signed flow / sum abs per-bar signed flow", "directional consistency"),
        ("flow_persistence", "share of bars agreeing with event pressure direction", "directional persistence"),
        ("notional_ratio", "current-window notional / prior historical median", "relative volume"),
        ("avg_trade_notional_ratio", "average trade notional / prior historical median", "trade-size expansion"),
        ("max_trade_notional_ratio", "maximum trade notional / prior historical median", "largest-print expansion"),
        ("activity_z", "log current-window notional z-score using prior history", "relative activity"),
        ("price_response_norm", "direction-adjusted window return / prior volatility", "price response"),
        ("pressure_effectiveness", "normalized price response / pressure magnitude", "impact efficiency"),
        ("impact_bps_per_million", "direction-adjusted bps per one million signed notional", "raw impact efficiency"),
        ("direction_close_location", "close location toward pressure-side bar extreme", "rejection versus acceptance"),
    ]
    return pd.DataFrame(rows, columns=["feature", "definition", "mechanism"])


def _research_design_markdown(args: argparse.Namespace, config: ConditionalEdgeConfig) -> str:
    return f"""# R02 Research Design

## Frozen question

Can broad causal conditions inside OKX active-flow pressure events produce positive expectancy after normal costs while retaining at least {config.minimum_total_events:,} events?

## External-research constraints

1. Cont, Kukanov & Stoikov, *The Price Impact of Order Book Events* (arXiv:1011.6402): order-flow imbalance is more robust than trade volume alone, and impact depends on depth.
2. Donier & Bonart, *A Million Metaorder Analysis of Market Impact on the Bitcoin* (arXiv:1412.4503): crypto market impact is nonlinear and part of uninformed impact decays.
3. Albers et al., *To Make, or to Take, That Is the Question* (arXiv:2502.18625): taker strategies must clear fees; maker assumptions introduce fill uncertainty and adverse selection.
4. Jeon, *When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions in Crypto Futures* (arXiv:2607.09230): order-flow value is state- and asset-dependent and can be an additive overlay to L2 rather than a replacement.
5. Kim & Hansen, *The Quarter-Hour Effect* (arXiv:2607.09426): crypto volume/order-flow bursts are clock-phase dependent. R02 reports clock phase only as a diagnostic and forbids it from candidate selection.

## Frozen time split

- Discovery ends: `{args.discovery_end}`
- Validation ends: `{args.validation_end}`
- Holdout: after validation end through `{args.end_date}`

All quantile thresholds, features, polarities, branches, horizons and feature pairs are selected using discovery rows only.

## Search boundary

- Event threshold: pressure z >= {float(args.min_pressure_z):.2f}, frozen from R01 frequency calibration.
- Single-feature cumulative tails first.
- Pairwise search only among discovery-frozen single features.
- Maximum two causal features.
- No TP/SL, ML classifier, Books/Liquidity, session hard gate, 4H hard gate, or maker-fill assumption.
- Normal cost includes 0.11% round-trip fees plus conservative slippage.
"""


def _build_brief(
    events: pd.DataFrame,
    frozen_singles: pd.DataFrame,
    frozen_pairs: pd.DataFrame,
    qualified: pd.DataFrame,
    final_rank: pd.DataFrame,
    config: ConditionalEdgeConfig,
    normal_cost: float,
) -> str:
    split_counts = events["research_split"].value_counts().to_dict()
    lines = [
        "# Round 02 Research Brief",
        "",
        "## Decision",
        "",
    ]
    has_frozen_single = bool(
        not frozen_singles.empty
        and "frozen_discovery_flag" in frozen_singles.columns
        and frozen_singles["frozen_discovery_flag"].astype(bool).any()
    )
    if not qualified.empty and qualified["qualified_edge_flag"].astype(bool).any():
        decision = "research_continue"
        lines.append(
            "At least one frozen condition passed the >=1,000-event, all-split positive-expectancy and frequency gates. It remains a research candidate, not yet a tradable strategy, because trigger timing and exits have not been designed."
        )
    elif has_frozen_single:
        decision = "rejected_conditional_cells"
        lines.append(
            "Discovery found broad conditional structure, but no frozen condition survived validation and holdout with the required sample size and costs. Do not stack more 1m filters on these cells."
        )
    else:
        decision = "rejected_1m_conditional_search"
        lines.append(
            "No broad single-feature condition cleared the discovery freeze rules. The aggregated 1m pressure event is too weak for further environment-filter mining; the next justified test is a micro-timing study using raw trades/5s impact decay, not another filter version."
        )
    lines.extend(
        [
            "",
            f"Primary status: **{decision}**",
            "",
            "## Sample",
            "",
            f"- Discovery events: **{int(split_counts.get('discovery', 0)):,}**",
            f"- Validation events: **{int(split_counts.get('validation', 0)):,}**",
            f"- Holdout events: **{int(split_counts.get('holdout', 0)):,}**",
            f"- Normal round-trip cost: **{normal_cost:.3%}**",
            f"- Frozen single-feature conditions: **{int(frozen_singles.get('frozen_discovery_flag', pd.Series(dtype=bool)).sum()) if not frozen_singles.empty else 0}**",
            f"- Frozen pairwise conditions: **{int(frozen_pairs.get('frozen_discovery_flag', pd.Series(dtype=bool)).sum()) if not frozen_pairs.empty else 0}**",
            f"- Fully qualified conditions: **{int(qualified.get('qualified_edge_flag', pd.Series(dtype=bool)).sum()) if not qualified.empty else 0}**",
            "",
            "## Hard qualification",
            "",
            f"- Full events >= {config.minimum_total_events:,}",
            f"- Discovery/validation/holdout events >= {config.minimum_discovery_events:,}/{config.minimum_validation_events:,}/{config.minimum_holdout_events:,}",
            f"- Positive net expectancy in all three splits",
            f"- Full net PF >= {config.minimum_full_profit_factor:.2f}",
            f"- Positive months >= {config.minimum_positive_month_ratio:.0%}",
            f"- Events/month within {config.target_monthly_events_low:.0f}-{config.target_monthly_events_high:.0f}",
            "",
            "## Best final rows",
            "",
        ]
    )
    if final_rank.empty:
        lines.append("No frozen rows were available for final ranking.")
    else:
        for row in final_rank.head(10).itertuples(index=False):
            second = f" + {row.feature_2} {row.polarity_2}" if str(row.feature_2) else ""
            lines.append(
                f"- `{row.spec_id}` w{int(row.pressure_window_bars)} {row.branch} h{int(row.horizon_bars)}: "
                f"{row.feature_1} {row.polarity_1}{second}; events={int(row.full_events):,}, "
                f"net={float(row.full_net_mean):.4%}, PF={float(row.full_net_profit_factor):.3f}, "
                f"validation={float(row.validation_net_mean):.4%}, holdout={float(row.holdout_net_mean):.4%}, "
                f"passed_gates={int(row.passed_gate_count)}/6."
            )
    lines.extend(
        [
            "",
            "## Stop rule",
            "",
            "If no condition qualifies, do not create R03 by adding more 1m environment filters. Move directly to raw-trade/5s pressure-efficiency decay and use Books only later as an incremental overlay on its available history.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_conditional_discovery(
    events: pd.DataFrame,
    audit: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    horizons_bars: tuple[int, ...],
    fee_only_cost: float,
    normal_cost: float,
    args: argparse.Namespace,
) -> dict[str, Path]:
    config = _conditional_config(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tail_quantiles = _parse_float_csv(args.tail_quantiles)
    if any(not 0.5 <= q < 1.0 for q in tail_quantiles):
        raise ValueError("tail quantiles must be in [0.5, 1.0)")

    feature_polarities = {
        feature: polarities
        for feature, polarities in DEFAULT_FEATURE_POLARITIES.items()
        if feature in events.columns
        and pd.to_numeric(events[feature], errors="coerce").notna().sum() >= 100
        and pd.to_numeric(events[feature], errors="coerce").nunique(dropna=True) >= 10
    }
    features = tuple(feature_polarities)
    feature_coverage = _feature_coverage(events, features)
    split_definition = _split_definition(events)

    print(f"[conditional] features={len(features)} tail_quantiles={tail_quantiles}", flush=True)
    thresholds = fit_discovery_quantiles(
        events,
        features=features,
        tail_quantiles=tail_quantiles,
    )
    base_long = evaluate_base_universes(events, horizons=horizons_bars)
    base_wide = pivot_split_results(base_long)

    univariate_specs = build_tail_specs(
        thresholds,
        feature_polarities=feature_polarities,
        tail_quantiles=tail_quantiles,
        horizons=horizons_bars,
    )
    # The broad scan touches discovery only. Validation and holdout are read
    # only after a specification is frozen.
    univariate_discovery_long = evaluate_specs(
        events,
        univariate_specs,
        splits=("discovery",),
        progress_enabled=not args.no_progress,
        progress_label="[conditional] discovery univariate tails",
    )
    univariate_wide = add_base_uplift(pivot_split_results(univariate_discovery_long), base_wide)
    monotonicity = feature_monotonicity(univariate_wide)
    frozen_single_discovery = freeze_discovery_candidates(univariate_wide, monotonicity, config)
    eligible_single_specs = _candidate_specs(
        frozen_single_discovery.loc[frozen_single_discovery["frozen_discovery_flag"].astype(bool)]
    )
    if eligible_single_specs.empty:
        eligible_single = pd.DataFrame()
    else:
        single_all_long = evaluate_specs(
            events,
            eligible_single_specs,
            progress_enabled=not args.no_progress,
            progress_label="[conditional] frozen singles OOS",
        )
        eligible_single = add_base_uplift(pivot_split_results(single_all_long), base_wide)
        frozen_meta = frozen_single_discovery[["spec_id", "discovery_spearman", "discovery_fdr_q"]].drop_duplicates("spec_id")
        eligible_single = eligible_single.merge(frozen_meta, on="spec_id", how="left", validate="one_to_one")
        eligible_single["frozen_discovery_flag"] = True

    pair_specs = build_pair_specs(frozen_single_discovery, config=config)
    if pair_specs.empty:
        frozen_pair_discovery = pd.DataFrame()
        eligible_pair = pd.DataFrame()
    else:
        pair_discovery_long = evaluate_specs(
            events,
            pair_specs,
            splits=("discovery",),
            progress_enabled=not args.no_progress,
            progress_label="[conditional] discovery frozen pairs",
        )
        pair_discovery_wide = add_base_uplift(pivot_split_results(pair_discovery_long), base_wide)
        frozen_pair_discovery = freeze_pair_candidates(pair_discovery_wide, config)
        eligible_pair_specs = _candidate_specs(
            frozen_pair_discovery.loc[frozen_pair_discovery["frozen_discovery_flag"].astype(bool)]
        )
        if eligible_pair_specs.empty:
            eligible_pair = pd.DataFrame()
        else:
            pair_all_long = evaluate_specs(
                events,
                eligible_pair_specs,
                progress_enabled=not args.no_progress,
                progress_label="[conditional] frozen pairs OOS",
            )
            eligible_pair = add_base_uplift(pivot_split_results(pair_all_long), base_wide)
            pair_meta = frozen_pair_discovery[["spec_id", "discovery_fdr_q"]].drop_duplicates("spec_id")
            eligible_pair = eligible_pair.merge(pair_meta, on="spec_id", how="left", validate="one_to_one")
            eligible_pair["frozen_discovery_flag"] = True

    frozen_all = pd.concat([eligible_single, eligible_pair], ignore_index=True, sort=False)
    final_rank = final_qualification(frozen_all, config) if not frozen_all.empty else pd.DataFrame()
    qualified = (
        final_rank.loc[final_rank["qualified_edge_flag"].astype(bool)].copy()
        if not final_rank.empty
        else pd.DataFrame()
    )

    stability_specs = _candidate_specs(final_rank.head(50)) if not final_rank.empty else pd.DataFrame()
    yearly_stability, monthly_stability = candidate_time_stability(events, stability_specs)
    clock_phase = clock_phase_diagnostic(events, horizons=horizons_bars)
    sample = _event_sample(events, stability_specs, int(args.event_sample_rows))
    feature_dictionary = _feature_dictionary()
    design = _research_design_markdown(args, config)
    brief = _build_brief(
        events,
        frozen_single_discovery,
        frozen_pair_discovery,
        qualified,
        final_rank,
        config,
        normal_cost,
    )

    artifact_frames = [
        (coverage, out_dir / "01_input_field_coverage.csv"),
        (feature_coverage, out_dir / "02_condition_feature_coverage.csv"),
        (split_definition, out_dir / "03_time_split_definition.csv"),
        (base_wide, out_dir / "04_base_universe_summary.csv"),
        (thresholds, out_dir / "05_discovery_quantile_thresholds.csv"),
        (univariate_wide, out_dir / "06_univariate_tail_scan.csv"),
        (monotonicity, out_dir / "07_feature_monotonicity.csv"),
        (frozen_single_discovery, out_dir / "08_frozen_single_candidates.csv"),
        (pair_specs, out_dir / "09_pair_specs.csv"),
        (frozen_pair_discovery, out_dir / "10_frozen_pair_candidates.csv"),
        (final_rank, out_dir / "11_final_candidate_ranking.csv"),
        (qualified, out_dir / "12_qualified_candidates.csv"),
        (yearly_stability, out_dir / "13_yearly_candidate_stability.csv"),
        (monthly_stability, out_dir / "14_monthly_candidate_stability.csv"),
        (clock_phase, out_dir / "15_clock_phase_diagnostic_only.csv"),
        (audit, out_dir / "16_signal_audit.csv"),
        (sample, out_dir / "17_candidate_event_sample.csv"),
        (feature_dictionary, out_dir / "18_feature_dictionary.csv"),
    ]
    reporter = ProgressReporter(
        "[artifacts] R02 tables",
        len(artifact_frames),
        every=1,
        enabled=not args.no_progress,
    )
    for done, (frame, path) in enumerate(artifact_frames, start=1):
        if frame is None:
            frame = pd.DataFrame()
        frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        reporter.update(done)
    reporter.close()

    if args.write_full_events:
        events.to_csv(
            out_dir / "17b_full_conditional_events.csv.gz",
            index=False,
            compression="gzip",
            float_format="%.10g",
        )

    (out_dir / "00_research_design.md").write_text(design, encoding="utf-8")
    (out_dir / "20_research_brief.md").write_text(brief, encoding="utf-8")
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "status": "research_only_not_tradable",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "discovery_end": args.discovery_end,
        "validation_end": args.validation_end,
        "holdout_policy": "all timestamps after validation_end; never used to choose features/specs",
        "min_pressure_z": float(args.min_pressure_z),
        "pressure_threshold_policy": "frozen from R01 frequency calibration, not selected from R02 outcomes",
        "horizon_bars": list(horizons_bars),
        "tail_quantiles": list(tail_quantiles),
        "causal_features": list(features),
        "maximum_features_per_condition": 2,
        "event_thresholds_fit_on": "discovery only",
        "normal_execution_cost": float(normal_cost),
        "fee_only_cost": float(fee_only_cost),
        "valid_window_events": int(len(events)),
        "unique_event_clusters": int(events["event_cluster_id"].nunique()),
        "univariate_specs": int(len(univariate_specs)),
        "frozen_single_specs": int(eligible_single.shape[0]),
        "generated_pair_specs": int(len(pair_specs)),
        "frozen_pair_specs": int(eligible_pair.shape[0]),
        "qualified_specs": int(len(qualified)),
        "minimum_total_events": int(config.minimum_total_events),
        "target_monthly_events": [
            float(config.target_monthly_events_low),
            float(config.target_monthly_events_high),
        ],
        "books_used": False,
        "liquidity_used": False,
        "clock_phase_used_for_selection": False,
        "tp_sl_optimized": False,
        "maker_fill_assumed": False,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "19_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not args.skip_review_pack:
        finalize_research_report(
            out_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
    print(
        f"[done] report_dir={out_dir} frozen_singles={len(eligible_single):,} "
        f"frozen_pairs={len(eligible_pair):,} qualified={len(qualified):,}",
        flush=True,
    )
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def _synthetic_events() -> pd.DataFrame:
    """Conditional-engine self-test with a known broad high-activity edge."""
    rng = np.random.default_rng(20260725)
    dates = pd.date_range("2023-01-01", "2026-06-30 23:59:00", freq="8h")
    n = len(dates)
    activity = rng.normal(0.0, 1.0, n)
    effectiveness = rng.normal(0.0, 1.0, n)
    side = rng.choice([-1, 1], size=n)
    windows = rng.choice([1, 3, 5], size=n)
    broad_edge = (activity >= np.quantile(activity[dates <= "2024-12-31"], 0.67)) & (effectiveness >= -0.5)
    gross_h30 = rng.normal(0.0005, 0.0040, n) + broad_edge * 0.0022
    events = pd.DataFrame(
        {
            "event_id": np.arange(1, n + 1),
            "event_cluster_id": np.arange(1, n + 1),
            "side": side,
            "side_name": np.where(side > 0, "LONG", "SHORT"),
            "signal_bar_start": dates,
            "entry_time": dates + pd.Timedelta(minutes=1),
            "pressure_window_bars": windows,
            "flow_ratio": side * np.clip(rng.normal(0.25, 0.10, n), 0.0, 1.0),
            "trade_imbalance": side * np.clip(rng.normal(0.20, 0.10, n), 0.0, 1.0),
            "large_flow_ratio": side * np.clip(rng.normal(0.20, 0.15, n), 0.0, 1.0),
            "pressure_z": rng.uniform(2.0, 4.0, n),
            "large_notional_share": rng.uniform(0.0, 0.5, n),
            "large_trade_share": rng.uniform(0.0, 0.2, n),
            "flow_concentration": rng.uniform(0.2, 1.0, n),
            "flow_persistence": rng.uniform(0.0, 1.0, n),
            "notional_ratio": np.exp(activity * 0.4),
            "avg_trade_notional_ratio": np.exp(rng.normal(0.0, 0.3, n)),
            "max_trade_notional_ratio": np.exp(rng.normal(0.0, 0.5, n)),
            "activity_z": activity,
            "price_response_norm": rng.normal(0.2, 0.8, n),
            "pressure_effectiveness": effectiveness,
            "impact_bps_per_million": effectiveness * 3.0,
            "direction_close_location": rng.uniform(0.0, 1.0, n),
        }
    )
    for horizon in DEFAULT_HORIZON_MINUTES:
        gross = gross_h30 if horizon == 30 else rng.normal(0.0, 0.003, n)
        events[f"continuation_gross_h{horizon}"] = gross
        events[f"continuation_net_h{horizon}"] = gross - 0.0015
        events[f"reversal_gross_h{horizon}"] = -gross
        events[f"reversal_net_h{horizon}"] = -gross - 0.0015
    events = prepare_conditional_features(events)
    config = ConditionalEdgeConfig(
        minimum_total_events=300,
        minimum_discovery_events=150,
        minimum_validation_events=50,
        minimum_holdout_events=50,
        minimum_year_events=30,
        target_monthly_events_low=5.0,
        target_monthly_events_high=40.0,
        minimum_active_date_ratio=0.05,
        discovery_fdr_alpha=0.50,
    )
    return assign_time_splits(events, config)


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] R02 conditional discovery", flush=True)
    events = _synthetic_events()
    with tempfile.TemporaryDirectory(prefix="flow_impact_r02_") as tmp:
        args.out_dir = tmp
        args.minimum_total_events = 300
        args.minimum_discovery_events = 150
        args.minimum_validation_events = 50
        args.minimum_holdout_events = 50
        args.minimum_year_events = 30
        args.target_monthly_events_low = 5.0
        args.target_monthly_events_high = 40.0
        args.minimum_active_date_ratio = 0.05
        args.discovery_fdr_alpha = 0.50
        args.skip_review_pack = True
        args.no_progress = True
        audit = pd.DataFrame(
            {
                "event_id": events["event_id"],
                "entry_not_next_open_flag": False,
                "entry_before_signal_available_flag": False,
                "entry_source_bar_observed_flag": True,
                "full_forward_observed_flag": True,
                "synthetic_bar_dependency_flag": False,
                "causal_or_data_fail_flag": False,
            }
        )
        coverage = pd.DataFrame({"field": ["synthetic"], "present": [True]})
        result = run_conditional_discovery(
            events,
            audit,
            coverage,
            horizons_bars=DEFAULT_HORIZON_MINUTES,
            fee_only_cost=0.0011,
            normal_cost=0.0015,
            args=args,
        )
        required = [
            "00_research_design.md",
            "06_univariate_tail_scan.csv",
            "08_frozen_single_candidates.csv",
            "11_final_candidate_ranking.csv",
            "16_signal_audit.csv",
            "19_manifest.json",
            "20_research_brief.md",
        ]
        missing = [name for name in required if not (result["report_dir"] / name).exists()]
        if missing:
            raise AssertionError(f"missing R02 self-test artifacts: {missing}")
        scan = pd.read_csv(result["report_dir"] / "06_univariate_tail_scan.csv")
        if scan.empty or not {"discovery_net_mean", "holdout_net_mean", "discovery_fdr_q"}.issubset(scan.columns):
            raise AssertionError("R02 self-test conditional scan is incomplete")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = load_rich_trade_bars(
        project_root=PROJECT_ROOT,
        symbol=args.symbol,
        timeframe=args.timeframe,
        warmup_start_date=args.warmup_start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        db_name=args.trade_bar_db_name,
    )
    events, audit, coverage, _, horizons_bars, fee_only_cost, normal_cost = _build_event_universe(bars, args)
    run_conditional_discovery(
        events,
        audit,
        coverage,
        horizons_bars=horizons_bars,
        fee_only_cost=fee_only_cost,
        normal_cost=normal_cost,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
