#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R04: ETH liquidity-event multi-horizon opportunity atlas.

Research question
-----------------
Do causal liquidity hierarchy / consumption features tell us whether a 5m
liquidity-exhaustion reclaim is merely a short rebound (0.3%-1%), a medium move
(1%-2%), or a multi-day 3%-5%+ reversal?

R04 does *not* introduce a time-stop.  Time windows are label/censor windows.
It also does not optimize partial-exit ratios.  A small algebraic diagnostic
only reports how much of a 0.5/0.75/1.0% short target would need to be realized
to cover the original structural stop plus costs if the runner later lost at
the original stop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2 import (  # noqa: E402
    R03Config,
    R04Config,
    attach_r04_tradebar_features,
    build_4h_continuation_summary,
    build_multi_horizon_path_labels,
    build_partial_risk_coverage_summary,
    build_rule_horizon_scoreboard,
    build_tradebar_horizon_summary,
    build_tradebar_microstructure_features,
    build_transition_ladder,
    build_unique_opportunity_features,
    r03_globalize_legacy_trade_ids,
    r04_causal_audit,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "4.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_MULTI_HORIZON_LIQUIDITY_OPPORTUNITY_R04"
EDGE_ID = "RESEARCH_ONLY_ETH_LIQUIDITY_HIERARCHY_MULTI_HORIZON_LONG"
TITLE = "ETH ICT MSS2 R04 Multi-Horizon Liquidity Opportunity Atlas"
DEFAULT_R02_DIR = "data/reports/research/ict/mss2/r02_liquidity_pool_stack_structural_exit"
DEFAULT_R033_DIR = "data/reports/research/ict/mss2/r03_3_liquidity_hierarchy_entry_exit"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r04_multi_horizon_liquidity_opportunity_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--tradebar-db-name", default="okx_trade_bars.db")
    p.add_argument("--r02-report-dir", default=DEFAULT_R02_DIR)
    p.add_argument("--r033-report-dir", default=DEFAULT_R033_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--skip-tradebar", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)


def _read_r02_feature_label(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpath = report_dir / "10_trade_features_causal.csv"
    lpath = report_dir / "11_trade_structural_exit_labels.csv"
    fcols = {
        "trade_event_id", "stage_id", "episode_id", "trade_direction", "execution_minutes", "trigger_type",
        "entry_pos_1m", "entry_time", "entry_price", "stop_price", "risk_bps", "signal_available_time",
        "episode_start_time_1m", "episode_start_pos_1m", "sweep_pos_1m", "year", "quarter", "month",
        "session_primary", "is_weekend_utc",
    }
    lcols = {"trade_event_id", "stage_id", "episode_id", "target_htf240_price", "target_htf240_outcome", "target_htf240_gross_return"}
    features = _read_csv(fpath, usecols=lambda c: c in fcols)
    labels = _read_csv(lpath, usecols=lambda c: c in lcols)
    if len(features) != len(labels):
        raise RuntimeError("R04: R02 feature/label row counts differ")
    features, labels = r03_globalize_legacy_trade_ids(features, labels)
    assert labels is not None
    return features, labels


def _year_horizon_rates(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features[["trade_event_id", "entry_time"]].merge(labels, on="trade_event_id", how="inner", validate="one_to_one")
    merged["year"] = pd.to_datetime(merged["entry_time"], errors="coerce").dt.year
    fields = [
        "short_0p5_6h_flag", "short_0p75_12h_flag", "medium_1p5_1d_flag",
        "medium_2p0_2d_flag", "swing_3p0_3d_flag", "major_5p0_7d_flag",
    ]
    rows = []
    for year, part in merged.groupby("year", dropna=True, sort=True):
        row = {"year": int(year), "opportunities": len(part)}
        for c in fields:
            row[c.replace("_flag", "_rate")] = float(pd.to_numeric(part[c], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _max_tier_summary(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features[["trade_event_id", "pool_n_bucket", "contains_4h_pool_flag", "contains_lt_pool_flag"]].merge(
        labels[["trade_event_id", "max_target_before_stop_14d", "max_target_label_complete_flag"]],
        on="trade_event_id", how="inner", validate="one_to_one",
    )
    merged = merged.loc[merged["max_target_label_complete_flag"].eq(1)].copy()
    groups = {
        "all": pd.Series(True, index=merged.index),
        "n3_plus": merged["pool_n_bucket"].isin(["3", "4+"]),
        "n4_plus": merged["pool_n_bucket"].eq("4+"),
        "contains_4h": merged["contains_4h_pool_flag"].astype(bool),
        "n4_plus_4h": merged["pool_n_bucket"].eq("4+") & merged["contains_4h_pool_flag"].astype(bool),
        "n4_plus_lt": merged["pool_n_bucket"].eq("4+") & merged["contains_lt_pool_flag"].astype(bool),
    }
    rows = []
    tiers = (0.0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
    for name, mask in groups.items():
        part = merged.loc[mask]
        if part.empty:
            continue
        x = pd.to_numeric(part["max_target_before_stop_14d"], errors="coerce")
        row = {"group": name, "rows": len(part), "median_max_target": float(x.median())}
        for tier in tiers[1:]:
            token = str(tier).replace(".", "p")
            row[f"prob_reach_at_least_{token}"] = float(x.ge(tier).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _time_to_target_summary(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features[["trade_event_id", "pool_n_bucket", "contains_4h_pool_flag"]].merge(labels, on="trade_event_id", how="inner", validate="one_to_one")
    groups = {
        "all": pd.Series(True, index=merged.index),
        "n3_plus": merged["pool_n_bucket"].isin(["3", "4+"]),
        "n4_plus": merged["pool_n_bucket"].eq("4+"),
        "n4_plus_4h": merged["pool_n_bucket"].eq("4+") & merged["contains_4h_pool_flag"].astype(bool),
    }
    rows = []
    for gname, mask in groups.items():
        part = merged.loc[mask]
        for target in (0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05):
            tok = f"{target * 100:.3f}".rstrip("0").rstrip(".").replace(".", "p")
            hit = part[f"tp_{tok}_before_stop_14d_flag"].eq(1)
            x = pd.to_numeric(part.loc[hit, f"minutes_to_tp_{tok}"], errors="coerce")
            rows.append({
                "group": gname, "target_return": target, "hits_before_stop": int(hit.sum()),
                "hit_rate": float(hit.mean()) if len(part) else np.nan,
                "median_minutes": float(x.median()) if x.notna().any() else np.nan,
                "p25_minutes": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "p75_minutes": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r02_dir = Path(args.r02_report_dir)
    r033_dir = Path(args.r033_report_dir)
    cfg = R04Config(market_roundtrip_cost=float(args.market_roundtrip_cost)).validate()

    print("[r04] load R03.3 hierarchy + R02 causal trades", flush=True)
    hierarchy_rows = _read_csv(r033_dir / "09_hierarchy_trade_rows_primary_long_5m_reclaim.csv.gz")
    hierarchy_stages = _read_csv(r033_dir / "04_episode_stages_hierarchy_causal.csv.gz")
    r02_features, r02_labels = _read_r02_feature_label(r02_dir)
    features = build_unique_opportunity_features(hierarchy_rows, hierarchy_stages, r02_features, r02_labels)
    if features.empty:
        raise RuntimeError("R04 built zero opportunities")

    tradebar_audit = pd.DataFrame()
    if not args.skip_tradebar:
        print("[r04] causal trade-bar context for full opportunity universe", flush=True)
        checkpoints = features[["trade_event_id", "signal_available_time", "episode_start_time_1m"]].rename(
            columns={"trade_event_id": "checkpoint_id", "signal_available_time": "decision_time", "episode_start_time_1m": "episode_start_time"}
        )
        tb, tb_build_audit = build_tradebar_microstructure_features(
            checkpoints,
            symbol=args.symbol,
            data_dir=args.data_dir,
            db_name=args.tradebar_db_name,
            config=R03Config(),
            show_progress=not args.no_progress,
        )
        features, tradebar_audit = attach_r04_tradebar_features(features, tb)
        if not tb_build_audit.empty:
            tb_build_audit.to_csv(out_dir / "10a_tradebar_build_audit.csv", index=False)

    print("[r04] load bare 1m K and build 14-day opportunity paths", flush=True)
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if bars.empty:
        raise RuntimeError("R04 bare 1m K is empty")
    labels, path_audit = build_multi_horizon_path_labels(features, bars, cfg)

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    months = max((end - start).total_seconds() / (86400.0 * 30.4375), 1.0)
    scoreboard, yearly_scoreboard = build_rule_horizon_scoreboard(features, labels, months=months)
    transitions = build_transition_ladder(features, labels)
    continuation = build_4h_continuation_summary(features, labels)
    partial_diag = build_partial_risk_coverage_summary(features, labels)
    tradebar_horizon = build_tradebar_horizon_summary(features, labels)
    max_tier = _max_tier_summary(features, labels)
    time_to_target = _time_to_target_summary(features, labels)
    yearly_base = _year_horizon_rates(features, labels)
    causal = r04_causal_audit(features, labels, path_audit)

    violations = int(pd.to_numeric(causal["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"R04 causal audit failed violations={violations}")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "market_roundtrip_cost": cfg.market_roundtrip_cost,
        "market_roundtrip_cost_2x": cfg.market_roundtrip_cost * 2.0,
        "target_returns": list(cfg.target_returns),
        "path_horizons_minutes": list(cfg.path_horizons_minutes),
        "max_label_horizon_minutes": cfg.max_horizon_minutes,
        "opportunity_grain": "one concrete causal 5m long episode-reclaim trade_event_id; rule scoreboards use first qualifying stage per episode",
        "important_semantics": [
            "No time-stop: horizon windows are labels/censoring only.",
            "Future path labels are physically separated from causal feature rows.",
            "Same-bar fixed target and structural stop is stop-first pessimistic.",
            "Opposing 4H liquidity is treated as a decision point; continuation starts next 1m bar after first touch.",
            "Partial-profit output is algebraic feasibility only; no split-position strategy is optimized in R04.",
        ],
        "opportunities": int(len(features)),
        "episodes": int(features["episode_id"].nunique()),
        "tradebar_enabled": not args.skip_tradebar,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"metric": "opportunities", "value": len(features)},
        {"metric": "unique_episodes", "value": features["episode_id"].nunique()},
        {"metric": "feature_ids_unique", "value": int(features["trade_event_id"].nunique() == len(features))},
        {"metric": "label_ids_unique", "value": int(labels["trade_event_id"].nunique() == len(labels))},
        {"metric": "causal_violations", "value": violations},
    ]).to_csv(out_dir / "01_engineering_audit.csv", index=False)
    (out_dir / "02_frozen_design.json").write_text(json.dumps({
        "short_targets": [0.003, 0.005, 0.0075, 0.01],
        "medium_targets": [0.015, 0.02],
        "swing_targets": [0.03, 0.05],
        "nested_horizon_labels": {
            "short_0p5_6h": "+0.5% before structural SL within 6h",
            "short_0p75_12h": "+0.75% before structural SL within 12h",
            "medium_1p5_1d": "+1.5% before structural SL within 1d",
            "medium_2p0_2d": "+2.0% before structural SL within 2d",
            "swing_3p0_3d": "+3.0% before structural SL within 3d",
            "major_5p0_7d": "+5.0% before structural SL within 7d",
        },
        "rules_are_descriptive_not_optimized": [
            "any_reclaim", "it_plus", "lt", "4h_plus", "n2_plus_key", "n3_plus_key", "n4_plus_key",
            "n3_plus_4h", "n4_plus_4h", "n3_plus_lt", "n4_plus_lt",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    features.to_csv(out_dir / "03_opportunity_features_causal.csv.gz", index=False, compression="gzip")
    labels.to_csv(out_dir / "04_opportunity_future_labels.csv.gz", index=False, compression="gzip")
    scoreboard.to_csv(out_dir / "05_rule_horizon_scoreboard.csv", index=False)
    yearly_scoreboard.to_csv(out_dir / "06_rule_year_scoreboard.csv", index=False)
    transitions.to_csv(out_dir / "07_target_transition_ladder.csv", index=False)
    continuation.to_csv(out_dir / "08_4h_target_continuation_summary.csv", index=False)
    partial_diag.to_csv(out_dir / "09_partial_risk_coverage_diagnostic.csv", index=False)
    tradebar_audit.to_csv(out_dir / "10_tradebar_join_audit.csv", index=False)
    tradebar_horizon.to_csv(out_dir / "11_tradebar_horizon_summary.csv", index=False)
    causal.to_csv(out_dir / "12_causal_audit.csv", index=False)
    max_tier.to_csv(out_dir / "13_max_target_tier_summary.csv", index=False)
    time_to_target.to_csv(out_dir / "14_time_to_target_summary.csv", index=False)
    yearly_base.to_csv(out_dir / "15_year_horizon_base_rates.csv", index=False)

    readme = "# R04 Multi-Horizon Liquidity Opportunity Atlas\n\n"
    readme += "R04 does not force ETH liquidity reversals into either a day-trade or a swing-trade bucket at entry. "
    readme += "It measures nested short/medium/swing outcomes from the same causal 5m episode reclaim.\n\n"
    readme += "Key files:\n"
    readme += "- `03_opportunity_features_causal.csv.gz`: entry-time-only liquidity / structure / optional trade-bar context.\n"
    readme += "- `04_opportunity_future_labels.csv.gz`: future path labels, physically separate from features.\n"
    readme += "- `05_rule_horizon_scoreboard.csv`: frequency vs 0.5% / 1.5% / 3% / 5% opportunity quality.\n"
    readme += "- `07_target_transition_ladder.csv`: P(1%|0.5%), P(2%|1%), P(3%|2%), P(5%|3%).\n"
    readme += "- `08_4h_target_continuation_summary.csv`: how much move remains after first opposing 4H liquidity touch.\n"
    readme += "- `09_partial_risk_coverage_diagnostic.csv`: algebraic partial size needed to cover original SL+cost; not a backtest.\n"
    readme += "\nNo fixed time stop is used. Right-edge incomplete horizons are censored, not filled with partial paths.\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r04] done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
