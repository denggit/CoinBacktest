#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R19 — Event-conditioned ICT entry study.

Frozen question
---------------
For the causally frozen 08:30 prominent-15m liquidity pair:

    raid one side -> visible 2m MSS state -> estimate P(opposite liquidity by EOD)

Then, *only when that 2m probability state was already available before the
entry order was placed*, compare entry archetypes on the same physical sweep:

    - 1m break-associated FVG near / CE
    - 2m MSS close -> next-open market
    - 2m-structure -> 1m FVG near
    - OB/FVG overlap mitigation

The study does not optimize a trading threshold.  Fixed probability bands and a
small predeclared threshold grid are diagnostics only.  No 25/50/75 targets are
used.  The only entry label is opposite-liquidity TP before terminal-extreme SL.
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

from src.research_common.ict.opposite_liquidity_probability import (
    DEFAULT_RANGE_MODEL,
    EVENT_HYPOTHESIS_FEATURES,
    fit_cumulative_hypothesis_models,
)
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

EVENT_GROUP_ORDER = [
    "H1_liquidity_context",
    "H2_terminal_maturity",
    "H3_mss_structure",
    "H4_displacement",
]

ENTRY_ARCHETYPES = (
    "mss_first_visible_close_break_next_open_market",
    "mss_first_visible_break_fvg_near",
    "mss_first_visible_break_fvg_ce",
    "mss_first_visible_ob_fvg_overlap_mid_limit",
    "mss_first_visible_2m_structure_1m_fvg_near_limit",
)

# These are diagnostic bands, not a strategy threshold search.
PROBABILITY_BINS = (-np.inf, 0.40, 0.50, 0.60, 0.70, 0.80, np.inf)
PROBABILITY_LABELS = ("<0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", ">=0.80")
DIAGNOSTIC_THRESHOLDS = (0.40, 0.50, 0.60, 0.70)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R19 event-conditioned ICT entry study")
    p.add_argument("--r18-cache-dir", default="data/reports/research/ict/soxl/mss/r18_opposite_liquidity_probability_hypotheses_alpaca_2023_2026_08")
    p.add_argument("--r16-cache-dir", default="data/reports/research/ict/soxl/mss/r16_entry_archetype_survival_atlas_alpaca_2023_2026_08")
    p.add_argument("--start-date", default="2023-07-01")
    p.add_argument("--end-date", default="2026-08-14")
    p.add_argument("--range-model", default=DEFAULT_RANGE_MODEL)
    p.add_argument("--out-dir", default="data/reports/research/ict/soxl/mss/r19_event_conditioned_entry_study")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    return p.parse_args(argv)


def _manifest(path: Path, filename: str, research_id: str, args: argparse.Namespace) -> dict[str, object]:
    p = path / filename
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    if str(data.get("research_id")) != research_id:
        raise RuntimeError(f"expected {research_id} cache, got {data.get('research_id')} at {p}")
    for key in ("start_date", "end_date"):
        if str(data.get(key)) != str(getattr(args, key)):
            raise RuntimeError(f"{research_id} cache {key}={data.get(key)} != requested {getattr(args, key)}")
    if research_id == "R18" and str(data.get("range_model")) != str(args.range_model):
        raise RuntimeError(f"R18 range_model={data.get('range_model')} != requested {args.range_model}")
    return data


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).astype("boolean").fillna(False).astype(bool)


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    pos = float(x[x > 0].sum())
    neg = float(-x[x < 0].sum())
    return pos / neg if neg > 0 else np.nan


def _rr_payoff(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    w = x[x > 0]
    l = x[x < 0]
    if len(w) == 0 or len(l) == 0:
        return np.nan
    return float(w.mean() / -l.mean())


def _period(values: pd.Series) -> pd.Series:
    d = pd.to_datetime(values, errors="coerce")
    return pd.Series(np.select(
        [d <= pd.Timestamp("2024-12-31"), d <= pd.Timestamp("2025-12-31")],
        ["discovery_2023H2_2024", "validation_2025"],
        default="forward_2026",
    ), index=values.index, dtype="object")


def _fit_frozen_2m_event_probability(event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = event.loc[event["stage"].astype(str).eq("visible_mss_2m")].copy()
    if q.empty:
        raise RuntimeError("R18 event cache has no visible_mss_2m rows")
    metrics, pred, _ = fit_cumulative_hypothesis_models(
        q,
        target="target_opposite_by_eod",
        groups=EVENT_HYPOTHESIS_FEATURES,
        group_order=EVENT_GROUP_ORDER,
    )
    # The H1-H4 full prediction is intentionally the frozen R19 event layer.
    meta = q[["path_event_id", "break_available_time", "target_opposite_by_eod", "ny_date"]].copy()
    meta = meta.sort_values("break_available_time", kind="mergesort").drop_duplicates("path_event_id", keep="first")
    out = pred.merge(meta, on=["path_event_id", "target_opposite_by_eod", "ny_date"], how="left", validate="one_to_one")
    out = out.rename(columns={"break_available_time": "event_probability_available_time"})
    out["event_probability_band"] = pd.cut(
        out["predicted_probability"],
        bins=PROBABILITY_BINS,
        labels=PROBABILITY_LABELS,
        right=False,
    ).astype(str)
    return metrics, out


def _load_lifecycle(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    p = path / "05_entry_survival_lifecycle.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    wanted = [
        "ny_date", "range_model", "path_event_id", "entry_archetype", "execution_tf", "entry_order_type",
        "entry_available_time", "fill_time", "filled", "unfilled_reason", "milestone_100_before_stop",
        "net_return_exit_100", "rr_to_100", "stop_hit", "stop_time", "target_price", "stop_price",
    ]
    header = pd.read_csv(p, nrows=0).columns.tolist()
    use = [c for c in wanted if c in header]
    q = pd.read_csv(p, usecols=use, low_memory=False)
    q = q.loc[
        q["range_model"].astype(str).eq(args.range_model)
        & q["entry_archetype"].astype(str).isin(ENTRY_ARCHETYPES)
    ].copy()
    q["period"] = _period(q["ny_date"]).to_numpy()
    q["filled"] = _to_bool(q["filled"])
    q["target_tp_before_terminal_sl"] = _to_bool(q["milestone_100_before_stop"]).astype(int)
    return q


def _causal_join(lifecycle: pd.DataFrame, event_pred: pd.DataFrame) -> pd.DataFrame:
    p = event_pred[[
        "path_event_id", "predicted_probability", "event_probability_band", "event_probability_available_time",
        "target_opposite_by_eod",
    ]].copy()
    q = lifecycle.merge(p, on="path_event_id", how="inner", validate="many_to_one")
    order_time = pd.to_datetime(q["entry_available_time"], errors="coerce", utc=True)
    prob_time = pd.to_datetime(q["event_probability_available_time"], errors="coerce", utc=True)
    q["event_probability_available_at_order"] = prob_time.notna() & order_time.notna() & prob_time.le(order_time)
    return q


def _availability_audit(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch, g in joined.groupby("entry_archetype", sort=True):
        rows.append({
            "entry_archetype": arch,
            "candidate_rows": int(len(g)),
            "probability_available_at_order": int(g["event_probability_available_at_order"].sum()),
            "availability_rate": float(g["event_probability_available_at_order"].mean()),
            "filled_rows": int(g["filled"].sum()),
            "filled_and_probability_available": int((g["filled"] & g["event_probability_available_at_order"]).sum()),
        })
    return pd.DataFrame(rows)


def _conditional_scorecard(joined: pd.DataFrame) -> pd.DataFrame:
    q = joined.loc[joined["event_probability_available_at_order"] & joined["filled"]].copy()
    rows: list[dict[str, object]] = []
    for (arch, period, band), g in q.groupby(["entry_archetype", "period", "event_probability_band"], sort=True, dropna=False):
        ret = pd.to_numeric(g["net_return_exit_100"], errors="coerce")
        rows.append({
            "entry_archetype": arch,
            "period": period,
            "event_probability_band": band,
            "filled_events": int(g["path_event_id"].nunique()),
            "mean_event_probability": float(g["predicted_probability"].mean()),
            "opposite_by_eod_rate": float(pd.to_numeric(g["target_opposite_by_eod"], errors="coerce").mean()),
            "tp_before_terminal_sl_rate": float(g["target_tp_before_terminal_sl"].mean()),
            "median_rr_to_opposite": float(pd.to_numeric(g["rr_to_100"], errors="coerce").median()),
            "mean_net_return": float(ret.mean()),
            "profit_factor": _profit_factor(ret),
            "payoff_ratio": _rr_payoff(ret),
        })
    return pd.DataFrame(rows)


def _threshold_diagnostic(joined: pd.DataFrame) -> pd.DataFrame:
    q = joined.loc[joined["event_probability_available_at_order"] & joined["filled"]].copy()
    rows: list[dict[str, object]] = []
    for threshold in DIAGNOSTIC_THRESHOLDS:
        x = q.loc[q["predicted_probability"] >= threshold].copy()
        for (arch, period), g in x.groupby(["entry_archetype", "period"], sort=True):
            ret = pd.to_numeric(g["net_return_exit_100"], errors="coerce")
            rows.append({
                "diagnostic_threshold": threshold,
                "entry_archetype": arch,
                "period": period,
                "filled_events": int(g["path_event_id"].nunique()),
                "tp_before_terminal_sl_rate": float(g["target_tp_before_terminal_sl"].mean()),
                "mean_net_return": float(ret.mean()),
                "profit_factor": _profit_factor(ret),
                "median_rr_to_opposite": float(pd.to_numeric(g["rr_to_100"], errors="coerce").median()),
            })
    return pd.DataFrame(rows)


def _paired_within_band(joined: pd.DataFrame) -> pd.DataFrame:
    q = joined.loc[joined["event_probability_available_at_order"] & joined["filled"]].copy()
    rows: list[dict[str, object]] = []
    arches = sorted(q["entry_archetype"].dropna().astype(str).unique())
    for (period, band), gb in q.groupby(["period", "event_probability_band"], sort=True):
        piv = gb.pivot_table(index="path_event_id", columns="entry_archetype", values="target_tp_before_terminal_sl", aggfunc="first")
        for i, a in enumerate(arches):
            for b in arches[i + 1:]:
                if a not in piv.columns or b not in piv.columns:
                    continue
                g = piv[[a, b]].dropna()
                if g.empty:
                    continue
                aa = g[a].astype(int); bb = g[b].astype(int)
                rows.append({
                    "period": period,
                    "event_probability_band": band,
                    "entry_a": a,
                    "entry_b": b,
                    "common_filled_events": int(len(g)),
                    "a_tp_before_sl_rate": float(aa.mean()),
                    "b_tp_before_sl_rate": float(bb.mean()),
                    "a_only_success": int(((aa == 1) & (bb == 0)).sum()),
                    "b_only_success": int(((aa == 0) & (bb == 1)).sum()),
                    "both_success": int(((aa == 1) & (bb == 1)).sum()),
                    "both_fail": int(((aa == 0) & (bb == 0)).sum()),
                })
    return pd.DataFrame(rows)


def _limit_miss_diagnostic(joined: pd.DataFrame) -> pd.DataFrame:
    q = joined.loc[joined["event_probability_available_at_order"]].copy()
    q = q.loc[q["entry_order_type"].astype(str).eq("limit")].copy()
    rows = []
    for (arch, period, band), g in q.groupby(["entry_archetype", "period", "event_probability_band"], sort=True):
        unfilled = ~g["filled"]
        reason = g.get("unfilled_reason", pd.Series("", index=g.index)).fillna("").astype(str)
        rows.append({
            "entry_archetype": arch,
            "period": period,
            "event_probability_band": band,
            "candidate_events": int(g["path_event_id"].nunique()),
            "fill_rate": float(g["filled"].mean()),
            "unfilled_events": int(unfilled.sum()),
            "opposite_before_fill": int((unfilled & reason.eq("opposite_before_fill")).sum()),
            "opposite_before_fill_rate_of_candidates": float((unfilled & reason.eq("opposite_before_fill")).mean()),
        })
    return pd.DataFrame(rows)


def _self_test() -> int:
    # Test the crucial causal availability gate independently of sklearn/R18 fitting.
    event_pred = pd.DataFrame({
        "path_event_id": ["a", "b"],
        "predicted_probability": [0.75, 0.25],
        "event_probability_band": ["0.70-0.80", "<0.40"],
        "event_probability_available_time": ["2026-01-01T10:00:00Z", "2026-01-01T10:00:00Z"],
        "target_opposite_by_eod": [1, 0],
    })
    life = pd.DataFrame({
        "path_event_id": ["a", "a", "b"],
        "entry_archetype": ["x", "y", "x"],
        "entry_available_time": ["2026-01-01T09:59:00Z", "2026-01-01T10:01:00Z", "2026-01-01T10:02:00Z"],
        "filled": [True, True, True],
        "target_tp_before_terminal_sl": [1, 1, 0],
        "milestone_100_before_stop": [True, True, False],
        "period": ["forward_2026"] * 3,
        "net_return_exit_100": [0.02, 0.02, -0.01],
        "rr_to_100": [2.0, 2.0, 3.0],
        "entry_order_type": ["limit", "limit", "limit"],
        "unfilled_reason": ["", "", ""],
    })
    q = _causal_join(life, event_pred)
    assert q["event_probability_available_at_order"].tolist() == [False, True, True]
    s = _conditional_scorecard(q)
    assert len(s) == 2
    print("R19 self-test PASS", flush=True)
    return 0


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _design_text(args: argparse.Namespace) -> str:
    return f"""# SOXL ICT R19 — Event-Conditioned Entry Study

## Frozen question
Range universe: `{args.range_model}` only.

R18 showed that the causal 2m H1-H4 state can rank `P(opposite external liquidity by EOD)`.
R19 asks whether that event probability improves actual entry survival and economics.

## Critical causality rule
A 2m event probability may condition an entry **only if the 2m snapshot available_time is <= entry_available_time**.
Earlier 1m orders are never retroactively filtered using a later 2m state.

## Entry label
Only `opposite-liquidity TP before terminal-extreme SL` is used.
No 25/50/75 target appears in this research.

## Diagnostics, not frozen thresholds
Probability bands and the fixed 0.40/0.50/0.60/0.70 threshold grid are diagnostic only.
No threshold is selected as a strategy rule in R19.
"""


def run_research(args: argparse.Namespace) -> bool:
    if args.self_test:
        return _self_test() == 0

    r18 = Path(args.r18_cache_dir); r16 = Path(args.r16_cache_dir)
    m18 = _manifest(r18, "13_manifest.json", "R18", args)
    m16 = _manifest(r16, "13_manifest.json", "R16", args)
    stage = ProgressReporter(label="[research] R19 stages", total=6, every=1, enabled=not args.no_progress)

    print("[stage 1/6] load R18 event snapshots + R16 lifecycle", flush=True)
    event = pd.read_csv(r18 / "02_event_snapshot_dataset.csv", low_memory=False)
    life = _load_lifecycle(r16, args)
    stage.update(1)

    print("[stage 2/6] fit frozen discovery-only 2m H1-H4 event probability", flush=True)
    event_metrics, event_pred = _fit_frozen_2m_event_probability(event)
    stage.update(1)

    print("[stage 3/6] causal availability join", flush=True)
    joined = _causal_join(life, event_pred)
    audit = _availability_audit(joined)
    stage.update(1)

    print("[stage 4/6] conditional entry scorecards + paired comparison", flush=True)
    score = _conditional_scorecard(joined)
    thresholds = _threshold_diagnostic(joined)
    pairs = _paired_within_band(joined)
    misses = _limit_miss_diagnostic(joined)
    stage.update(1)

    print("[stage 5/6] write reports", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "00_research_design.md").write_text(_design_text(args), encoding="utf-8")
    _write(event_metrics, out / "01_frozen_2m_event_probability_metrics.csv")
    _write(event_pred, out / "02_event_probability_predictions.csv")
    _write(audit, out / "03_causal_probability_availability_audit.csv")
    _write(score, out / "04_entry_performance_by_event_probability_band.csv")
    _write(thresholds, out / "05_fixed_threshold_diagnostic.csv")
    _write(pairs, out / "06_paired_entry_within_probability_band.csv")
    _write(misses, out / "07_limit_miss_by_probability_band.csv")
    joined_cols = [
        "ny_date", "path_event_id", "entry_archetype", "execution_tf", "entry_order_type",
        "entry_available_time", "event_probability_available_time", "event_probability_available_at_order",
        "predicted_probability", "event_probability_band", "filled", "unfilled_reason",
        "target_opposite_by_eod", "target_tp_before_terminal_sl", "net_return_exit_100", "rr_to_100", "period",
    ]
    _write(joined[[c for c in joined_cols if c in joined.columns]], out / "08_causal_conditioned_entry_lifecycle.csv")
    manifest = {
        "research_id": "R19",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "range_model": args.range_model,
        "event_layer": "visible_mss_2m H1+H2+H3+H4, discovery-fit only",
        "event_group_order": EVENT_GROUP_ORDER,
        "entry_archetypes": list(ENTRY_ARCHETYPES),
        "probability_bands": list(PROBABILITY_LABELS),
        "diagnostic_thresholds": list(DIAGNOSTIC_THRESHOLDS),
        "causality": "event_probability_available_time <= entry_available_time",
        "r18_manifest": m18,
        "r16_manifest": m16,
        "joined_rows": int(len(joined)),
        "causally_conditionable_rows": int(joined["event_probability_available_at_order"].sum()),
    }
    (out / "09_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    stage.update(1)

    print("[stage 6/6] finalize review pack", flush=True)
    if not args.skip_review_pack:
        try:
            finalize_research_report(out)
        except Exception as exc:
            print(f"[review-pack] warning: {exc}", flush=True)
    stage.update(1); stage.close()
    print(f"[done] {out}", flush=True)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return 0 if run_research(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
