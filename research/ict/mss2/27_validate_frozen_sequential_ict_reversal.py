#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-time 2025H1 validation of the frozen R27 S0-S6 definitions."""
from __future__ import annotations

import argparse
import hashlib
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
from src.research_common.ict_mss2.r27 import (  # noqa: E402
    R27Config,
    build_sequential_state_rows,
    causal_audit,
    config_record,
    prepare_root_universe,
    summarize_state_progression,
    summarize_state_quality_divergence,
    validation_decision,
)

SCRIPT_VERSION = "27.0.1"
EXPERIMENT_ID = "ETH_ICT_MSS2_SEQUENTIAL_REVERSAL_PATH_R27"
DEFAULT_R13_ROWS = "data/reports/research/ict/mss2/r13_reversal_quality_entry_discovery/04_reversal_quality_feature_rows.csv.gz"
DEFAULT_OUT = "data/reports/research/ict/mss2/r27_sequential_ict_reversal_path"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{EXPERIMENT_ID} frozen validation")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01")
    p.add_argument("--r13-rows", default=DEFAULT_R13_ROWS)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--reporting-patch", action="store_true", help="Guarded replay that may only add diagnostics while proving frozen core rows/decision unchanged")
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _independent_replay_audit(bars: pd.DataFrame, rows: pd.DataFrame, cfg: R27Config) -> pd.DataFrame:
    """Recompute entry and raw first passage without calling the R27 replay."""
    b = bars.copy(); b.index = pd.to_datetime(b.index); b = b.sort_index()
    hi = pd.to_numeric(b["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(b["low"], errors="coerce").to_numpy(float)
    op = pd.to_numeric(b["open"], errors="coerce").to_numpy(float)
    idx = b.index
    checks = {
        "market_entry_matches_raw_open": 0,
        "limit_fill_touched_raw_bar": 0,
        "outcome_matches_independent_first_passage": 0,
        "exit_time_matches_independent_first_passage": 0,
        "limit_target_not_credited_on_fill_bar": 0,
        "raw_bar_at_or_after_holdout_loaded": int((idx >= cfg.holdout_start).sum()),
    }
    for r in rows.loc[rows["entry_status"].eq("filled")].itertuples(index=False):
        ep = int(idx.searchsorted(pd.Timestamp(r.entry_time), side="left"))
        if ep >= len(idx) or idx[ep] != pd.Timestamp(r.entry_time):
            checks["market_entry_matches_raw_open"] += 1; continue
        direction = 1 if r.root_side == "SSL" else -1
        if getattr(r, "order_type", "market") == "limit":
            touched = lo[ep] <= float(r.entry_price) + 1e-12 <= hi[ep]
            checks["limit_fill_touched_raw_bar"] += int(not touched)
        else:
            checks["market_entry_matches_raw_open"] += int(not np.isclose(float(r.entry_price), op[ep], rtol=0, atol=1e-10))
        rp = int(idx.searchsorted(pd.Timestamp(r.root_sweep_time), side="left"))
        end = min(len(idx) - 1, rp + cfg.outcome_horizon_minutes)
        credit = ep + 1 if getattr(r, "order_type", "market") == "limit" else ep
        expected = "censored"; xp = end
        for i in range(ep, end + 1):
            stop_hit = lo[i] <= float(r.stop_price) + 1e-12 if direction == 1 else hi[i] >= float(r.stop_price) - 1e-12
            target_hit = i >= credit and (hi[i] >= float(r.target_price) - 1e-12 if direction == 1 else lo[i] <= float(r.target_price) + 1e-12)
            if stop_hit:
                expected, xp = "sl_first", i; break
            if target_hit:
                expected, xp = "tp_first", i; break
        checks["outcome_matches_independent_first_passage"] += int(str(r.outcome) != expected)
        checks["exit_time_matches_independent_first_passage"] += int(pd.Timestamp(r.exit_time) != idx[xp])
        if getattr(r, "order_type", "market") == "limit" and str(r.outcome) == "tp_first":
            checks["limit_target_not_credited_on_fill_bar"] += int(pd.Timestamp(r.exit_time) == pd.Timestamp(r.entry_time))
    return pd.DataFrame([{"check": k, "violations": int(v)} for k, v in checks.items()])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); cfg = R27Config().validate(); out = Path(args.out_dir)
    freeze_path = out / "07_discovery_validation_freeze.csv"
    discovery_manifest_path = out / "00_discovery_manifest.json"
    validation_manifest_path = out / "08_validation_manifest.json"
    precommit = REPO_ROOT / "research/ict/mss2/R27_PRECOMMITMENT.md"
    source = Path(args.r13_rows)
    for path in (freeze_path, discovery_manifest_path, precommit, source):
        if not path.exists():
            raise FileNotFoundError(path)
    reporting_patch = validation_manifest_path.exists() and bool(args.reporting_patch)
    if validation_manifest_path.exists() and not reporting_patch:
        raise RuntimeError("R27 validation has already been opened; refusing a second validation run")
    if args.reporting_patch and not validation_manifest_path.exists():
        raise RuntimeError("reporting patch requires an existing immutable validation run")
    old_rows = pd.read_csv(out / "10_validation_state_rows.csv.gz") if reporting_patch else None
    old_decision = pd.read_csv(out / "15_frozen_validation_decision.csv") if reporting_patch else None
    discovery_manifest = json.loads(discovery_manifest_path.read_text(encoding="utf-8"))
    if discovery_manifest["precommitment_sha256"] != _sha256(precommit):
        raise RuntimeError("precommitment changed after discovery freeze")
    freeze = pd.read_csv(freeze_path)

    print("[r27-validation] select frozen 2025H1 roots", flush=True)
    roots = prepare_root_universe(pd.read_csv(source), split="validation", config=cfg)
    print("[r27-validation] one-time physical price load through embargo end", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start, str(cfg.embargo_end)
    )
    if bars.empty:
        raise RuntimeError("no OKX validation bars")
    actual_start = pd.Timestamp(bars.index.min()); actual_end = pd.Timestamp(bars.index.max())
    if actual_end < cfg.embargo_end.floor("min"):
        raise RuntimeError(f"validation maturity coverage ends early: {actual_end}")
    if actual_end >= cfg.holdout_start:
        raise RuntimeError("holdout-era price was loaded")

    print("[r27-validation] frozen S0-S6 replay", flush=True)
    rows, diagnostics = build_sequential_state_rows(bars, roots, physical_end=cfg.embargo_end, config=cfg)
    summary = summarize_state_progression(rows)
    quality = summarize_state_quality_divergence(rows)
    audit = causal_audit(rows, diagnostics, config=cfg)
    independent = _independent_replay_audit(bars, rows, cfg)
    combined_audit = pd.concat([
        audit.assign(audit_layer="state_engine"),
        independent.assign(audit_layer="independent_raw_replay"),
    ], ignore_index=True)
    decision = validation_decision(freeze, summary, combined_audit)
    if reporting_patch:
        core = [
            "root_event_id", "state_id", "state_reached", "entry_status", "outcome",
            "entry_time", "entry_price", "stop_price", "target_price", "exit_time",
            "gross_return", "net_return_cost1x", "net_return_cost2x", "net_return_cost3x",
        ]
        core = [c for c in core if c in old_rows and c in rows]
        left = old_rows[core].reset_index(drop=True).copy()
        right = rows[core].reset_index(drop=True).copy()
        for col in [c for c in core if c.endswith("_time")]:
            left[col] = pd.to_datetime(left[col], errors="coerce")
            right[col] = pd.to_datetime(right[col], errors="coerce")
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(
            old_decision.fillna("").astype(str).reset_index(drop=True),
            decision.fillna("").astype(str).reset_index(drop=True),
        )

    manifest = {
        "script_version": SCRIPT_VERSION, "experiment_id": EXPERIMENT_ID,
        "phase": "one_time_2025H1_validation", "symbol": args.symbol,
        "discovery_manifest_sha256": _sha256(discovery_manifest_path),
        "discovery_freeze_sha256": _sha256(freeze_path),
        "precommitment_sha256": _sha256(precommit), "source_sha256": _sha256(source),
        "price_requested_start": args.warmup_start, "price_requested_end": str(cfg.embargo_end),
        "price_actual_start": str(actual_start), "price_actual_end": str(actual_end),
        "validation_root_count": len(roots), "validation_state_rows": len(rows),
        "config": config_record(cfg), "validation_opened": True, "holdout_opened": False,
        "validation_selection_runs": 1,
        "reporting_only_replays": 1 if reporting_patch else 0,
        "reporting_patch": "protected-stop diagnostic added; frozen core rows and decision byte-semantically compared unchanged" if reporting_patch else None,
    }
    validation_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    roots.to_csv(out / "09_validation_root_universe.csv.gz", index=False, compression="gzip")
    rows.to_csv(out / "10_validation_state_rows.csv.gz", index=False, compression="gzip")
    diagnostics.to_csv(out / "11_validation_path_diagnostics.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "12_validation_state_progression.csv", index=False)
    quality.to_csv(out / "13_validation_quality_divergence.csv", index=False)
    combined_audit.to_csv(out / "14_validation_causal_independent_audit.csv", index=False)
    decision.to_csv(out / "15_frozen_validation_decision.csv", index=False)
    print(decision.to_string(index=False), flush=True)
    print(f"[r27-validation] immutable result -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
