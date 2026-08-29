#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R27 discovery-only ordered ICT reversal path study.

This process physically loads price bars only through 2024-12-31.  It writes a
machine-readable state freeze before the separate validation process is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

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
    freeze_discovery_decision,
    prepare_root_universe,
    summarize_state_progression,
    summarize_state_quality_divergence,
)

SCRIPT_VERSION = "27.0.1"
EXPERIMENT_ID = "ETH_ICT_MSS2_SEQUENTIAL_REVERSAL_PATH_R27"
DEFAULT_R13_ROWS = "data/reports/research/ict/mss2/r13_reversal_quality_entry_discovery/04_reversal_quality_feature_rows.csv.gz"
DEFAULT_OUT = "data/reports/research/ict/mss2/r27_sequential_ict_reversal_path"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=EXPERIMENT_ID)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01")
    p.add_argument("--discovery-end", default="2024-12-31 23:59:59")
    p.add_argument("--r13-rows", default=DEFAULT_R13_ROWS)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = R27Config(discovery_end=pd.Timestamp(args.discovery_end)).validate()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    source = Path(args.r13_rows)
    precommit = REPO_ROOT / "research/ict/mss2/R27_PRECOMMITMENT.md"
    if not source.exists() or not precommit.exists():
        raise FileNotFoundError("R13 source rows and frozen R27 precommitment are required")

    print("[r27-discovery] select discovery roots", flush=True)
    source_rows = pd.read_csv(source)
    roots = prepare_root_universe(source_rows, split="discovery", config=cfg)
    if roots.empty:
        raise RuntimeError("no R27 discovery roots")

    print("[r27-discovery] physical OKX 1m load through discovery end only", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start, str(cfg.discovery_end)
    )
    if bars.empty:
        raise RuntimeError("no OKX 1m discovery bars")
    actual_start = pd.Timestamp(bars.index.min()); actual_end = pd.Timestamp(bars.index.max())
    if actual_end < cfg.discovery_end.floor("min"):
        raise RuntimeError(f"discovery price coverage ends early: {actual_end}")
    if actual_end >= cfg.validation_start:
        raise RuntimeError("discovery process loaded validation-era price bars")

    print("[r27-discovery] causal S0-S6 state replay", flush=True)
    rows, diagnostics = build_sequential_state_rows(bars, roots, physical_end=cfg.discovery_end, config=cfg)
    summary = summarize_state_progression(rows)
    quality = summarize_state_quality_divergence(rows)
    audit = causal_audit(rows, diagnostics, config=cfg)
    freeze = freeze_discovery_decision(summary, audit)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "discovery_only",
        "symbol": args.symbol,
        "source_rows": str(source),
        "source_sha256": _sha256(source),
        "precommitment": str(precommit.relative_to(REPO_ROOT)),
        "precommitment_sha256": _sha256(precommit),
        "price_requested_start": args.warmup_start,
        "price_requested_end": str(cfg.discovery_end),
        "price_actual_start": str(actual_start),
        "price_actual_end": str(actual_end),
        "root_count": len(roots),
        "state_row_count": len(rows),
        "config": config_record(cfg),
        "validation_opened": False,
        "holdout_opened": False,
    }
    (out / "00_discovery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    roots.to_csv(out / "01_discovery_root_universe.csv.gz", index=False, compression="gzip")
    rows.to_csv(out / "02_discovery_state_rows.csv.gz", index=False, compression="gzip")
    diagnostics.to_csv(out / "03_discovery_path_diagnostics.csv.gz", index=False, compression="gzip")
    summary.to_csv(out / "04_discovery_state_progression.csv", index=False)
    quality.to_csv(out / "05_discovery_quality_divergence.csv", index=False)
    audit.to_csv(out / "06_discovery_causal_audit.csv", index=False)
    freeze.to_csv(out / "07_discovery_validation_freeze.csv", index=False)
    (out / "07_discovery_validation_freeze.json").write_text(
        json.dumps({"experiment_id": EXPERIMENT_ID, "precommitment_sha256": _sha256(precommit), "decisions": freeze.where(pd.notna(freeze), None).to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[r27-discovery] frozen -> {out}", flush=True)
    print(freeze.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
