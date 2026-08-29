#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R11.1: Continuous Visible Liquidity Path Atlas for ETH.

ETH trades 24/7.  Calendar date is reporting-only; 00:00 never freezes or
resets liquidity.  Every causally confirmed, still-unconsumed 15m/30m/1H/4H
classical IT/LT swing enters the rolling active map immediately at its true
availability time.  Each sweep freezes the opposite-side active liquidity at
that exact time and follows the path continuously, including across midnight.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.r11 import (  # noqa: E402
    R11Config,
    build_continuous_path_atlas,
    build_continuous_sweep_events,
    build_event_time_liquidity_snapshot,
    build_visible_it_lt_liquidity,
    r11_causal_audit,
    summarize_first_sweep,
    summarize_path_archetypes,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "11.1.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_CONTINUOUS_VISIBLE_LIQUIDITY_PATH_R11_1"
EDGE_ID = "ICT_CONTINUOUS_VISIBLE_LIQUIDITY_PATH"
TITLE = "ETH ICT MSS2 R11.1 Continuous Visible Liquidity Path Atlas"
DEFAULT_R08_DIR = "data/reports/research/ict/mss2/r08_1_full_trend_ict_structure_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r11_1_continuous_liquidity_path_atlas"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r08-dir", default=DEFAULT_R08_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def _load_r08(path: Path, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mp = path / "00_manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"R08.1 manifest missing: {mp}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    covered = pd.Timestamp(manifest.get("research_end_date"))
    if covered < end:
        raise RuntimeError(f"R08.1 only covers through {covered}; rerun through {end}")
    hierarchy = pd.read_csv(path / "01_classical_recursive_swing_hierarchy.csv.gz")
    parts = []
    for name in ("05_trend_qualified_key_liquidity.csv.gz", "05b_nested_lower_tf_liquidity.csv.gz"):
        fp = path / name
        if fp.exists():
            parts.append(pd.read_csv(fp))
    trend = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return hierarchy, trend, manifest


def _cluster_sensitivity(bars, visible, start, end, cfg):
    rows = []
    for bps in cfg.region_cluster_sensitivities_bps:
        e = build_continuous_sweep_events(
            bars, visible, research_start=start, research_end=end, tolerance_bps=bps
        )
        rows.append({
            "cluster_bps": bps,
            "continuous_root_sweep_events": int(e.groupby("sweep_time").ngroups) if len(e) else 0,
            "side_sweep_events": len(e),
            "mean_swept_regions_per_side_event": float(pd.to_numeric(e.get("swept_region_count"), errors="coerce").mean()) if len(e) else 0.0,
            "mean_swept_levels_per_side_event": float(pd.to_numeric(e.get("swept_level_count"), errors="coerce").mean()) if len(e) else 0.0,
        })
    return pd.DataFrame(rows)


def _landmark_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    q0 = paths.loc[paths["root_sweep_side"].isin(["SSL", "BSL"])].copy()
    rows = []
    for side, p in q0.groupby("root_sweep_side", sort=True):
        for name, col in [
            ("reclaim", "reclaim_available_time"),
            ("mss_1m", "post_sweep_st_mss_1m_available_time"),
            ("mss_2m", "post_sweep_st_mss_2m_available_time"),
            ("mss_5m", "post_sweep_st_mss_5m_available_time"),
            ("directional_fvg", "first_directional_fvg_available_time"),
        ]:
            q = p.loc[p[col].notna()].copy() if col in p else pd.DataFrame()
            rows.append({
                "root_sweep_side": side,
                "landmark": name,
                "root_events": len(p),
                "available_count": len(q),
                "availability_rate": len(q) / len(p) if len(p) else float("nan"),
                "opposite_24h_hit_rate_when_available": pd.to_numeric(q.get("opposite_target_hit_1440m_flag"), errors="coerce").mean() if len(q) else float("nan"),
                "mean_ret_360m_when_available": pd.to_numeric(q.get("ret_360m"), errors="coerce").mean() if len(q) else float("nan"),
                "mean_ret_1440m_when_available": pd.to_numeric(q.get("ret_1440m"), errors="coerce").mean() if len(q) else float("nan"),
            })
    return pd.DataFrame(rows)


def _manual(out: Path, bars, visible: pd.DataFrame, paths: pd.DataFrame, sweeps: pd.DataFrame, cfg: R11Config):
    d = out / "manual_review"
    d.mkdir(parents=True, exist_ok=True)
    if len(paths):
        recent = paths.sort_values("root_sweep_time", kind="stable").tail(30)
        recent.to_csv(d / "01_recent_30_continuous_paths.csv", index=False, encoding="utf-8-sig")
        times = pd.to_datetime(recent["root_sweep_time"], errors="coerce").dropna()
        snap = build_event_time_liquidity_snapshot(
            bars, visible, times, tolerance_bps=cfg.region_cluster_bps
        )
        snap.to_csv(d / "02_recent_30_active_liquidity_at_sweep.csv", index=False, encoding="utf-8-sig")
        if len(sweeps):
            lo = times.min() - pd.Timedelta(hours=6)
            hi = times.max() + pd.Timedelta(hours=48)
            q = sweeps.loc[pd.to_datetime(sweeps["sweep_time"]).between(lo, hi, inclusive="both")]
            q.to_csv(d / "03_recent_continuous_sweep_sequence.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R11.1 manual chart review\n\n"
        "ETH is treated as continuous 24/7. There is no day-open liquidity snapshot. "
        "Start with `01_recent_30_continuous_paths.csv`; for each root sweep use "
        "`02_recent_30_active_liquidity_at_sweep.csv` to draw the IT/LT regions that "
        "were actually active at that exact sweep time. `03_recent_continuous_sweep_sequence.csv` "
        "shows subsequent sweeps and may cross midnight. Calendar date is reporting-only.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    cfg = R11Config().validate()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[r11.1] load R08.1 classical hierarchy (broad IT/LT; no trend filter)", flush=True)
    hierarchy, trend, r08_manifest = _load_r08(Path(args.r08_dir), end)
    print("[r11.1] load bare 1m K", flush=True)
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir).fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    if bars.empty:
        raise RuntimeError("No 1m OHLCV rows returned")

    print("[r11.1] continuous physical IT/LT lifecycle", flush=True)
    visible = build_visible_it_lt_liquidity(bars, hierarchy, trend)
    print("[r11.1] continuous root sweep events (no 00:00 reset)", flush=True)
    sweeps = build_continuous_sweep_events(
        bars, visible, research_start=start, research_end=end, tolerance_bps=cfg.region_cluster_bps
    )
    sensitivity = _cluster_sensitivity(bars, visible, start, end, cfg)
    print("[r11.1] continuous sweep -> opposite liquidity paths", flush=True)
    paths = build_continuous_path_atlas(bars, visible, sweeps, config=cfg)

    arch = summarize_path_archetypes(paths)
    first = summarize_first_sweep(paths)
    landmarks = _landmark_summary(paths)
    audit = r11_causal_audit(visible, sweeps, paths)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "research_start_date": args.start_date,
        "research_end_date": args.end_date,
        "r08_report": args.r08_dir,
        "r08_manifest": r08_manifest,
        "market_time_semantics": "ETH continuous 24/7; calendar date is reporting-only; no day-open/session reset",
        "liquidity_universe": "all causally confirmed 15m/30m/1H/4H classical IT/LT; ST excluded; completed-trend context descriptive only",
        "region_cluster_bps": cfg.region_cluster_bps,
        "target_freeze_semantics": "nearest opposite active unconsumed region frozen at each root sweep time",
        "path_semantics": "continuous across midnight; no session boundary",
        "strategy_semantics": "path atlas only; no promoted entry/SL/TP strategy",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    visible.to_csv(out / "01_visible_it_lt_liquidity_lifecycle.csv.gz", index=False, compression="gzip")
    sensitivity.to_csv(out / "02_region_cluster_sensitivity.csv", index=False)
    sweeps.to_csv(out / "03_continuous_root_sweep_events.csv.gz", index=False, compression="gzip")
    paths.to_csv(out / "04_continuous_liquidity_paths.csv.gz", index=False, compression="gzip")
    arch.to_csv(out / "05_path_archetype_summary.csv", index=False)
    first.to_csv(out / "06_root_sweep_path_summary.csv", index=False)
    landmarks.to_csv(out / "07_confirmation_landmark_summary.csv", index=False)
    audit.to_csv(out / "08_causal_audit.csv", index=False)
    _manual(out, bars, visible, paths, sweeps, cfg)
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r11.1] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
