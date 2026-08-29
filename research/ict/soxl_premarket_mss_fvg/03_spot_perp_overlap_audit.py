#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03 SOXL Alpaca spot vs OKX perpetual structural overlap audit.

Purpose
-------
Validate whether split-adjusted Alpaca SOXL 1m bars are a defensible long-history
proxy for the much shorter OKX SOXL-USDT-SWAP history. Both sources are clipped
to New York 04:00-16:30 before any ICT structure is built.

This is not a PnL optimization pass. The proxy gates are declared in
``src.research_common.ict.spot_perp_overlap`` and are independent of strategy
returns.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE  # noqa: E402
from src.research_common.ict.premarket_mss_fvg import (  # noqa: E402
    NY_TZ,
    ResearchConfig,
    build_data_quality_table,
    eligible_ny_dates,
    make_synthetic_ict_day,
    ny_date_bounds_to_source_naive,
    source_naive_to_new_york,
)
from src.research_common.ict.premarket_mss_fvg_v2 import (  # noqa: E402
    build_all_premarket_levels_v2,
    build_signal_attempts_v2,
    build_sweep_events_v2,
)
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_aligned_minute_paths,
    build_equity_proxy_data_quality_table,
    clip_equity_research_session,
    densify_equity_minutes_causally,
    pair_unique_events,
    summarize_proxy_audit,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

DEFAULT_START = "2026-05-20"
DEFAULT_END = "2026-06-30"
DEFAULT_OUT = "data/reports/research/ict/soxl/mss/r03_spot_perp_overlap_audit"


def _source_offset_hours(text: str) -> int:
    value = str(text).strip().upper().replace("UTC", "")
    try:
        return int(value)
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOXL spot/perpetual overlap audit before long-history ICT MSS research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--okx-symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", choices=("sip", "iex", "boats"), default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_spot(args: argparse.Namespace) -> pd.DataFrame:
    start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
    end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
    start_utc = start_ny.tz_convert("UTC")
    end_utc = end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1)
    loader = AlpacaStockLoader(
        symbol=args.alpaca_symbol,
        timeframe="1Min",
        feed=args.alpaca_feed,
        adjustment=args.alpaca_adjustment,
        data_dir=args.data_dir,
    )
    raw = loader.fetch_data_by_date_range(start_utc, end_utc, local_only=True)
    if raw.empty:
        raise RuntimeError(
            f"No local Alpaca rows in {loader.db_path} table={loader.table_name} for overlap window"
        )
    out = raw.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert(NY_TZ)
    out.index.name = "bar_start_ny"
    return densify_equity_minutes_causally(out)


def _load_perp(args: argparse.Namespace) -> pd.DataFrame:
    offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
    start_source, end_source = ny_date_bounds_to_source_naive(
        args.start_date,
        args.end_date,
        source_offset_hours=offset,
    )
    loader = OKXDataLoader(symbol=args.okx_symbol, timeframe="1m", db_dir=args.data_dir)
    raw = loader.load_local_data()
    if not raw.empty:
        raw = raw.loc[(raw.index >= start_source) & (raw.index <= end_source)].copy()
    if raw.empty:
        raise RuntimeError(
            "No local OKX SOXL 1m overlap data. R03 intentionally does not fetch remote data; "
            "prebuild/load the overlap through src.data_feed first."
        )
    return clip_equity_research_session(source_naive_to_new_york(raw, source_offset_hours=offset))


def _event_tables(bars: pd.DataFrame, days: list, cfg: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = build_all_premarket_levels_v2(
        bars,
        days,
        pivot_left=cfg.premarket_pivot_left,
        pivot_right=cfg.premarket_pivot_right,
    )
    sweeps = build_sweep_events_v2(bars, levels)
    attempts = build_signal_attempts_v2(
        bars,
        sweeps,
        config=cfg,
        displacement_body_multipliers=[cfg.displacement_body_mult],
    )
    return levels, sweeps, attempts


def _pair_levels(spot_levels: pd.DataFrame, perp_levels: pd.DataFrame, daily_paths: pd.DataFrame) -> pd.DataFrame:
    if spot_levels.empty or perp_levels.empty:
        return pd.DataFrame()
    s = spot_levels.loc[spot_levels["level_type"].astype(str).eq("premarket_extreme")].copy()
    p = perp_levels.loc[perp_levels["level_type"].astype(str).eq("premarket_extreme")].copy()
    cols = ["ny_date", "liquidity_side", "level_price", "source_bar_time"]
    out = s[cols].rename(columns={"level_price": "spot_level_price", "source_bar_time": "spot_level_time"}).merge(
        p[cols].rename(columns={"level_price": "perp_level_price", "source_bar_time": "perp_level_time"}),
        on=["ny_date", "liquidity_side"],
        how="inner",
    )
    if out.empty:
        return out
    out["abs_level_time_diff_minutes"] = (
        pd.to_datetime(out["spot_level_time"]) - pd.to_datetime(out["perp_level_time"])
    ).abs().dt.total_seconds() / 60.0
    if not daily_paths.empty:
        out = out.merge(daily_paths[["ny_date", "median_basis_ratio"]], on="ny_date", how="left")
        denom = pd.to_numeric(out["spot_level_price"], errors="coerce") * pd.to_numeric(out["median_basis_ratio"], errors="coerce")
        out["basis_adjusted_level_diff_bps"] = (
            pd.to_numeric(out["perp_level_price"], errors="coerce") / denom - 1.0
        ) * 10_000
    return out


def _findings(metrics: pd.DataFrame, detail: dict[str, object], common_days: int) -> str:
    lines = [
        "# SOXL Spot ↔ Perpetual Overlap Audit R03",
        "",
        f"- Verdict: **{detail['verdict']}**",
        f"- Common fully-covered New York sessions: **{common_days}**",
        "- Both sources are clipped to **04:00-16:30 America/New_York** before structure construction.",
        "- The gates below are structural-proxy gates declared before looking at long-history PnL; they are not parameter optimization.",
        "",
        "## Gate metrics",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            f"- `{row['metric']}` = {float(row['value']):.6f}; "
            f"PASS >= {float(row['pass_threshold']):.6f}; caution >= {float(row['caution_threshold']):.6f}."
        )
    lines += [
        "",
        "## Interpretation",
        "- PASS: Alpaca split-adjusted SOXL can be used as a long-history **structure proxy**, but final live/perpetual validation still belongs to OKX overlap/forward data.",
        "- CAUTION: long-history spot research is still useful for hypothesis screening, but do not promote a strategy from spot PnL alone.",
        "- FAIL: do not use Alpaca history to claim an OKX perpetual ICT edge; investigate the structural mismatches first.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_audit_from_frames(
    spot_ny: pd.DataFrame,
    perp_ny: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = ResearchConfig(required_day_coverage=float(args.required_day_coverage))
    spot_ny = densify_equity_minutes_causally(spot_ny)
    perp_ny = clip_equity_research_session(perp_ny)

    days = eligible_ny_dates(
        spot_ny,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_equity_holidays=not bool(args.include_us_equity_holidays),
    )
    spot_quality = build_equity_proxy_data_quality_table(spot_ny, days)
    perp_quality = build_data_quality_table(perp_ny, days, required_coverage=cfg.required_day_coverage)
    spot_valid = set(spot_quality.loc[spot_quality["coverage_pass"], "ny_date"].astype(str))
    perp_valid = set(perp_quality.loc[perp_quality["coverage_pass"], "ny_date"].astype(str))
    common_text = sorted(spot_valid & perp_valid)
    common_days = [pd.Timestamp(x).date() for x in common_text]
    if not common_days:
        raise RuntimeError("No common fully-covered sessions for spot/perpetual overlap audit")

    progress = ProgressReporter(label="[audit] structure stages", total=4, every=1, enabled=not bool(args.no_progress))
    aligned, daily_paths = build_aligned_minute_paths(spot_ny, perp_ny)
    progress.update(1)
    spot_levels, spot_sweeps, spot_attempts = _event_tables(spot_ny, common_days, cfg)
    progress.update(2)
    perp_levels, perp_sweeps, perp_attempts = _event_tables(perp_ny, common_days, cfg)
    progress.update(3)

    # Limit path metrics to common valid dates, not partial/bad sessions.
    aligned = aligned.loc[aligned["ny_date"].isin(common_text)].copy() if not aligned.empty else aligned
    daily_paths = daily_paths.loc[daily_paths["ny_date"].isin(common_text)].copy() if not daily_paths.empty else daily_paths
    metrics, detail = summarize_proxy_audit(
        spot_ny=spot_ny.loc[[str(ts.date()) in set(common_text) for ts in spot_ny.index]],
        perp_ny=perp_ny.loc[[str(ts.date()) in set(common_text) for ts in perp_ny.index]],
        aligned=aligned,
        daily_paths=daily_paths,
        spot_sweeps=spot_sweeps,
        perp_sweeps=perp_sweeps,
        spot_attempts=spot_attempts,
        perp_attempts=perp_attempts,
    )
    progress.update(4)
    progress.close()

    level_pairs = _pair_levels(spot_levels, perp_levels, daily_paths)
    sweep_pairs = pair_unique_events(
        spot_sweeps.loc[spot_sweeps["level_type"].astype(str).eq("premarket_extreme")].copy() if not spot_sweeps.empty else spot_sweeps,
        perp_sweeps.loc[perp_sweeps["level_type"].astype(str).eq("premarket_extreme")].copy() if not perp_sweeps.empty else perp_sweeps,
        keys=["ny_date", "trade_side"],
        spot_time_col="sweep_time",
    )
    spot_base = spot_attempts.loc[spot_attempts["level_type"].astype(str).eq("premarket_extreme")].copy() if not spot_attempts.empty else spot_attempts
    perp_base = perp_attempts.loc[perp_attempts["level_type"].astype(str).eq("premarket_extreme")].copy() if not perp_attempts.empty else perp_attempts
    setup_pairs = pair_unique_events(
        spot_base,
        perp_base,
        keys=["ny_date", "trade_side", "execution_tf"],
        spot_time_col="signal_time",
    )

    _write_csv(spot_quality, out_dir / "01_spot_data_quality.csv")
    _write_csv(perp_quality, out_dir / "02_perp_data_quality.csv")
    _write_csv(daily_paths, out_dir / "03_daily_rebased_path_alignment.csv")
    _write_csv(level_pairs, out_dir / "04_premarket_extreme_alignment.csv")
    _write_csv(sweep_pairs, out_dir / "05_external_sweep_alignment.csv")
    _write_csv(setup_pairs, out_dir / "06_base_setup_alignment.csv")
    _write_csv(metrics, out_dir / "07_proxy_gate_metrics.csv")
    (out_dir / "08_findings.md").write_text(_findings(metrics, detail, len(common_days)), encoding="utf-8")
    manifest = {
        "experiment_id": "SOXL_ICT_SPOT_PERP_OVERLAP_R03",
        "title": "SOXL Alpaca spot vs OKX perpetual structural overlap audit",
        "start_date_ny": args.start_date,
        "end_date_ny": args.end_date,
        "common_valid_sessions": len(common_days),
        "spot_loader": "src.data_feed.alpaca_stock_loader.AlpacaStockLoader",
        "spot_symbol": args.alpaca_symbol,
        "spot_feed": args.alpaca_feed,
        "spot_adjustment": args.alpaca_adjustment,
        "perp_loader": "src.data_feed.okx_loader.OKXDataLoader",
        "perp_symbol": args.okx_symbol,
        "session_clip_ny": "04:00-16:30",
        "verdict": detail["verdict"],
        "detail": detail,
    }
    (out_dir / "09_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not bool(args.skip_review_pack):
        finalize_research_report(
            out_dir,
            experiment_id=manifest["experiment_id"],
            edge_id="SOXL_ICT_PM_SWEEP_MSS_FVG",
            title=manifest["title"],
            print_log=True,
        )
    print(
        f"[verdict] {detail['verdict']} | common_sessions={len(common_days)} | "
        + " | ".join(f"{r['metric']}={float(r['value']):.4f}" for r in metrics.to_dict("records")),
        flush=True,
    )
    return {"report_dir": out_dir, "verdict": detail["verdict"], "metrics": metrics, "detail": detail}


def run_self_test(args: argparse.Namespace) -> int:
    spot = make_synthetic_ict_day()
    perp = spot.copy()
    for col in ("open", "high", "low", "close"):
        perp[col] = pd.to_numeric(perp[col], errors="coerce") * 1.003
    with tempfile.TemporaryDirectory(prefix="soxl_overlap_r03_") as tmp:
        args.start_date = "2026-06-02"
        args.end_date = "2026-06-02"
        args.out_dir = tmp
        args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0
        args.no_progress = True
        args.skip_review_pack = True
        result = run_audit_from_frames(spot, perp, args)
        if result["verdict"] != "PASS":
            raise AssertionError(result["metrics"].to_dict("records"))
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    spot = _load_spot(args)
    perp = _load_perp(args)
    print(f"[load] spot_rows={len(spot):,} perp_rows={len(perp):,}", flush=True)
    run_audit_from_frames(spot, perp, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
