#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble immutable R27 discovery + validation artifacts and figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.ict_mss2.r27 import summarize_protected_stop_diagnostic  # noqa: E402

EXPERIMENT_ID = "ETH_ICT_MSS2_SEQUENTIAL_REVERSAL_PATH_R27"
EDGE_ID = "REJECTED_SEQUENTIAL_COMPLETED_TREND_REVERSAL"
TITLE = "ETH ICT MSS2 R27 Sequential ICT Reversal Path Study"
DEFAULT_OUT = "data/reports/research/ict/mss2/r27_sequential_ict_reversal_path"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    return p.parse_args(argv)


def _load_required(out: Path, name: str) -> pd.DataFrame:
    path = out / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _plot_progression(summary: pd.DataFrame, figures: Path) -> None:
    q = summary.loc[summary["grain"].eq("overall")].copy()
    q["expectancy_cost2x_pct"] = pd.to_numeric(q["expectancy_cost2x"], errors="coerce") * 100.0
    colors = {"discovery": "#2364aa", "validation": "#d1495b"}
    font = ImageFont.load_default(); bold = font

    def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], side: str, metric: str, ymin: float, ymax: float, *, reference: float | None = None, sample_col: str = "reached") -> None:
        x0, y0, x1, y1 = box; left, right, top, bottom = x0 + 72, x1 - 24, y0 + 46, y1 - 62
        lookup_side = side.split(" —", 1)[0]
        draw.rectangle((left, top, right, bottom), outline="#777777", width=1)
        for k in range(5):
            yy = top + (bottom - top) * k / 4
            draw.line((left, yy, right, yy), fill="#dddddd", width=1)
            val = ymax - (ymax - ymin) * k / 4
            draw.text((left - 56, yy - 6), f"{val:.2f}", fill="#333333", font=font)
        if reference is not None and ymin <= reference <= ymax:
            yy = bottom - (reference - ymin) / (ymax - ymin) * (bottom - top)
            draw.line((left, yy, right, yy), fill="#222222", width=2)
        title = side if " —" in side else f"{side} ({'Long' if lookup_side == 'SSL' else 'Short'})"
        draw.text((x0 + 12, y0 + 12), title, fill="#111111", font=bold)
        for split in ("discovery", "validation"):
            p = q.loc[(q["root_side"] == lookup_side) & (q["research_split"] == split)].sort_values("state_id")
            pts: list[tuple[float, float]] = []
            for _, r in p.iterrows():
                sid = int(r["state_id"]); value = float(r[metric]) if pd.notna(r[metric]) else np.nan
                xx = left + sid / 6 * (right - left)
                draw.text((xx - 7, bottom + 16), f"S{sid}", fill="#333333", font=font)
                if not np.isfinite(value): continue
                value = min(max(value, ymin), ymax); yy = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
                pts.append((xx, yy)); draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=colors[split])
                n = int(r[sample_col]); draw.text((xx - 14, yy - 20), f"n={n}", fill=colors[split], font=font)
            if len(pts) > 1: draw.line(pts, fill=colors[split], width=3)

    for metric, ylabel, name, ymax in (
        ("root_reach_rate", "Share of root events", "01_state_reach_funnel.png", 1.0),
        ("direct_delivery_probability", "Direct opposite-delivery probability", "02_direct_delivery_by_state.png", 0.55),
    ):
        im = Image.new("RGB", (1400, 620), "white"); draw = ImageDraw.Draw(im)
        draw.text((24, 18), f"R27 — {ylabel}", fill="#111111", font=bold)
        draw.line((1020, 24, 1060, 24), fill=colors["discovery"], width=4); draw.text((1068, 18), "Discovery", fill="#111111", font=font)
        draw.line((1180, 24, 1220, 24), fill=colors["validation"], width=4); draw.text((1228, 18), "Validation", fill="#111111", font=font)
        panel(draw, (10, 58, 695, 610), "SSL", metric, 0, ymax)
        panel(draw, (705, 58, 1390, 610), "BSL", metric, 0, ymax)
        im.save(figures / name)

    im = Image.new("RGB", (1400, 1040), "white"); draw = ImageDraw.Draw(im)
    draw.text((24, 16), "R27 — executable economics by ordered state", fill="#111111", font=bold)
    draw.line((1020, 24, 1060, 24), fill=colors["discovery"], width=4); draw.text((1068, 18), "Discovery", fill="#111111", font=font)
    draw.line((1180, 24, 1220, 24), fill=colors["validation"], width=4); draw.text((1228, 18), "Validation", fill="#111111", font=font)
    # Rare S5/S6 one-winner PFs are clipped at 3; n labels expose the instability.
    panel(draw, (10, 54, 695, 535), "SSL — 2x expectancy (%)", "expectancy_cost2x_pct", -2.0, 2.0, reference=0, sample_col="filled")
    panel(draw, (705, 54, 1390, 535), "BSL — 2x expectancy (%)", "expectancy_cost2x_pct", -2.0, 2.0, reference=0, sample_col="filled")
    panel(draw, (10, 545, 695, 1028), "SSL — 2x PF (clipped at 3)", "pf_cost2x", 0, 3, reference=1, sample_col="filled")
    panel(draw, (705, 545, 1390, 1028), "BSL — 2x PF (clipped at 3)", "pf_cost2x", 0, 3, reference=1, sample_col="filled")
    im.save(figures / "03_state_economics_cost2x.png")


def _manual_review(out: Path, rows: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    d = out / "manual_review"; d.mkdir(parents=True, exist_ok=True)
    reached = rows.loc[rows["state_reached"].eq(1)].copy()
    highest = reached.sort_values(["root_event_id", "state_id"], kind="stable").groupby(["research_split", "root_event_id"], as_index=False).tail(1)
    highest.sort_values("root_sweep_time", kind="stable").tail(50).to_csv(d / "01_recent_50_highest_state_paths.csv", index=False, encoding="utf-8-sig")
    highest.loc[(highest["state_id"] >= 3) & highest["direct_reversal_label"].eq(1)].tail(30).to_csv(d / "02_meaningful_mss_direct_reversals.csv", index=False, encoding="utf-8-sig")
    highest.loc[(highest["state_id"] >= 3) & highest["direct_reversal_label"].eq(0)].tail(30).to_csv(d / "03_meaningful_mss_false_reversals.csv", index=False, encoding="utf-8-sig")
    reached.loc[reached["state_id"].ge(5)].sort_values(["root_sweep_time", "state_id"], kind="stable").to_csv(d / "04_all_fvg_and_protected_paths.csv", index=False, encoding="utf-8-sig")
    filled = reached.loc[reached["entry_status"].eq("filled")].copy()
    filled.sort_values("net_return_cost2x", ascending=False, kind="stable").head(30).to_csv(d / "05_best_30_filled_stage_entries.csv", index=False, encoding="utf-8-sig")
    filled.sort_values("net_return_cost2x", ascending=True, kind="stable").head(30).to_csv(d / "06_worst_30_filled_stage_entries.csv", index=False, encoding="utf-8-sig")
    for (split, year), p in highest.groupby(["research_split", "year"], sort=True):
        sample = pd.concat([p.loc[p["direct_reversal_label"].eq(1)].tail(5), p.loc[p["direct_reversal_label"].eq(0)].tail(5)]).sort_values("root_sweep_time", kind="stable")
        sample.to_csv(d / f"07_{split}_{int(year)}_representative_paths.csv", index=False, encoding="utf-8-sig")
    diagnostics.sort_values("root_sweep_time", kind="stable").tail(50).to_csv(d / "08_recent_50_state_diagnostics.csv", index=False, encoding="utf-8-sig")
    (d / "README.md").write_text(
        "# R27 chart-review pack\n\nEach root is a completed-trend ITH/ITL/LTH/LTL sweep. "
        "SSL maps to a Long reversal and BSL to Short. `highest_state_paths` keeps one final causal state per root; "
        "the FVG/protected file is intentionally exhaustive because those states are rare. Entry outcomes use the "
        "common sweep-extreme + 0.10 ATR invalidation and frozen opposite-liquidity target. Same-bar ambiguity is stop-first.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); out = Path(args.out_dir)
    dm = json.loads((out / "00_discovery_manifest.json").read_text(encoding="utf-8"))
    vm = json.loads((out / "08_validation_manifest.json").read_text(encoding="utf-8"))
    discovery_rows = _load_required(out, "02_discovery_state_rows.csv.gz")
    validation_rows = _load_required(out, "10_validation_state_rows.csv.gz")
    discovery_diag = _load_required(out, "03_discovery_path_diagnostics.csv.gz")
    validation_diag = _load_required(out, "11_validation_path_diagnostics.csv.gz")
    discovery_summary = _load_required(out, "04_discovery_state_progression.csv")
    validation_summary = _load_required(out, "12_validation_state_progression.csv")
    freeze = _load_required(out, "07_discovery_validation_freeze.csv")
    decision = _load_required(out, "15_frozen_validation_decision.csv")
    audits = pd.concat([
        _load_required(out, "06_discovery_causal_audit.csv").assign(phase="discovery", audit_layer="state_engine"),
        _load_required(out, "14_validation_causal_independent_audit.csv").assign(phase="validation"),
    ], ignore_index=True)
    rows = pd.concat([discovery_rows, validation_rows], ignore_index=True)
    diagnostics = pd.concat([discovery_diag, validation_diag], ignore_index=True)
    summary = pd.concat([discovery_summary, validation_summary], ignore_index=True)
    quality = pd.concat([_load_required(out, "05_discovery_quality_divergence.csv"), _load_required(out, "13_validation_quality_divergence.csv")], ignore_index=True)

    summary.to_csv(out / "16_full_state_progression.csv", index=False)
    quality.to_csv(out / "17_full_quality_divergence.csv", index=False)
    audits.to_csv(out / "18_full_causal_audit.csv", index=False)
    summarize_protected_stop_diagnostic(rows).to_csv(out / "17b_protected_stop_diagnostic.csv", index=False)
    pd.DataFrame([{
        "holdout_start": "2025-08-01 00:00:00", "holdout_price_rows_loaded": 0,
        "holdout_root_rows_in_outputs": int(pd.to_datetime(rows["root_sweep_time"]).ge("2025-08-01").sum()),
        "status": "SEALED_UNOPENED",
    }]).to_csv(out / "19_holdout_seal.csv", index=False)
    pd.DataFrame([
        {"check": "discovery_roots", "value": discovery_rows["root_event_id"].nunique()},
        {"check": "validation_roots", "value": validation_rows["root_event_id"].nunique()},
        {"check": "state_rows", "value": len(rows)},
        {"check": "causal_and_replay_violations", "value": int(pd.to_numeric(audits["violations"], errors="coerce").fillna(0).sum())},
        {"check": "discovery_directions_frozen", "value": int(freeze["decision"].eq("FREEZE_FOR_VALIDATION").sum())},
        {"check": "validation_directions_advanced", "value": int(decision["decision"].eq("ADVANCE").sum())},
    ]).to_csv(out / "20_engineering_decision_audit.csv", index=False)

    figures = out / "figures"; figures.mkdir(parents=True, exist_ok=True)
    _plot_progression(summary, figures); _manual_review(out, rows, diagnostics)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID, "title": TITLE,
        "study_status": "complete_rejected_no_stable_divergence", "strategy_promoted": False,
        "discovery_manifest": dm, "validation_manifest": vm,
        "discovery_decision": freeze.where(pd.notna(freeze), None).to_dict("records"),
        "validation_decision": decision.where(pd.notna(decision), None).to_dict("records"),
        "holdout_status": "SEALED_UNOPENED",
        "primary_result": "No S0-S6 state passes the frozen discovery gate; SSL S2/S3 positive discovery economics fail 2025H1 and top-five removal; BSL is negative throughout.",
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r27-finalize] complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
