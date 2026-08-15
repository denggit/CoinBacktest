#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R02.4 economic ceiling audit."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from .config import EconomicCeilingConfig, MODEL_NAME, STAGE_ID, STAGE_NAME


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def write_reports(
    *,
    config: EconomicCeilingConfig,
    source_gate: pd.DataFrame,
    episodes: pd.DataFrame,
    distribution: pd.DataFrame,
    performance: pd.DataFrame,
    yearly: pd.DataFrame,
    decision_gate: pd.DataFrame,
    causal: pd.DataFrame,
    decision: str,
    scanned_rows: int,
    skip_review_pack: bool,
) -> Path:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL_NAME,
        "stage_id": STAGE_ID,
        "stage_name": STAGE_NAME,
        "decision": decision,
        "scanned_source_rows": int(scanned_rows),
        "release_episodes": int(len(episodes)),
        "oracle_future_information": True,
        "tradable_strategy": False,
        "live_approval": False,
        "config": config.to_dict(),
    }
    (report_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(report_dir / "01_source_gate.csv", source_gate)
    _write_csv(report_dir / "02_episode_universe_summary.csv", (
        episodes.groupby(["period", "event_side", "outcome_type"], sort=True)
        .agg(episodes=("release_episode_id", "size"), mean_extension_bp=("future_extension_bp", "mean"), mean_reversal_bp=("future_reversal_after_extreme_bp", "mean"))
        .reset_index()
    ))
    _write_csv(report_dir / "03_oracle_ceiling_distribution.csv", distribution)
    _write_csv(report_dir / "04_fixed_r_oracle_performance.csv", performance)
    _write_csv(report_dir / "05_yearly_oracle_stability.csv", yearly)
    _write_csv(report_dir / "06_decision_gate.csv", decision_gate)
    _write_csv(report_dir / "07_causal_and_oracle_audit.csv", causal)
    sample_cols = [
        "release_episode_id", "event_time", "event_side", "period", "path_cluster",
        "outcome_type", "favorable_reversal", "event_reference_price",
        f"oracle_adverse_bp_{config.primary_horizon_seconds}s",
        f"oracle_favorable_bp_{config.primary_horizon_seconds}s",
        f"oracle_risk_bp_{config.primary_horizon_seconds}s",
        f"oracle_net_mfe_bp_{config.primary_horizon_seconds}s_c{int(config.primary_cost_bp)}",
    ]
    rr_tag = str(config.primary_reward_risk).replace(".", "p")
    sample_cols.append(f"oracle_net_bp_{config.primary_horizon_seconds}s_r{rr_tag}_c{int(config.primary_cost_bp)}")
    _write_csv(report_dir / "08_oracle_episode_sample.csv", episodes.loc[:, sample_cols].head(20_000))

    lines = [
        f"# {MODEL_NAME} {STAGE_ID} decision",
        "",
        "## Primary decision",
        "",
        f"`{decision}`",
        "",
        "## What this stage means",
        "",
        "- This is an **economic upper-bound audit**, not a strategy backtest.",
        "- The entry level is the future-known true release reference price.",
        "- The stop distance uses the future maximum adverse excursion plus a fixed buffer, so it is deliberately oracle/non-causal.",
        "- Fixed 1R/1.5R/2R targets and 11bp/22bp/33bp cost stresses test whether enough economic room exists even under this favorable setup.",
        "- Passing means only that the mechanism contains enough money to justify identification research. It does not approve a model or live trading.",
        "- Failing means the latent-liquidity reversal branch should be stopped rather than rescued with more features.",
        "",
        "## Frozen gate",
        "",
    ]
    if decision_gate.empty:
        lines.append("No gate rows were produced.")
    else:
        for row in decision_gate.to_dict("records"):
            lines.append(
                f"- {row['period']} {row['cost_gate']} {row['cost_bp']:.0f}bp: episodes={row['episodes']}, "
                f"perfect-exit mean net-MFE={row['mean_net_mfe_bp']:.2f}bp, PF={row['net_mfe_profit_factor']:.3f}, "
                f"top10-removed mean net-MFE={row['top10_removed_mean_net_mfe_bp']:.2f}bp, "
                f"positive net-MFE rate={row['positive_net_mfe_rate']:.3f} -> **{row['status']}**."
            )
    (report_dir / "09_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prompt = """# GPT review prompt\n\nReview R02.4 as an oracle economic-ceiling audit, not as a live strategy.\n\nAnswer in this order:\n1. Does the favorable-reversal oracle retain substantial net room at 11bp and 22bp in both Validation and Holdout?\n2. Is the ceiling broad or dependent on a few extreme events (top-10 removal / yearly stability)?\n3. How different are ALL_RELEASE_EPISODES, frozen reversal clusters, and the continuation control?\n4. Does the evidence justify more identification research, or should the latent-liquidity reversal branch be stopped?\n5. Do not propose model tuning unless the frozen economic ceiling passes.\n"""
    (report_dir / "GPT_REVIEW_PROMPT.md").write_text(prompt, encoding="utf-8")

    if not skip_review_pack:
        names = [
            "00_manifest.json", "01_source_gate.csv", "02_episode_universe_summary.csv",
            "03_oracle_ceiling_distribution.csv", "04_fixed_r_oracle_performance.csv",
            "05_yearly_oracle_stability.csv", "06_decision_gate.csv",
            "07_causal_and_oracle_audit.csv", "08_oracle_episode_sample.csv",
            "09_decision.md", "GPT_REVIEW_PROMPT.md",
        ]
        with zipfile.ZipFile(report_dir / "gpt_review_pack.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name in names:
                zf.write(report_dir / name, arcname=name)
    return report_dir
