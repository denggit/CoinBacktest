from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.strategy_research.eth_source_locked_portfolio.config import SourceLockedConfig
from src.strategy_research.eth_source_locked_portfolio.data import load_data
from src.strategy_research.eth_source_locked_portfolio.engine import run_turtle_system2
from src.strategy_research.eth_source_locked_portfolio.rules import build_turtle_context

from .atlas import build_path_tables, checkpoint_outcome_stats, grouped_episode_stats
from .config import TurtlePathConfig
from .reporting import write_decision, write_review_pack


def _validate(cfg: TurtlePathConfig) -> None:
    if pd.Timestamp(cfg.research_end) >= pd.Timestamp(cfg.sealed_start):
        raise ValueError(f"R04 refuses sealed data: research_end={cfg.research_end} sealed_start={cfg.sealed_start}")
    if pd.Timestamp(cfg.discovery_end) >= pd.Timestamp(cfg.research_end):
        raise ValueError("discovery_end must be earlier than research_end so 2025 remains validation")


def run_turtle_path_atlas(cfg: TurtlePathConfig | None = None) -> dict[str, str]:
    cfg = cfg or TurtlePathConfig()
    _validate(cfg)
    root = Path(cfg.report_root)
    root.mkdir(parents=True, exist_ok=True)

    print("[run] R04 Turtle Path Atlas")
    print(f"[window] warmup={cfg.warmup_start} research={cfg.research_start} -> {cfg.research_end}")
    print(f"[discovery] through {cfg.discovery_end}; [validation] 2025; [seal] {cfg.sealed_start}+ CLOSED")
    print("[principle] learn the path; do not alter Turtle entry/exit rules")

    base_cfg = SourceLockedConfig(
        symbol=cfg.symbol,
        warmup_start=cfg.warmup_start,
        research_start=cfg.research_start,
        research_end=cfg.research_end,
        sealed_start=cfg.sealed_start,
        round_trip_cost=cfg.round_trip_cost,
        initial_capital=cfg.initial_capital,
        timezone_offset_hours=cfg.timezone_offset_hours,
    )
    data = load_data(base_cfg)
    context = build_turtle_context(data.daily())
    baseline = run_turtle_system2(data.one_minute, context, base_cfg)
    episodes, checkpoints, adds = build_path_tables(
        data.one_minute,
        baseline.events,
        baseline.minute_equity,
        context,
        discovery_end=cfg.discovery_end,
        checkpoints_minutes=cfg.checkpoints_minutes,
    )
    stats = grouped_episode_stats(episodes)
    checkpoint_stats = checkpoint_outcome_stats(checkpoints)

    episodes.to_csv(root / "01_episode_summary.csv", index=False)
    checkpoints.to_csv(root / "02_path_checkpoints.csv", index=False)
    adds.to_csv(root / "03_add_stage_paths.csv", index=False)
    stats.to_csv(root / "04_episode_group_stats.csv", index=False)
    checkpoint_stats.to_csv(root / "05_checkpoint_outcome_stats.csv", index=False)
    baseline.events.to_csv(root / "06_baseline_turtle_events.csv", index=False)
    pd.DataFrame([baseline.metrics]).to_csv(root / "07_baseline_metrics.csv", index=False)
    pd.DataFrame([baseline.audit]).to_csv(root / "08_causal_audit.csv", index=False)
    write_decision(root, episodes, stats, checkpoint_stats)
    pack = write_review_pack(root)
    print(f"[episodes] {len(episodes)}")
    print("[done]", root.resolve())
    print("[review-pack]", pack.resolve())
    return {"report_root": str(root), "review_pack": str(pack)}
