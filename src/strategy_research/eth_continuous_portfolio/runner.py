from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.progress import ProgressReporter

from .config import ContinuousPortfolioConfig, PortfolioSpec, frozen_specs
from .data import load_data
from .engine import ContinuousBacktestResult, run_continuous_backtest
from .reporting import monthly_table, top_day_dependency, write_review_pack, yearly_table
from .signals import build_raw_target, build_sleeves


def _validate(cfg: ContinuousPortfolioConfig) -> None:
    start = pd.Timestamp(cfg.research_start)
    end = pd.Timestamp(cfg.research_end)
    sealed = pd.Timestamp(cfg.sealed_start)
    if start >= end:
        raise ValueError("research_start must be before research_end")
    if end >= sealed:
        raise ValueError(f"R02 refuses to open sealed data: research_end={end} sealed_start={sealed}")


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics["max_flat_days"]),
        float(metrics["max_consecutive_losing_days"]),
        float(metrics["max_drawdown_pct"]),
        -float(metrics["cagr_pct"]),
        -float(metrics["total_return_pct"]),
    )


def _write_spec(root: Path, result: ContinuousBacktestResult, schedule: pd.DataFrame) -> None:
    d = root / "specs" / result.spec_id
    d.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(d / "daily_equity.csv")
    result.rebalances.to_csv(d / "rebalances.csv", index=False)
    result.position.groupby(result.position.index.normalize()).agg(["last", "mean", "min", "max"]).to_csv(d / "daily_exposure.csv")
    schedule.to_csv(d / "target_schedule.csv")
    pd.DataFrame([{**result.metrics, **result.audit}]).to_csv(d / "summary.csv", index=False)


def run_continuous_portfolio(cfg: ContinuousPortfolioConfig | None = None, *, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or ContinuousPortfolioConfig()
    _validate(cfg)
    root = Path(cfg.report_root)
    root.mkdir(parents=True, exist_ok=True)
    specs = frozen_specs()
    pd.DataFrame([s.__dict__ for s in specs]).to_csv(root / "00_portfolio_specs.csv", index=False)

    print("[run] R02 Continuous Risk-Managed ETH Portfolio")
    print(f"[window] warmup={cfg.warmup_start} research={cfg.research_start} -> {cfg.research_end}")
    print(f"[seal] closed from {cfg.sealed_start}")
    print(f"[cost] round_trip={cfg.round_trip_cost:.4%} | single net ETH exposure")
    print("[architecture] 4 equal-weight signal families -> vol target -> optional DD governor -> deadband -> net exposure")

    data = load_data(cfg)
    sleeves = build_sleeves(data)
    sleeves.loc[pd.Timestamp(cfg.research_start) : pd.Timestamp(cfg.research_end)].to_csv(root / "07_sleeve_snapshot.csv")

    summary_rows: list[dict[str, Any]] = []
    yearly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    stress_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    base_results: dict[str, ContinuousBacktestResult] = {}
    schedules: dict[str, pd.DataFrame] = {}

    reporter = ProgressReporter("[R02 portfolio specs]", total=len(specs), every=1, enabled=progress)
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.spec_id} | {spec.name}")
        schedule = build_raw_target(sleeves, spec)
        schedules[spec.spec_id] = schedule
        base = run_continuous_backtest(data.one_minute, schedule, cfg, spec)
        base_results[spec.spec_id] = base
        _write_spec(root, base, schedule)
        summary_rows.append({"spec_id": spec.spec_id, "name": spec.name, **base.metrics})
        yearly_parts.append(yearly_table(spec.spec_id, base.daily))
        monthly_parts.append(monthly_table(spec.spec_id, base.daily))
        top_rows.extend(top_day_dependency(spec.spec_id, base.daily))
        audit_rows.append(base.audit)
        for label, cm, delay in (
            ("base", 1.0, 0),
            ("cost_2x", 2.0, 0),
            ("cost_3x", 3.0, 0),
            ("delay_plus_1m", 1.0, 1),
            ("delay_plus_2m", 1.0, 2),
        ):
            r = base if label == "base" else run_continuous_backtest(data.one_minute, schedule, cfg, spec, cost_mult=cm, extra_delay_minutes=delay)
            stress_rows.append({"spec_id": spec.spec_id, "stress": label, **r.metrics})
        reporter.update(i)
    reporter.close()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "01_summary.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(root / "02_yearly.csv", index=False)
    pd.concat(monthly_parts, ignore_index=True).to_csv(root / "03_monthly.csv", index=False)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(root / "04_cost_delay_stress.csv", index=False)
    pd.DataFrame(top_rows).to_csv(root / "05_top_day_dependency.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(root / "06_causal_audit.csv", index=False)

    selection_rows = []
    for spec in specs:
        m = base_results[spec.spec_id].metrics
        cost2 = stress[(stress.spec_id == spec.spec_id) & (stress.stress == "cost_2x")].iloc[0]
        # Gate is deliberately broad; ranking follows user's operational priority lexicographically.
        pass_gate = (
            m["total_return_pct"] > 0
            and m["profit_factor"] > 1.0
            and m["max_drawdown_pct"] <= 25.0
            and m["positive_years"] >= 2
            and float(cost2["total_return_pct"]) > 0
        )
        selection_rows.append({"spec_id": spec.spec_id, "pass_gate": pass_gate, "cost_2x_return_pct": float(cost2["total_return_pct"]), **m})
    selection = pd.DataFrame(selection_rows)
    eligible = selection[selection.pass_gate].copy()
    if not eligible.empty:
        eligible["selection_key"] = eligible.apply(lambda r: str(_selection_key(r.to_dict())), axis=1)
        ordered = sorted(eligible.spec_id.astype(str), key=lambda sid: _selection_key(base_results[sid].metrics))
        ranks = {sid: i + 1 for i, sid in enumerate(ordered)}
        selection["selection_rank"] = selection.spec_id.map(ranks)
        winner = ordered[0]
    else:
        selection["selection_rank"] = pd.NA
        winner = None
    selection.to_csv(root / "08_selection.csv", index=False)

    lines = [
        "# R02 Continuous Risk-Managed ETH Portfolio — Decision",
        "",
        "This stage tests continuous net exposure management, not discrete entry/TP/SL trades.",
        f"- Frozen portfolio specs: **{len(specs)}**.",
        f"- Gate survivors: **{int(selection.pass_gate.sum())}**.",
        f"- Selected research winner: **{winner or 'NONE'}**.",
        "- 2026 sealed holdout opened: **NO**.",
        "- Exchange execution semantics: **single net ETH exposure; no simultaneous long+short hedge legs**.",
        "",
        "## Frozen signal architecture",
        "",
        "- Channel family: Donchian multi-speed ensemble + Turtle 55/20 state.",
        "- MA family: daily 20/50 and 50/200 cross states.",
        "- TSMOM family: 21/63/126/252-day own-return direction.",
        "- 4H family: Supertrend(10,3), Keltner(20,2ATR), ADX14>=25 with EMA50 direction.",
        "- Families are equal-weighted; 90D realized vol scales to 25% annualized target; exposure hard-capped at 1.5x.",
        "",
        "## Next action",
        "",
        "If a spec survives across 2023-2025 and cost stress, freeze it before opening 2026 once. If none survive, replace the portfolio construction/risk layer rather than parameter-mining individual indicators.",
    ]
    (root / "99_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack = write_review_pack(root)
    print("[done]", root.resolve())
    print("[review-pack]", pack.resolve())
    return {"report_root": str(root), "review_pack": str(pack), "winner": winner, "survivors": int(selection.pass_gate.sum())}
