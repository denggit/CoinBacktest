from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.progress import ProgressReporter

from .config import SourceLockedConfig
from .data import load_data
from .engine import BacktestResult, run_target_schedule, run_turtle_system2
from .reporting import monthly, top_days, write_review_pack, yearly
from .rules import build_mop_tsmom, build_turtle_context, build_zarattini


def _validate(cfg: SourceLockedConfig) -> None:
    if pd.Timestamp(cfg.research_end) >= pd.Timestamp(cfg.sealed_start):
        raise ValueError(f"R03 refuses sealed data: research_end={cfg.research_end} sealed_start={cfg.sealed_start}")


def _rank_key(m: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(m["max_flat_days"]),
        float(m["max_consecutive_losing_days"]),
        float(m["max_drawdown_pct"]),
        -float(m["cagr_pct"]),
        -float(m["total_return_pct"]),
    )


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "strategy_id": "SL01_ZARATTINI_LONG",
            "source_confidence": "HIGH_CORE_RULES",
            "replication_scope": "SOURCE_CORE + ETH_EXECUTION_ADAPTATION",
            "source": "Zarattini, Pagani, Barbon (2025), Catching Crypto Trends, Section 4.3",
            "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907",
            "source_rule": "9 Donchian CLOSE channels, midpoint trailing stop, 25%/90D-vol sizing, 2x cap, equal-weight, 20% vol rebalance threshold",
            "adaptation": "ETH-USDT-SWAP single asset; +8 day; 0.11% project cost; next-observable-open causal execution",
        },
        {
            "strategy_id": "SL02_ZARATTINI_LS",
            "source_confidence": "HIGH_CORE_RULES",
            "replication_scope": "SOURCE_CORE + ETH_PERP_ADAPTATION",
            "source": "Zarattini, Pagani, Barbon (2025), Appendix Long-Short Implementation",
            "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907",
            "source_rule": "Symmetric long-short Donchian extension with direction-aware monotone midpoint trailing stops; same sizing and ensemble",
            "adaptation": "ETH perpetual shorting is directly tradable; funding intentionally excluded because project baseline tests directional strategy before financing overlay",
        },
        {
            "strategy_id": "SL03_MOP_TSMOM_12M",
            "source_confidence": "HIGH_SIGNAL_MEDIUM_VOL_IMPLEMENTATION",
            "replication_scope": "SOURCE_SIGNAL/SIZING + SINGLE_ETH/VOL_ESTIMATOR_ADAPTATION",
            "source": "Moskowitz, Ooi, Pedersen (2012), Time Series Momentum",
            "source_url": "https://www.sciencedirect.com/science/article/pii/S0304405X11002613",
            "source_rule": "Long positive past-12M return / short negative; position size 40% divided by ex-ante volatility; canonical monthly holding",
            "adaptation": "Past 12M raw ETH-perp return is used as the tradable proxy for the paper's contract excess return; 365 crypto days for 12 months; standard 60D EWMA vol replication; single ETH rather than diversified 58 futures",
        },
        {
            "strategy_id": "SL04_TURTLE_SYSTEM2",
            "source_confidence": "HIGH_CORE_RULES_CAPITAL_ACCOUNTING_ADAPTED",
            "replication_scope": "SOURCE ENTRY/EXIT/N/UNIT/PYRAMID CORE + LIVE_EQUITY ADAPTATION",
            "source": "Curtis Faith / OriginalTurtles.org Original Turtle Trading Rules (2003), System 2",
            "source_url": "https://studylib.net/doc/28042300/turtle-rules.251218",
            "source_rule": "55D intraday breakout, 20D opposite exit, N=20D EMA True Range, 1N=1% equity Unit, 2N stop, add every 0.5N, max 4 Units",
            "adaptation": "ETH perpetual has no contract roll; daily context uses +8 bars; project causality delays new daily threshold until strictly after 08:00. Unit size uses current live equity rather than the Turtles' historical notional-account process, whose annual reset was discretionary and therefore not mechanically reproducible.",
        },
    ])


def run_source_locked(cfg: SourceLockedConfig | None = None, *, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or SourceLockedConfig()
    _validate(cfg)
    root = Path(cfg.report_root)
    root.mkdir(parents=True, exist_ok=True)
    catalog = _catalog()
    catalog.to_csv(root / "00_source_catalog.csv", index=False)

    print("[run] R03 Source-Locked ETH Trend Replication")
    print(f"[window] warmup={cfg.warmup_start} research={cfg.research_start} -> {cfg.research_end}")
    print(f"[seal] {cfg.sealed_start}+ CLOSED")
    print(f"[cost] round_trip={cfg.round_trip_cost:.4%}")
    print("[principle] source rule first; adaptations disclosed; no parameter mining")

    data = load_data(cfg)
    daily = data.daily()
    z_long = build_zarattini(daily, allow_short=False)
    z_ls = build_zarattini(daily, allow_short=True)
    tsmom = build_mop_tsmom(daily)
    turtle_ctx = build_turtle_context(daily)

    interpretations = pd.DataFrame([
        {"strategy_id": z_long.strategy_id, "source_rule": z_long.source_rule, "adaptation": z_long.adaptation},
        {"strategy_id": z_ls.strategy_id, "source_rule": z_ls.source_rule, "adaptation": z_ls.adaptation},
        {"strategy_id": tsmom.strategy_id, "source_rule": tsmom.source_rule, "adaptation": tsmom.adaptation},
        {"strategy_id": "SL04_TURTLE_SYSTEM2", "source_rule": catalog.iloc[3].source_rule, "adaptation": catalog.iloc[3].adaptation},
    ])
    interpretations.to_csv(root / "07_rule_interpretations.csv", index=False)

    runners = [
        (z_long.strategy_id, lambda cm, delay: run_target_schedule(data.one_minute, z_long.schedule, cfg, z_long.strategy_id, cost_mult=cm, extra_delay_minutes=delay)),
        (z_ls.strategy_id, lambda cm, delay: run_target_schedule(data.one_minute, z_ls.schedule, cfg, z_ls.strategy_id, cost_mult=cm, extra_delay_minutes=delay)),
        (tsmom.strategy_id, lambda cm, delay: run_target_schedule(data.one_minute, tsmom.schedule, cfg, tsmom.strategy_id, cost_mult=cm, extra_delay_minutes=delay)),
        ("SL04_TURTLE_SYSTEM2", lambda cm, delay: run_turtle_system2(data.one_minute, turtle_ctx, cfg, cost_mult=cm, extra_delay_minutes=delay)),
    ]

    summary_rows = []
    yearly_parts = []
    monthly_parts = []
    stress_rows = []
    top_rows = []
    audit_rows = []
    base_results: dict[str, BacktestResult] = {}
    reporter = ProgressReporter("[R03 source-locked]", total=len(runners), every=1, enabled=progress)
    for n, (sid, fn) in enumerate(runners, start=1):
        print(f"[{n}/{len(runners)}] {sid}")
        base = fn(1.0, 0)
        base_results[sid] = base
        summary_rows.append(base.metrics)
        yearly_parts.append(yearly(sid, base.daily))
        monthly_parts.append(monthly(sid, base.daily))
        top_rows.extend(top_days(sid, base.daily))
        audit_rows.append(base.audit)
        for label, cost_mult, delay in (
            ("base", 1.0, 0), ("cost_2x", 2.0, 0), ("cost_3x", 3.0, 0),
            ("delay_plus_1m", 1.0, 1), ("delay_plus_2m", 1.0, 2),
        ):
            r = base if label == "base" else fn(cost_mult, delay)
            stress_rows.append({"strategy_id": sid, "stress": label, **r.metrics})
        d = root / "strategies" / sid
        d.mkdir(parents=True, exist_ok=True)
        base.daily.to_csv(d / "daily_equity.csv")
        base.position.groupby(base.position.index.normalize()).agg(["last", "mean", "min", "max"]).to_csv(d / "daily_exposure.csv")
        base.events.to_csv(d / "events.csv", index=False)
        reporter.update(n)
    reporter.close()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "01_summary.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(root / "02_yearly.csv", index=False)
    pd.concat(monthly_parts, ignore_index=True).to_csv(root / "03_monthly.csv", index=False)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(root / "04_cost_delay_stress.csv", index=False)
    pd.DataFrame(top_rows).to_csv(root / "05_top_day_dependency.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(root / "06_causal_audit.csv", index=False)

    select_rows = []
    for sid, r in base_results.items():
        m = r.metrics
        c2 = stress[(stress.strategy_id == sid) & (stress.stress == "cost_2x")].iloc[0]
        c3 = stress[(stress.strategy_id == sid) & (stress.stress == "cost_3x")].iloc[0]
        pass_gate = (
            m["total_return_pct"] > 0 and m["profit_factor"] > 1.0 and m["max_drawdown_pct"] <= 25.0
            and m["positive_years"] >= 2 and float(c2["total_return_pct"]) > 0
        )
        select_rows.append({"strategy_id": sid, "pass_gate": pass_gate, "cost_2x_return_pct": float(c2["total_return_pct"]), "cost_3x_return_pct": float(c3["total_return_pct"]), **m})
    selection = pd.DataFrame(select_rows)
    eligible = selection[selection.pass_gate].copy()
    ranks: dict[str, int] = {}
    if not eligible.empty:
        order = sorted(eligible.strategy_id.astype(str), key=lambda sid: _rank_key(base_results[sid].metrics))
        ranks = {sid: i + 1 for i, sid in enumerate(order)}
    selection["selection_rank"] = selection.strategy_id.map(ranks)
    selection.to_csv(root / "08_selection.csv", index=False)

    survivors = list(selection.loc[selection.pass_gate, "strategy_id"].astype(str))
    lines = [
        "# R03 Source-Locked ETH Trend Replication — Decision",
        "",
        "This stage deliberately avoids invented portfolio governors and parameter mining.",
        f"- Source-locked replications: **{len(runners)}**.",
        f"- Gate survivors: **{len(survivors)}**.",
        f"- Survivors: **{', '.join(survivors) if survivors else 'NONE'}**.",
        "- 2026 sealed holdout opened: **NO**.",
        "",
        "## Interpretation rule",
        "",
        "A strategy may advance only if its source-locked baseline survives first. Portfolio construction is postponed until this gate is known.",
        "No failed source baseline will be rescued by tweaking lookbacks, stops, volatility targets, or drawdown overlays.",
        "",
        "## Next action",
        "",
        "If >=2 source baselines survive, freeze them and build an equal-risk sleeve portfolio as a separate R04 experiment. If fewer than 2 survive, do not force a portfolio; search for additional complete public strategies.",
    ]
    (root / "99_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack = write_review_pack(root)
    print("[done]", root.resolve())
    print("[review-pack]", pack.resolve())
    return {"report_root": str(root), "review_pack": str(pack), "survivors": survivors}
