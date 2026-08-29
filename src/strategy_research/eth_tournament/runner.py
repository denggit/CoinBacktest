from __future__ import annotations

import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

from .catalog import strategy_catalog
from .config import TournamentConfig
from .contracts import BacktestResult, StrategySignals, StrategySpec
from .data import TournamentData, load_base_data
from .engines import run_event_backtest, run_weight_backtest
from .metrics import (
    calculate_metrics,
    monthly_table,
    passes_survivor_gate,
    selection_key,
    top_trade_dependency,
    yearly_table,
)
from .strategies import s01_donchian, s02_ma, s03_bollinger, s05_absorption, s06_cvd, s07_flow_breakout, s08_quarter_hour
from .strategies.s04_turtle import run_turtle_system2


def _validate_window(cfg: TournamentConfig) -> None:
    start = pd.Timestamp(cfg.research_start)
    end = pd.Timestamp(cfg.research_end)
    sealed = pd.Timestamp(cfg.sealed_start)
    if end >= sealed:
        raise ValueError(
            f"Tournament V1 refuses to open sealed data: research_end={end} sealed_start={sealed}. "
            "Keep 2026 sealed until external-rule selection is frozen."
        )
    if start >= end:
        raise ValueError("research_start must be before research_end")


def _event_signals(data: TournamentData, spec: StrategySpec) -> StrategySignals:
    if spec.family_id == "S03":
        return s03_bollinger.build_signals(data, spec)
    if spec.family_id == "S05":
        return s05_absorption.build_signals(data, spec)
    if spec.family_id == "S06":
        return s06_cvd.build_signals(data, spec)
    if spec.family_id == "S07":
        return s07_flow_breakout.build_signals(data, spec)
    if spec.family_id == "S08":
        return s08_quarter_hour.build_signals(data, spec)
    raise KeyError(f"no event builder for {spec.strategy_id}")


def _run_spec(
    data: TournamentData,
    spec: StrategySpec,
    cfg: TournamentConfig,
    *,
    cost_mult: float = 1.0,
    extra_delay_minutes: int = 0,
    prebuilt_signals: StrategySignals | None = None,
) -> BacktestResult:
    if spec.engine == "weight":
        if spec.family_id == "S01":
            target = s01_donchian.build_target(data, spec)
        elif spec.family_id == "S02":
            target = s02_ma.build_target(data, spec)
        else:
            raise KeyError(spec.strategy_id)
        return run_weight_backtest(
            spec.strategy_id,
            data.one_minute,
            target,
            cfg,
            cost_mult=cost_mult,
            extra_delay_minutes=extra_delay_minutes,
        )
    if spec.engine == "turtle":
        return run_turtle_system2(
            data,
            spec,
            cfg,
            cost_mult=cost_mult,
            extra_delay_minutes=extra_delay_minutes,
        )
    signals = prebuilt_signals if prebuilt_signals is not None else _event_signals(data, spec)
    return run_event_backtest(
        spec.strategy_id,
        data.one_minute,
        signals,
        cfg,
        cost_mult=cost_mult,
        extra_delay_minutes=extra_delay_minutes,
    )


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True if df.index.name is not None else False)


def _result_summary_row(spec: StrategySpec, result: BacktestResult) -> dict[str, Any]:
    return {
        "strategy_id": spec.strategy_id,
        "family_id": spec.family_id,
        "family_name": spec.family_name,
        "variant_name": spec.variant_name,
        "source_class": spec.source_class.value,
        **result.metrics,
    }


def _survivor_rows(
    specs: list[StrategySpec],
    base: dict[str, BacktestResult],
    stress_df: pd.DataFrame,
    cfg: TournamentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_id = {s.strategy_id: s for s in specs}
    for sid, result in base.items():
        passed, reason = passes_survivor_gate(
            result.metrics,
            max_mdd_pct=cfg.survivor_max_mdd_pct,
            min_pf=cfg.survivor_min_pf,
            min_positive_years=cfg.survivor_min_positive_years,
            min_trades=cfg.survivor_min_trades,
        )
        cost2 = stress_df[(stress_df["strategy_id"] == sid) & (stress_df["stress"] == "cost_2x")]
        cost2_ret = float(cost2.iloc[0]["total_return_pct"]) if not cost2.empty else float("nan")
        robust = passed and np.isfinite(cost2_ret) and cost2_ret > 0
        robust_reason = reason if not passed else ("PASS" if robust else "2x_cost_return<=0")
        row = {
            "strategy_id": sid,
            "family_id": by_id[sid].family_id,
            "base_pass": bool(passed),
            "robust_survivor": bool(robust),
            "gate_reason": robust_reason,
            "cost_2x_return_pct": cost2_ret,
            **result.metrics,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["selection_key"] = out.apply(
            lambda r: str(
                selection_key(
                    {
                        "max_flat_days": r["max_flat_days"],
                        "max_consecutive_losing_days": r["max_consecutive_losing_days"],
                        "max_drawdown_pct": r["max_drawdown_pct"],
                        "cagr_pct": r["cagr_pct"],
                        "total_return_pct": r["total_return_pct"],
                    }
                )
            ),
            axis=1,
        )
        robust_ids = set(out.loc[out["robust_survivor"], "strategy_id"])
        base_ids = set(out.loc[out["base_pass"], "strategy_id"])
        rank_pool = robust_ids if robust_ids else base_ids
        order = sorted(
            rank_pool,
            key=lambda sid: selection_key(base[sid].metrics),
        )
        rank = {sid: i + 1 for i, sid in enumerate(order)}
        out["selection_rank"] = out["strategy_id"].map(rank)
    return out


def _daily_return_matrix(results: dict[str, BacktestResult]) -> pd.DataFrame:
    cols = {}
    for sid, r in results.items():
        if r.daily_equity.empty:
            continue
        s = r.daily_equity["equity"].astype(float).pct_change().fillna(0.0)
        s.index = pd.to_datetime(s.index)
        cols[sid] = s
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def _portfolio_result(
    selected_ids: list[str],
    base: dict[str, BacktestResult],
    cfg: TournamentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not selected_ids:
        return pd.DataFrame(columns=["equity"]), {"status": "NO_SURVIVORS"}
    matrix = _daily_return_matrix({sid: base[sid] for sid in selected_ids})
    if matrix.empty:
        return pd.DataFrame(columns=["equity"]), {"status": "NO_RETURNS"}
    portfolio_ret = matrix.mean(axis=1, skipna=True).fillna(0.0)
    equity = cfg.initial_capital * (1.0 + portfolio_ret).cumprod()
    de = equity.to_frame("equity")
    combined_trades = pd.concat([base[sid].trades for sid in selected_ids if not base[sid].trades.empty], ignore_index=True) if any(not base[sid].trades.empty for sid in selected_ids) else pd.DataFrame()
    metrics = calculate_metrics(
        combined_trades,
        de,
        initial_capital=cfg.initial_capital,
        start=pd.Timestamp(cfg.research_start),
        end=pd.Timestamp(cfg.research_end),
    )
    metrics.update({"status": "PASS", "strategy_count": len(selected_ids), "strategy_ids": ",".join(selected_ids), "weighting": "equal_daily_return_sleeves"})
    return de, metrics


def _write_review_pack(report_root: Path) -> Path:
    pack = report_root / "gpt_review_pack.zip"
    include_names = [
        "00_strategy_catalog.csv",
        "01_tournament_summary.csv",
        "02_yearly.csv",
        "03_monthly.csv",
        "04_cost_delay_stress.csv",
        "05_top_trade_dependency.csv",
        "06_causal_audit.csv",
        "07_strategy_return_correlation.csv",
        "08_survivors.csv",
        "09_portfolio_summary.csv",
        "99_decision.md",
    ]
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include_names:
            path = report_root / name
            if path.exists():
                zf.write(path, arcname=name)
        strategies_dir = report_root / "strategies"
        if strategies_dir.exists():
            for path in strategies_dir.rglob("summary.json"):
                zf.write(path, arcname=path.relative_to(report_root).as_posix())
    return pack


def run_tournament(cfg: TournamentConfig | None = None, *, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or TournamentConfig()
    _validate_window(cfg)
    report_root = Path(cfg.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    specs = strategy_catalog()
    pd.DataFrame([s.to_dict() for s in specs]).to_csv(report_root / "00_strategy_catalog.csv", index=False)

    print("[run] ETH External Strategy Tournament V1")
    print(f"[window] warmup={cfg.warmup_start} research={cfg.research_start} -> {cfg.research_end}")
    print(f"[seal] 2026 remains closed from {cfg.sealed_start}")
    print(f"[cost] round_trip={cfg.round_trip_cost:.4%} stress=2x/3x")
    print(f"[catalog] {len(specs)} frozen strategy specs across 8 families")

    data = load_base_data(cfg)
    base_results: dict[str, BacktestResult] = {}
    signal_cache: dict[str, StrategySignals] = {}
    stress_rows: list[dict[str, Any]] = []
    yearly_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    top_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    reporter = ProgressReporter("[Strategy Tournament]", total=len(specs), every=1, enabled=progress)
    for n, spec in enumerate(specs, start=1):
        print(f"\n[{n:02d}/{len(specs):02d}] {spec.strategy_id} | {spec.family_name} | {spec.source_class.value}")
        sig = None
        if spec.engine == "event":
            sig = _event_signals(data, spec)
            signal_cache[spec.strategy_id] = sig
            print(f"  signals={len(sig.entries)} exits={len(sig.exits)}")
        base = _run_spec(data, spec, cfg, prebuilt_signals=sig)
        base_results[spec.strategy_id] = base

        strategy_dir = report_root / "strategies" / spec.strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        base.trades.to_csv(strategy_dir / "trades.csv", index=False)
        base.daily_equity.to_csv(strategy_dir / "daily_equity.csv")
        if sig is not None and not sig.audit.empty:
            sig.audit.to_csv(strategy_dir / "signal_audit.csv", index=False)
        with (strategy_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump({"spec": spec.to_dict(), "metrics": base.metrics, "causal_audit": base.causal_audit}, fh, ensure_ascii=False, indent=2, default=str)

        yearly_parts.append(yearly_table(spec.strategy_id, base.daily_equity))
        monthly_parts.append(monthly_table(spec.strategy_id, base.daily_equity))
        audit_rows.append(base.causal_audit)
        for row in top_trade_dependency(base.trades):
            top_rows.append({"strategy_id": spec.strategy_id, **row})

        # Frozen stress battery. No stress result is fed back into rule parameters.
        for label, cost_mult, delay in [
            ("base", 1.0, 0),
            ("cost_2x", 2.0, 0),
            ("cost_3x", 3.0, 0),
            ("delay_plus_1m", 1.0, 1),
            ("delay_plus_2m", 1.0, 2),
        ]:
            r = base if label == "base" else _run_spec(data, spec, cfg, cost_mult=cost_mult, extra_delay_minutes=delay, prebuilt_signals=sig)
            stress_rows.append({"strategy_id": spec.strategy_id, "stress": label, **r.metrics})
        reporter.update(n)
    reporter.close()

    summary = pd.DataFrame([_result_summary_row(s, base_results[s.strategy_id]) for s in specs])
    summary.to_csv(report_root / "01_tournament_summary.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(report_root / "02_yearly.csv", index=False)
    pd.concat(monthly_parts, ignore_index=True).to_csv(report_root / "03_monthly.csv", index=False)
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(report_root / "04_cost_delay_stress.csv", index=False)
    pd.DataFrame(top_rows).to_csv(report_root / "05_top_trade_dependency.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(report_root / "06_causal_audit.csv", index=False)

    ret_matrix = _daily_return_matrix(base_results)
    corr = ret_matrix.corr() if not ret_matrix.empty else pd.DataFrame()
    corr.to_csv(report_root / "07_strategy_return_correlation.csv")
    survivors = _survivor_rows(specs, base_results, stress_df, cfg)
    survivors.to_csv(report_root / "08_survivors.csv", index=False)
    robust = survivors[survivors.get("robust_survivor", False) == True] if not survivors.empty else pd.DataFrame()  # noqa: E712
    if robust.empty and not survivors.empty:
        robust = survivors[survivors.get("base_pass", False) == True]  # noqa: E712
    selected_ids = []
    if not robust.empty:
        ranked = robust.dropna(subset=["selection_rank"]).sort_values("selection_rank")
        # one representative per family first, then fill remaining slots
        seen_family: set[str] = set()
        for _, row in ranked.iterrows():
            if row["family_id"] in seen_family:
                continue
            selected_ids.append(str(row["strategy_id"]))
            seen_family.add(str(row["family_id"]))
            if len(selected_ids) >= cfg.portfolio_max_strategies:
                break
        if len(selected_ids) < cfg.portfolio_max_strategies:
            for sid in ranked["strategy_id"].astype(str):
                if sid not in selected_ids:
                    selected_ids.append(sid)
                if len(selected_ids) >= cfg.portfolio_max_strategies:
                    break
    portfolio_equity, portfolio_metrics = _portfolio_result(selected_ids, base_results, cfg)
    portfolio_equity.to_csv(report_root / "portfolio_daily_equity.csv")
    pd.DataFrame([portfolio_metrics]).to_csv(report_root / "09_portfolio_summary.csv", index=False)

    # Decision is intentionally answer-first and tied to live-strategy goal, not prediction metrics.
    robust_count = int(survivors["robust_survivor"].sum()) if not survivors.empty else 0
    base_count = int(survivors["base_pass"].sum()) if not survivors.empty else 0
    lines = [
        "# ETH External Strategy Tournament V1 — Decision",
        "",
        f"- Frozen external/source-backed specs tested: **{len(specs)}** across **8 families**.",
        f"- Base survivors: **{base_count}**.",
        f"- Robust survivors with positive 2x-cost return: **{robust_count}**.",
        f"- Portfolio sleeves selected: **{', '.join(selected_ids) if selected_ids else 'NONE'}**.",
        "- 2026 sealed holdout opened: **NO**.",
        "",
        "## Interpretation rule",
        "",
        "A source or paper is not treated as proof of live profitability. Only strategies surviving this ETH-specific causal, after-cost tournament can proceed. Failed families are archived rather than parameter-mined.",
        "",
        "## Next action",
        "",
        "If robust survivors exist: perform final frozen-spec 2026 sealed validation before AetherEdge migration. If none exist: do not tweak these parameters against losses; replace failed families with a new batch of externally specified strategies.",
    ]
    (report_root / "99_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack = _write_review_pack(report_root)
    print("\n[done] tournament reports:", report_root.resolve())
    print("[review-pack]", pack.resolve())
    return {
        "report_root": str(report_root),
        "review_pack": str(pack),
        "strategy_count": len(specs),
        "base_survivors": base_count,
        "robust_survivors": robust_count,
        "portfolio_ids": selected_ids,
    }
