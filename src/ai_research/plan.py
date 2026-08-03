#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical gated stage plan for the ETH AI trading programme."""

from __future__ import annotations

from collections import defaultdict, deque

from .config import DEFAULT_AI_RESEARCH_CONFIG, DEFAULT_PLAN_DOC
from .models import ResearchPlan, StageDefinition


DEFAULT_RESEARCH_PLAN = ResearchPlan(
    plan_id="ETH_AI_TRADING",
    title="ETH AI Trading Research-to-Live Programme",
    version=4,
    config=DEFAULT_AI_RESEARCH_CONFIG,
    plan_doc=DEFAULT_PLAN_DOC.relative_to(DEFAULT_PLAN_DOC.parents[1]).as_posix(),
    stages=(
        StageDefinition(
            stage_id="R00",
            name="Framework and governance",
            owner="coinbacktest",
            goal="Freeze project boundaries, causal rules, research windows, costs, artifacts, and stop conditions.",
            ai_methods=("ai_assisted_research",),
            deliverables=("versioned stage plan", "research charter", "plan-to-doc tests"),
            acceptance_gates=("stage graph validates", "CoinBacktest and AetherEdge boundaries are explicit"),
            stop_conditions=("research and live exchange execution are mixed",),
        ),
        StageDefinition(
            stage_id="R01",
            name="Short-horizon trades-only diagnostic baseline",
            owner="coinbacktest",
            goal="Measure whether 1s Trade Flow alone predicts several-minute moves after real market-order costs, and archive the result without treating fixed-time exits as the final bot.",
            depends_on=("R00",),
            ai_methods=("supervised_learning", "signal_scoring", "ai_assisted_research"),
            deliverables=("causal 1s cache", "Ridge/LightGBM baseline", "cost and delay matrix", "MFE/MAE diagnostic"),
            acceptance_gates=("all data uses src.data_feed", "results are interpreted as a short-horizon diagnostic"),
            stop_conditions=("a larger model is used only to hide a failed simple baseline",),
        ),
        StageDefinition(
            stage_id="R02",
            name="Three-sleeve framework and contracts",
            owner="coinbacktest",
            goal="Freeze independent Short, Intraday, and Swing research contracts that share one candidate and target-position schema but keep labels, training, exits, and gates separate.",
            depends_on=("R01",),
            ai_methods=("signal_scoring", "risk_and_portfolio_management"),
            deliverables=("sleeve registry", "TradeCandidate contract", "TargetPositionDecision contract", "updated phase plan"),
            acceptance_gates=("three sleeves validate", "non-short sleeves use time only as a safety cap", "no sleeve directly places exchange orders"),
            stop_conditions=("one model is forced to learn minutes and days simultaneously",),
        ),
        StageDefinition(
            stage_id="R03",
            name="Swing long-context 3%-5% direction and entry MVP",
            owner="coinbacktest",
            goal="Use up to 365D daily, 120D 4H, and 30D 1H causal context plus lower-timeframe entry context to find 3%-5% moves before a bounded adverse path; holding duration is an outcome, not a minimum requirement.",
            depends_on=("R02",),
            ai_methods=("supervised_learning", "market_state_recognition", "signal_scoring"),
            deliverables=("exact target-before-adverse labels", "long-context causal feature profile", "direction-specific entry models", "target-centric trade replay", "one-sided MVP contract when justified"),
            acceptance_gates=("2024 supports the 2025 validation candidate", "2025 survives 2x cost and 5m delay", "locked 2026 confirms the selected candidate"),
            stop_conditions=("profit requires a few tail trades", "changing exits is used to rescue a model with no entry edge"),
        ),
        StageDefinition(
            stage_id="R04",
            name="Intraday trend sleeve",
            owner="coinbacktest",
            goal="Model 1-12h trends and 1%-2.5% moves using 4H/1H/30m direction and 30m/15m/5m/1m entries.",
            depends_on=("R03",),
            ai_methods=("supervised_learning", "market_state_recognition", "signal_scoring"),
            deliverables=("intraday direction model", "entry timing model", "state/structure exits"),
            acceptance_gates=("real-cost validation is positive", "frequency and MAE complement Swing"),
            stop_conditions=("it duplicates Swing with no frequency or risk benefit",),
        ),
        StageDefinition(
            stage_id="R05",
            name="Short-horizon sleeve redesign",
            owner="coinbacktest",
            goal="Rebuild the 5-60m sleeve around target-hit probability, low MAE, MFE protection, and dynamic exits rather than fixed 1/3/5/15m holding.",
            depends_on=("R04",),
            ai_methods=("supervised_learning", "market_state_recognition", "signal_scoring"),
            deliverables=("TP-before-risk labels", "bad-trade model", "dynamic micro exit", "short-horizon cost stress"),
            acceptance_gates=("0.3%-0.8% opportunities cover costs", "tail losses are controlled"),
            stop_conditions=("turnover and costs consume the predicted edge",),
        ),
        StageDefinition(
            stage_id="R06",
            name="Unified multi-sleeve decision layer",
            owner="coinbacktest",
            goal="Convert all sleeve candidates into one ETH target net position with direction hierarchy, duplicate-edge discounts, and hard-risk vetoes.",
            depends_on=("R05",),
            ai_methods=("signal_scoring", "risk_and_portfolio_management"),
            deliverables=("single target-position decision", "sleeve conflict policy", "risk-weighted contribution audit"),
            acceptance_gates=("no independent sleeve orders", "combined result is not hidden leverage"),
            stop_conditions=("sleeves fight each other or simply add exposure",),
        ),
        StageDefinition(
            stage_id="R07",
            name="Sequence-model challengers",
            owner="coinbacktest",
            goal="Test compact sequence models only against frozen tabular baselines inside each suitable sleeve.",
            depends_on=("R06",),
            ai_methods=("deep_sequence_learning", "supervised_learning"),
            deliverables=("TCN challengers", "tabular/sequence/fusion comparison", "CPU latency benchmark"),
            acceptance_gates=("trading metrics improve across seeds", "inference fits the live budget"),
            stop_conditions=("deep learning adds complexity without net trading gain",),
        ),
        StageDefinition(
            stage_id="R08",
            name="Incremental data ablation",
            owner="coinbacktest",
            goal="Add OI, Books, Range, Footprint, liquidity, and prior research evidence one source at a time to the relevant sleeve.",
            depends_on=("R07",),
            ai_methods=("supervised_learning", "market_state_recognition", "signal_scoring"),
            deliverables=("source ablation matrix", "coverage disclosure", "feature latency budget"),
            acceptance_gates=("every retained source improves locked net performance or risk"),
            stop_conditions=("features are retained because they are interesting rather than useful",),
        ),
        StageDefinition(
            stage_id="R09",
            name="Portfolio and risk management",
            owner="coinbacktest",
            goal="Freeze risk budgets, drawdown governors, model drift controls, maximum ETH exposure, and deterministic kill switches.",
            depends_on=("R08",),
            ai_methods=("risk_and_portfolio_management", "signal_scoring"),
            deliverables=("risk budget engine", "drawdown and drift stress", "kill-switch contract"),
            acceptance_gates=("MDD remains inside limits", "2x costs remain profitable"),
            stop_conditions=("risk improvement is only lower reported exposure without realistic sizing",),
        ),
        StageDefinition(
            stage_id="R10",
            name="Execution optimisation",
            owner="coinbacktest",
            goal="Challenge the conservative market-order baseline with timing, split execution, and maker/taker choices only after directional edge exists.",
            depends_on=("R09",),
            ai_methods=("execution_optimisation",),
            deliverables=("delay simulation", "split-order challenger", "conservative fill report"),
            acceptance_gates=("execution adds net value after missed fills and slippage"),
            stop_conditions=("profit depends on optimistic queue assumptions",),
        ),
        StageDefinition(
            stage_id="R11",
            name="Constrained reinforcement-learning exit overlay",
            owner="coinbacktest",
            goal="Allow RL only to choose hold, reduce, or exit within deterministic risk limits after supervised edges are proven.",
            depends_on=("R10",),
            ai_methods=("reinforcement_learning", "risk_and_portfolio_management"),
            deliverables=("restricted action contract", "RL versus deterministic exit", "policy stability report"),
            acceptance_gates=("RL beats deterministic exits on locked data", "hard risk cannot be bypassed"),
            stop_conditions=("RL is needed to invent the original direction edge", "policy exploits the simulator"),
        ),
        StageDefinition(
            stage_id="R12",
            name="Model package and replay parity",
            owner="cross_project",
            goal="Export versioned models and prove CoinBacktest-to-AetherEdge bar, feature, prediction, and decision parity.",
            depends_on=("R11",),
            ai_methods=("ai_assisted_research", "risk_and_portfolio_management"),
            deliverables=("model manifest", "feature schema", "golden replay vectors", "fail-closed validation"),
            acceptance_gates=("same market data produces the same decisions", "AetherEdge does not import research code"),
            stop_conditions=("offline and online features cannot be reconciled",),
        ),
        StageDefinition(
            stage_id="R13",
            name="AetherEdge shadow deployment",
            owner="aetheredge",
            goal="Run real-time ingestion and inference without orders and reconcile every shadow decision with offline replay.",
            depends_on=("R12",),
            ai_methods=("execution_optimisation", "risk_and_portfolio_management"),
            deliverables=("shadow decisions", "latency telemetry", "parity and drift report"),
            acceptance_gates=("no unexplained drift", "runtime failures default to no new position"),
            stop_conditions=("shadow decisions cannot reproduce offline decisions",),
        ),
        StageDefinition(
            stage_id="R14",
            name="Small-capital live acceptance",
            owner="aetheredge",
            goal="Validate the complete system with tightly capped capital before any portfolio-scale promotion.",
            depends_on=("R13",),
            ai_methods=("risk_and_portfolio_management", "execution_optimisation"),
            deliverables=("small-capital live run", "realised cost report", "incident report", "promotion decision"),
            acceptance_gates=("real fills remain inside tested assumptions", "enough independent trades support promotion"),
            stop_conditions=("live slippage, drift, or failures invalidate the backtest",),
        ),
    ),
)


def validate_research_plan(plan: ResearchPlan = DEFAULT_RESEARCH_PLAN) -> None:
    """Validate ordering, dependency existence, and acyclic stage topology."""
    plan.config.validate()
    if plan.version <= 0:
        raise ValueError("plan version must be positive")
    if not plan.stages:
        raise ValueError("research plan must contain at least one stage")

    ids = [stage.stage_id for stage in plan.stages]
    if len(ids) != len(set(ids)):
        raise ValueError("stage ids must be unique")
    positions = {stage_id: index for index, stage_id in enumerate(ids)}
    for stage in plan.stages:
        if not stage.goal or not stage.acceptance_gates:
            raise ValueError(f"{stage.stage_id} must define a goal and acceptance gates")
        for dependency in stage.depends_on:
            if dependency not in positions:
                raise ValueError(f"{stage.stage_id} depends on missing stage {dependency}")
            if positions[dependency] >= positions[stage.stage_id]:
                raise ValueError(f"{stage.stage_id} dependency {dependency} must appear earlier")

    graph: dict[str, list[str]] = defaultdict(list)
    indegree = {stage_id: 0 for stage_id in ids}
    for stage in plan.stages:
        for dependency in stage.depends_on:
            graph[dependency].append(stage.stage_id)
            indegree[stage.stage_id] += 1
    queue = deque(stage_id for stage_id in ids if indegree[stage_id] == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in graph[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(ids):
        raise ValueError("stage dependencies contain a cycle")
