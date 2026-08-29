from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import Thresholds


def bp_change(yield_now: float, yield_previous: float) -> float:
    return (yield_now - yield_previous) * 100.0


def threshold_triggered(change: float, threshold: float) -> bool:
    if threshold <= 0:
        raise ValueError("Alert threshold must be positive")
    return abs(change) >= threshold


def severity_for(change: float, threshold: float) -> int:
    ratio = abs(change) / threshold
    if ratio >= 2.0:
        return 3
    if ratio >= 1.5:
        return 2
    return 1


@dataclass(frozen=True)
class Alert:
    key: str
    title: str
    detail: str
    direction: str
    severity: int
    window_minutes: int


@dataclass(frozen=True)
class FedWatchState:
    cut_probability: float | None
    hold_probability: float | None
    hike_probability: float | None
    expected_rate: float | None


@dataclass(frozen=True)
class FedWatchComparison:
    previous: FedWatchState
    current: FedWatchState
    direction: str
    signed_change_pct: float
    driver_metric: str
    driver_change_pct: float
    expected_rate_change_bp: float | None


def compare_fedwatch(previous: FedWatchState, current: FedWatchState) -> FedWatchComparison | None:
    if (
        previous.cut_probability is None
        or previous.hike_probability is None
        or current.cut_probability is None
        or current.hike_probability is None
    ):
        return None
    cut_change = current.cut_probability - previous.cut_probability
    hike_change = current.hike_probability - previous.hike_probability
    bias_change = cut_change - hike_change
    if abs(bias_change) < 1e-12:
        return None
    direction = "dovish" if bias_change > 0 else "hawkish"
    candidates = (
        (("CUT", cut_change), ("HIKE", hike_change))
        if direction == "dovish"
        else (("HIKE", hike_change), ("CUT", cut_change))
    )
    relevant = [
        (metric, change)
        for metric, change in candidates
        if (direction == "dovish" and ((metric == "CUT" and change > 0) or (metric == "HIKE" and change < 0)))
        or (direction == "hawkish" and ((metric == "HIKE" and change > 0) or (metric == "CUT" and change < 0)))
    ]
    driver_metric, driver_change = max(relevant, key=lambda item: abs(item[1]))
    expected_change = None
    if previous.expected_rate is not None and current.expected_rate is not None:
        expected_change = (current.expected_rate - previous.expected_rate) * 100.0
    return FedWatchComparison(
        previous=previous,
        current=current,
        direction=direction,
        signed_change_pct=abs(bias_change) if direction == "dovish" else -abs(bias_change),
        driver_metric=driver_metric,
        driver_change_pct=driver_change,
        expected_rate_change_bp=expected_change,
    )


def _probability_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def fedwatch_comparison_detail(comparison: FedWatchComparison, minutes: int) -> str:
    old = comparison.previous
    new = comparison.current
    driver_old = old.hike_probability if comparison.driver_metric == "HIKE" else old.cut_probability
    driver_new = new.hike_probability if comparison.driver_metric == "HIKE" else new.cut_probability
    lines = [
        (
            f"Driver {comparison.driver_metric.title()}: {_probability_text(driver_old)} -> "
            f"{_probability_text(driver_new)} ({comparison.driver_change_pct:+.1f} pct, {minutes}m)"
        ),
        (
            "Cut / Hold / Hike: "
            f"{_probability_text(old.cut_probability)} / {_probability_text(old.hold_probability)} / {_probability_text(old.hike_probability)}"
            " -> "
            f"{_probability_text(new.cut_probability)} / {_probability_text(new.hold_probability)} / {_probability_text(new.hike_probability)}"
        ),
    ]
    if old.expected_rate is not None and new.expected_rate is not None:
        lines.append(f"Expected Rate: {old.expected_rate:.3f}% -> {new.expected_rate:.3f}%")
        lines.append(f"ΔExpected Rate: {comparison.expected_rate_change_bp:+.1f} bp")
    else:
        lines.append("ΔExpected Rate: n/a")
    return "\n".join(lines)


def fedwatch_repricing_alerts(
    current: FedWatchState,
    previous: Mapping[int, FedWatchState],
    thresholds: Thresholds,
) -> list[Alert]:
    results: list[Alert] = []
    limits = {15: thresholds.fedwatch_15m_pct, 60: thresholds.fedwatch_60m_pct}
    for minutes, old in previous.items():
        comparison = compare_fedwatch(old, current)
        if comparison is None:
            continue
        limit = limits[minutes]
        if not threshold_triggered(comparison.signed_change_pct, limit):
            continue
        driver_direction = "UP" if comparison.driver_change_pct > 0 else "DOWN"
        results.append(
            Alert(
                key=f"FEDWATCH_{comparison.driver_metric}_{minutes}M_{driver_direction}",
                title=(
                    f"FedWatch {comparison.direction} repricing: "
                    f"{comparison.driver_metric.title()} {driver_direction.lower()}"
                ),
                detail=fedwatch_comparison_detail(comparison, minutes),
                direction=comparison.direction,
                severity=severity_for(comparison.signed_change_pct, limit),
                window_minutes=minutes,
            )
        )
    return results


def classify_macro(
    *,
    fedwatch_change_pct: float | None,
    fedwatch_threshold_pct: float,
    us2y_change_bp: float | None,
    us2y_threshold_bp: float,
) -> str | None:
    directions: list[str] = []
    if fedwatch_change_pct is not None and threshold_triggered(fedwatch_change_pct, fedwatch_threshold_pct):
        directions.append("dovish" if fedwatch_change_pct > 0 else "hawkish")
    if us2y_change_bp is not None and threshold_triggered(us2y_change_bp, us2y_threshold_bp):
        directions.append("dovish" if us2y_change_bp < 0 else "hawkish")
    if not directions:
        return None
    if len(directions) >= 2 and len(set(directions)) == 1:
        return f"STRONG {directions[0].upper()} REPRICING"
    if len(set(directions)) > 1:
        return "MIXED MACRO REPRICING"
    return f"{directions[0].upper()} REPRICING"


def yield_alerts(metric: str, current: float, previous: Mapping[int, float], thresholds: Thresholds) -> list[Alert]:
    if metric == "us2y_yield":
        label = "US2Y"
        limits = {5: thresholds.us2y_5m_bp, 15: thresholds.us2y_15m_bp, 60: thresholds.us2y_60m_bp}
    elif metric == "us10y_yield":
        label = "US10Y"
        limits = {5: thresholds.us10y_5m_bp, 15: thresholds.us10y_15m_bp, 60: thresholds.us10y_60m_bp}
    else:
        raise ValueError(f"Unsupported yield metric: {metric}")
    results: list[Alert] = []
    for minutes, old in previous.items():
        change = bp_change(current, old)
        limit = limits[minutes]
        if not threshold_triggered(change, limit):
            continue
        direction = "UP" if change > 0 else "DOWN"
        results.append(
            Alert(
                key=f"{label}_{minutes}M_{direction}",
                title=f"{label} yield {direction.lower()}",
                detail=f"{old:.3f}% -> {current:.3f}% ({change:+.1f} bp, {minutes}m)",
                direction=("dovish" if change < 0 else "hawkish") if metric == "us2y_yield" else direction.lower(),
                severity=severity_for(change, limit),
                window_minutes=minutes,
            )
        )
    return results
