from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Observation:
    timestamp_utc: str
    source: str
    metric: str
    meeting_date: str | None
    target_range: str | None
    value: float | None
    status: str = "ok"


@dataclass(frozen=True)
class TargetProbability:
    target_range: str
    probability: float


def target_range_midpoint(target_range: str) -> float | None:
    if "-" not in target_range:
        return None
    try:
        lower, upper = (float(part.strip()) for part in target_range.split("-", 1))
    except ValueError:
        return None
    return (lower + upper) / 2.0


def expected_target_rate(probabilities: Iterable[TargetProbability]) -> float | None:
    weighted_total = 0.0
    probability_total = 0.0
    for item in probabilities:
        midpoint = target_range_midpoint(item.target_range)
        if midpoint is None:
            continue
        weighted_total += midpoint * item.probability
        probability_total += item.probability
    if probability_total <= 0:
        return None
    return weighted_total / probability_total


@dataclass(frozen=True)
class FedWatchSnapshot:
    timestamp_utc: str
    source: str
    meeting_date: str
    probabilities: tuple[TargetProbability, ...]
    cut_probability: float | None
    hold_probability: float | None
    hike_probability: float | None = None
    current_target_range: str | None = None

    @property
    def expected_rate(self) -> float | None:
        return expected_target_rate(self.probabilities)

    def observations(self) -> list[Observation]:
        rows = [
            Observation(
                self.timestamp_utc,
                self.source,
                "fedwatch_target_probability",
                self.meeting_date,
                item.target_range,
                item.probability,
            )
            for item in self.probabilities
        ]
        for metric, value in (
            ("fedwatch_cut_probability", self.cut_probability),
            ("fedwatch_hold_probability", self.hold_probability),
            ("fedwatch_hike_probability", self.hike_probability),
            ("fedwatch_expected_rate", self.expected_rate),
        ):
            if value is not None:
                rows.append(
                    Observation(
                        self.timestamp_utc,
                        self.source,
                        metric,
                        self.meeting_date,
                        None,
                        value,
                    )
                )
        return rows


@dataclass(frozen=True)
class TreasurySnapshot:
    timestamp_utc: str
    source_2y: str
    source_10y: str
    us2y_yield: float | None
    us10y_yield: float | None

    @property
    def spread(self) -> float | None:
        if self.us2y_yield is None or self.us10y_yield is None:
            return None
        return self.us10y_yield - self.us2y_yield

    def observations(self) -> list[Observation]:
        rows: list[Observation] = []
        if self.us2y_yield is not None:
            rows.append(Observation(self.timestamp_utc, self.source_2y, "us2y_yield", None, None, self.us2y_yield))
        if self.us10y_yield is not None:
            rows.append(Observation(self.timestamp_utc, self.source_10y, "us10y_yield", None, None, self.us10y_yield))
        if self.spread is not None:
            rows.append(Observation(
                self.timestamp_utc,
                f"{self.source_10y}-{self.source_2y}",
                "us10y_2y_spread",
                None,
                None,
                self.spread,
            ))
        return rows


@dataclass(frozen=True)
class DxySnapshot:
    timestamp_utc: str
    source: str
    value: float | None

    def observations(self) -> list[Observation]:
        if self.value is None:
            return []
        return [Observation(self.timestamp_utc, self.source, "dxy_index", None, None, self.value)]


def unavailable_observation(source: str, metric: str, timestamp_utc: str | None = None) -> Observation:
    return Observation(timestamp_utc or utc_now_iso(), source, metric, None, None, None, "unavailable")


def observation_values(rows: Iterable[Observation]) -> dict[str, float]:
    return {row.metric: row.value for row in rows if row.value is not None}
