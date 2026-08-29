"""Causal path diagnostics and counterfactual replay for ETH dynamic positioning.

This module operates only on already-produced RDPOS-01 report artifacts.  It
never reads future state into a path feature.  Forward returns are diagnostic
labels only.  Counterfactual replays alter target exposure at decision times
using current/past state and then reuse the frozen hourly price/funding path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PathRule:
    strength_threshold: float = 0.25
    extension_threshold: float = 0.50
    minimum_age_hours: int = 12
    strong_share_72h: float = 0.60

    def validate(self) -> None:
        if not 0 < self.strength_threshold < 1:
            raise ValueError("strength_threshold must be in (0, 1)")
        if not -1 <= self.extension_threshold <= 1:
            raise ValueError("extension_threshold must be in [-1, 1]")
        if self.minimum_age_hours <= 0 or self.minimum_age_hours % 4:
            raise ValueError("minimum_age_hours must be a positive multiple of 4")
        if not 0 <= self.strong_share_72h <= 1:
            raise ValueError("strong_share_72h must be in [0, 1]")


REQUIRED_DECISION_COLUMNS = {
    "timestamp", "available_time", "decision_close", "state_ready",
    "medium_trend", "slow_trend", "medium_extension", "slow_extension",
    "medium_location_multiplier", "slow_location_multiplier",
    "medium_desired_close", "slow_desired_close",
}

REQUIRED_EQUITY_COLUMNS = {
    "timestamp", "next_timestamp", "execution_decision", "price_return",
    "funding_rate", "equity", "drawdown", "gross_exposure", "turnover",
}


def validate_report_inputs(decisions: pd.DataFrame, equity: pd.DataFrame) -> None:
    missing_dec = sorted(REQUIRED_DECISION_COLUMNS.difference(decisions.columns))
    missing_eq = sorted(REQUIRED_EQUITY_COLUMNS.difference(equity.columns))
    if missing_dec:
        raise RuntimeError(f"decision_audit missing columns: {missing_dec}")
    if missing_eq:
        raise RuntimeError(f"equity_hourly missing columns: {missing_eq}")
    if decisions.empty or equity.empty:
        raise RuntimeError("RDPOS-01 report inputs are empty")
    if not decisions["available_time"].is_monotonic_increasing:
        raise RuntimeError("decision available_time must be monotonic")
    if not equity["timestamp"].is_monotonic_increasing:
        raise RuntimeError("equity timestamp must be monotonic")


def build_path_table(decisions: pd.DataFrame, rule: PathRule) -> pd.DataFrame:
    """Build path state using current and past decision states only."""
    rule.validate()
    x = decisions.loc[
        decisions["decision_close"].astype(bool) & decisions["state_ready"].astype(bool)
    ].copy()
    x = x.sort_values("available_time", kind="stable").reset_index(drop=True)

    same_sign = np.sign(x["medium_trend"]) == np.sign(x["slow_trend"])
    strong = (
        same_sign
        & (x["medium_trend"].abs() >= rule.strength_threshold)
        & (x["slow_trend"].abs() >= rule.strength_threshold)
    )
    x["state"] = np.select(
        [strong, same_sign], ["STRONG_AGREE", "WEAK_AGREE"], default="DISAGREE"
    )
    x["direction"] = np.where(strong, np.sign(x["medium_trend"]), 0.0)

    run_id = x["state"].ne(x["state"].shift()).cumsum()
    x["state_age_blocks"] = x.groupby(run_id, sort=False).cumcount() + 1
    x["state_age_hours"] = x["state_age_blocks"] * 4

    x["medium_aligned_extension"] = x["direction"] * x["medium_extension"]
    x["slow_aligned_extension"] = x["direction"] * x["slow_extension"]
    x["aligned_extension_mean"] = (
        x["medium_aligned_extension"] + x["slow_aligned_extension"]
    ) / 2.0

    # 18 four-hour decisions = 72 hours.  Current state is allowed because it is
    # already known at decision time; no future row participates.
    x["strong_share_72h"] = (
        x["state"].eq("STRONG_AGREE").astype(float).rolling(18, min_periods=18).mean()
    )
    x["flip_count_72h"] = (
        x["state"].ne(x["state"].shift()).astype(int).rolling(18, min_periods=1).sum()
    )

    x["mature_expansion"] = (
        x["state"].eq("STRONG_AGREE")
        & (x["state_age_hours"] >= rule.minimum_age_hours)
        & (x["aligned_extension_mean"] >= rule.extension_threshold)
        & (x["strong_share_72h"] >= rule.strong_share_72h)
    )

    for sleeve in ("medium", "slow"):
        mult = pd.to_numeric(x[f"{sleeve}_location_multiplier"], errors="coerce")
        desired = pd.to_numeric(x[f"{sleeve}_desired_close"], errors="coerce")
        pre_location = desired / mult.replace(0.0, np.nan)
        pre_location = pre_location.where(np.isfinite(pre_location), desired)
        aligned_extension = x["direction"] * pd.to_numeric(
            x[f"{sleeve}_extension"], errors="coerce"
        )
        reward_mult = (1.0 + 0.25 * aligned_extension).clip(0.50, 1.50)

        x[f"{sleeve}_target_base"] = desired
        # Counterfactual A: stop penalising a mature expansion.  This restores
        # trend+vol sizing but does not exceed it.
        x[f"{sleeve}_target_no_penalty"] = np.where(
            x["mature_expansion"], pre_location, desired
        )
        # Counterfactual B: exploratory symmetric reward using the already-frozen
        # location_strength=0.25 magnitude.  This is not the promotion scenario.
        x[f"{sleeve}_target_reward"] = np.where(
            x["mature_expansion"], pre_location * reward_mult, desired
        )
    return x


def add_forward_market_labels(
    path: pd.DataFrame,
    equity: pd.DataFrame,
    horizons: Iterable[int] = (12, 24, 72),
) -> pd.DataFrame:
    """Attach open-to-open future market returns as diagnostic labels.

    Path features are already formed before this function is called.  These
    forward columns must never be fed back into a positioning decision.
    """
    x = path.copy()
    eq = equity.set_index("timestamp", drop=False)
    price_by_time = eq["open"].astype(float) if "open" in eq.columns else None
    if price_by_time is None:
        # Reconstruct a consistent open index from price_return chain if older
        # report artifacts do not expose open.  Current RDPOS-01 does expose it.
        raise RuntimeError("equity_hourly must contain open for forward labels")

    for horizon in horizons:
        future = price_by_time.shift(-int(horizon))
        future_ret = future / price_by_time - 1.0
        mapped = future_ret.reindex(pd.DatetimeIndex(x["available_time"])).to_numpy()
        x[f"market_return_{horizon}h"] = mapped
        x[f"aligned_return_{horizon}h"] = x["direction"] * mapped
    return x


def path_neighborhood_table(
    decisions: pd.DataFrame,
    equity: pd.DataFrame,
    *,
    strength_values: Iterable[float] = (0.20, 0.25, 0.30),
    extension_values: Iterable[float] = (0.40, 0.50, 0.60),
    age_values: Iterable[int] = (8, 12, 24),
    share_values: Iterable[float] = (0.50, 0.60, 0.70),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strength in strength_values:
        for extension in extension_values:
            for age in age_values:
                for share in share_values:
                    rule = PathRule(strength, extension, age, share)
                    p = add_forward_market_labels(build_path_table(decisions, rule), equity)
                    trade_start = pd.Timestamp(equity["timestamp"].min())
                    q = p[p["mature_expansion"] & (p["available_time"] >= trade_start)].copy()
                    q = q.dropna(subset=["aligned_return_12h", "aligned_return_24h", "aligned_return_72h"])
                    for year, g in q.groupby(q["available_time"].dt.year):
                        rows.append({
                            "strength_threshold": strength,
                            "extension_threshold": extension,
                            "minimum_age_hours": age,
                            "strong_share_72h_threshold": share,
                            "year": int(year),
                            "n": int(len(g)),
                            "aligned_return_12h": float(g["aligned_return_12h"].mean()),
                            "aligned_return_24h": float(g["aligned_return_24h"].mean()),
                            "aligned_return_72h": float(g["aligned_return_72h"].mean()),
                        })
    return pd.DataFrame(rows)


def summarize_neighborhood(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    keys = [
        "strength_threshold", "extension_threshold", "minimum_age_hours",
        "strong_share_72h_threshold",
    ]
    rows: list[dict[str, object]] = []
    for key, g in table.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row.update({
            "years": int(g["year"].nunique()),
            "n": int(g["n"].sum()),
            "all_years_positive_12h": bool((g["aligned_return_12h"] > 0).all()),
            "all_years_positive_24h": bool((g["aligned_return_24h"] > 0).all()),
            "all_years_positive_72h": bool((g["aligned_return_72h"] > 0).all()),
            "worst_year_24h": float(g["aligned_return_24h"].min()),
            "mean_year_24h": float(g["aligned_return_24h"].mean()),
            "worst_year_72h": float(g["aligned_return_72h"].min()),
            "mean_year_72h": float(g["aligned_return_72h"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _step(current: float, desired: float, *, band: float, max_step: float) -> float:
    gap = desired - current
    if abs(gap) < band:
        return float(current)
    return float(current + np.clip(gap, -max_step, max_step))


def _apply_caps(
    medium: float,
    slow: float,
    *,
    sleeve_cap: float,
    gross_cap: float,
    net_cap: float,
) -> tuple[float, float]:
    medium = float(np.clip(medium, -sleeve_cap, sleeve_cap))
    slow = float(np.clip(slow, -sleeve_cap, sleeve_cap))
    gross = abs(medium) + abs(slow)
    if gross > gross_cap and gross > 0:
        scale = gross_cap / gross
        medium *= scale
        slow *= scale
    net = medium + slow
    if abs(net) > net_cap and abs(net) > 0:
        scale = net_cap / abs(net)
        medium *= scale
        slow *= scale
    return medium, slow


def replay_targets(
    path: pd.DataFrame,
    equity: pd.DataFrame,
    config: dict[str, object],
    *,
    target_suffix: str,
    static_scale: float = 1.0,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Replay a counterfactual from frozen hourly price/funding observations.

    The hot path intentionally uses NumPy arrays rather than DataFrame row
    iteration.  This keeps multi-scenario/risk-match research fast without
    changing any timestamp or execution semantics.
    """
    target_cols = [f"medium_target_{target_suffix}", f"slow_target_{target_suffix}"]
    lookup = path.set_index("available_time")[target_cols]
    target_at_hour = lookup.reindex(pd.DatetimeIndex(equity["timestamp"]))
    target_medium = target_at_hour[target_cols[0]].to_numpy(dtype=float) * float(static_scale)
    target_slow = target_at_hour[target_cols[1]].to_numpy(dtype=float) * float(static_scale)

    timestamps = pd.to_datetime(equity["timestamp"]).to_numpy()
    next_timestamps = pd.to_datetime(equity["next_timestamp"]).to_numpy()
    execution = equity["execution_decision"].astype(bool).to_numpy()
    price_returns = pd.to_numeric(equity["price_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    funding_rates = pd.to_numeric(equity["funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    n = len(equity)
    med_pos = np.empty(n, dtype=float)
    slow_pos = np.empty(n, dtype=float)
    net_exposures = np.empty(n, dtype=float)
    gross_exposures = np.empty(n, dtype=float)
    turnovers = np.empty(n, dtype=float)
    gross_returns = np.empty(n, dtype=float)
    funding_returns = np.empty(n, dtype=float)
    trading_costs = np.empty(n, dtype=float)
    net_returns = np.empty(n, dtype=float)
    equities = np.empty(n, dtype=float)
    drawdowns = np.empty(n, dtype=float)

    medium = slow = 0.0
    capital = peak = 1.0
    fee = (
        float(config["fee_rate_per_side"]) + float(config["slippage_rate_per_side"])
    ) * float(cost_multiplier)
    band = float(config["no_trade_band"])
    max_step = float(config["max_step_per_decision"])
    sleeve_cap = float(config["sleeve_notional_cap"])
    gross_cap = float(config["gross_notional_cap"])
    net_cap = float(config["net_notional_cap"])

    for i in range(n):
        if execution[i] and np.isfinite(target_medium[i]) and np.isfinite(target_slow[i]):
            new_medium = _step(medium, target_medium[i], band=band, max_step=max_step)
            new_slow = _step(slow, target_slow[i], band=band, max_step=max_step)
            new_medium, new_slow = _apply_caps(
                new_medium, new_slow, sleeve_cap=sleeve_cap, gross_cap=gross_cap, net_cap=net_cap
            )
        else:
            new_medium, new_slow = medium, slow

        turnover = abs(new_medium - medium) + abs(new_slow - slow)
        net_exposure = new_medium + new_slow
        gross_exposure = abs(new_medium) + abs(new_slow)
        gross_return = net_exposure * price_returns[i]
        funding_return = -net_exposure * funding_rates[i]
        trading_cost = turnover * fee
        net_return = gross_return + funding_return - trading_cost
        capital *= max(0.0, 1.0 + net_return)
        peak = max(peak, capital)

        med_pos[i] = new_medium
        slow_pos[i] = new_slow
        net_exposures[i] = net_exposure
        gross_exposures[i] = gross_exposure
        turnovers[i] = turnover
        gross_returns[i] = gross_return
        funding_returns[i] = funding_return
        trading_costs[i] = trading_cost
        net_returns[i] = net_return
        equities[i] = capital
        drawdowns[i] = capital / peak - 1.0
        medium, slow = new_medium, new_slow

    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "next_timestamp": pd.to_datetime(next_timestamps),
        "medium_position": med_pos,
        "slow_position": slow_pos,
        "net_exposure": net_exposures,
        "gross_exposure": gross_exposures,
        "turnover": turnovers,
        "gross_return": gross_returns,
        "funding_return": funding_returns,
        "trading_cost": trading_costs,
        "net_return": net_returns,
        "equity": equities,
        "drawdown": drawdowns,
    })


def account_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    years = max(
        (frame["next_timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1.0 / 365.25,
    )
    final_equity = float(frame["equity"].iloc[-1])
    cagr = final_equity ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    mdd = float(frame["drawdown"].min())
    return {
        "total_return": final_equity - 1.0,
        "cagr": cagr,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "mean_abs_net_exposure": float(frame["net_exposure"].abs().mean()),
        "annual_turnover": float(frame["turnover"].sum() / years),
        "adjustments_per_day": float((frame["turnover"] > 1e-12).sum() / (len(frame) / 24.0)),
        "total_trading_cost_return": float(frame["trading_cost"].sum()),
        "total_funding_return": float(frame["funding_return"].sum()),
    }


def yearly_summary(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, g in frame.groupby(frame["timestamp"].dt.year):
        curve = (1.0 + g["net_return"]).cumprod()
        dd = curve / curve.cummax() - 1.0
        rows.append({
            "scenario": scenario,
            "year": int(year),
            "return": float(curve.iloc[-1] - 1.0),
            "max_drawdown": float(dd.min()),
            "mean_gross_exposure": float(g["gross_exposure"].mean()),
            "turnover": float(g["turnover"].sum()),
        })
    return pd.DataFrame(rows)


def exposure_matched_base(
    path: pd.DataFrame,
    equity: pd.DataFrame,
    config: dict[str, object],
    *,
    target_mean_gross: float,
    scales: Iterable[float] = tuple(np.round(np.arange(1.00, 1.21, 0.01), 2)),
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    frames: dict[float, pd.DataFrame] = {}
    for scale in scales:
        f = replay_targets(path, equity, config, target_suffix="base", static_scale=float(scale))
        s = account_summary(f)
        rows.append({"scale": float(scale), **s})
        frames[float(scale)] = f
    grid = pd.DataFrame(rows)
    grid["gross_exposure_gap"] = (grid["mean_gross_exposure"] - target_mean_gross).abs()
    selected = grid.sort_values(["gross_exposure_gap", "scale"], kind="stable").iloc[0].to_dict()
    selected["selection_rule"] = "nearest mean gross exposure only; PnL not used"
    return frames[float(selected["scale"])], selected, grid


__all__ = [
    "PathRule", "account_summary", "add_forward_market_labels", "build_path_table",
    "exposure_matched_base", "path_neighborhood_table", "replay_targets",
    "summarize_neighborhood", "validate_report_inputs", "yearly_summary",
]
