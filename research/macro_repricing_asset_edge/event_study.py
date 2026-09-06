from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def _numeric_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().sort_index()
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["open", "high", "low", "close"])


def align_intraday_events(
    events: pd.DataFrame,
    asset_bars: dict[str, pd.DataFrame],
    horizons_minutes: Iterable[int] = (5, 15, 30, 60),
) -> pd.DataFrame:
    """Align alerts to the first bar strictly after each observation."""

    if events.empty:
        return pd.DataFrame()
    output: list[dict[str, object]] = []
    for asset, raw_bars in asset_bars.items():
        if raw_bars.empty:
            continue
        bars = _numeric_ohlc(raw_bars)
        index = pd.DatetimeIndex(bars.index)
        if index.tz is None:
            raise ValueError(f"{asset} bars must have a timezone-aware index")
        for _, event in events.iterrows():
            timestamp = pd.Timestamp(event["timestamp_utc"])
            entry_position = int(index.searchsorted(timestamp, side="right"))
            if entry_position >= len(bars):
                continue
            entry_time = index[entry_position]
            entry_price = float(bars.iloc[entry_position]["open"])
            for horizon in horizons_minutes:
                target = timestamp + pd.Timedelta(minutes=int(horizon))
                exit_position = int(index.searchsorted(target, side="left"))
                if exit_position >= len(bars):
                    continue
                # Reject stale equity bars after a session gap. Crypto remains
                # continuous and naturally passes this bound.
                if index[exit_position] - target > pd.Timedelta(minutes=5):
                    continue
                segment = bars.iloc[entry_position : exit_position + 1]
                if segment.empty:
                    continue
                exit_price = float(segment.iloc[-1]["close"])
                forward_return = (exit_price / entry_price - 1.0) * 100.0
                mae = (float(segment["low"].min()) / entry_price - 1.0) * 100.0
                mfe = (float(segment["high"].max()) / entry_price - 1.0) * 100.0
                output.append(
                    {
                        "asset": asset,
                        "timestamp_utc": timestamp,
                        "timestamp_bjt": timestamp.tz_convert("Asia/Shanghai"),
                        "regime": event["regime"],
                        "severity": int(event["severity"]),
                        "score": float(event["score"]),
                        "drivers": event.get("drivers", ""),
                        "entry_time_utc": entry_time,
                        "entry_price": entry_price,
                        "horizon_minutes": int(horizon),
                        "exit_time_utc": segment.index[-1],
                        "exit_price": exit_price,
                        "forward_return_pct": forward_return,
                        "mae_pct": mae,
                        "mfe_pct": mfe,
                    }
                )
    return pd.DataFrame(output)


def _bootstrap_mean_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    # Resample events, not individual price bars.
    for i in range(samples):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cluster_means(group: pd.DataFrame, value_column: str, cluster_column: str | None) -> np.ndarray:
    values = pd.to_numeric(group[value_column], errors="coerce")
    if not cluster_column or cluster_column not in group:
        return values.dropna().to_numpy(dtype=float)
    work = pd.DataFrame({"value": values, "cluster": group[cluster_column]}).dropna()
    if work.empty:
        return np.array([], dtype=float)
    return work.groupby("cluster", sort=True)["value"].mean().to_numpy(dtype=float)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().clip(0.0, 1.0)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    n = len(ordered)
    adjusted = ordered.to_numpy() * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def summarize_event_returns(
    aligned: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    random_seed: int = 20260829,
    return_column: str = "forward_return_pct",
    cluster_column: str | None = None,
) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    grouped = aligned.groupby(["asset", "regime", "horizon_minutes"], dropna=False)
    for group_number, (key, group) in enumerate(grouped):
        values = pd.to_numeric(group[return_column], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        independent = _cluster_means(group, return_column, cluster_column)
        lower, upper = _bootstrap_mean_interval(independent, bootstrap_samples, random_seed + group_number)
        p_value = float(stats.ttest_1samp(independent, 0.0).pvalue) if len(independent) >= 3 else float("nan")
        records.append(
            {
                "asset": key[0],
                "regime": key[1],
                "horizon_minutes": int(key[2]),
                "events": int(len(values)),
                "independent_clusters": int(len(independent)),
                "mean_return_pct": float(np.mean(values)),
                "median_return_pct": float(np.median(values)),
                "hit_rate_positive_pct": float(np.mean(values > 0.0) * 100.0),
                "mean_mae_pct": float(pd.to_numeric(group["mae_pct"], errors="coerce").mean()),
                "mean_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").mean()),
                "bootstrap_95_low_pct": lower,
                "bootstrap_95_high_pct": upper,
                "t_test_p_value": p_value,
            }
        )
    summary = pd.DataFrame(records)
    if not summary.empty:
        summary["bh_adjusted_p_value"] = benjamini_hochberg(summary["t_test_p_value"])
        summary["evidence_grade"] = np.select(
            [
                summary["independent_clusters"].lt(20),
                summary["bh_adjusted_p_value"].le(0.05) & summary["independent_clusters"].ge(50),
                summary["independent_clusters"].ge(20),
            ],
            ["case-study only", "statistically supported", "preliminary"],
            default="insufficient",
        )
    return summary


def apply_round_trip_costs(
    aligned: pd.DataFrame,
    cost_basis_points: dict[str, float],
) -> pd.DataFrame:
    result = aligned.copy()
    result["round_trip_cost_bp"] = result["asset"].map(cost_basis_points).fillna(10.0)
    result["net_forward_return_pct"] = (
        pd.to_numeric(result["forward_return_pct"], errors="coerce")
        - result["round_trip_cost_bp"] / 100.0
    )
    return result


def assign_chronological_fold(signal_dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(signal_dates, errors="coerce")
    years = dates.dt.year
    return pd.Series(
        np.select(
            [years.le(2022), years.between(2023, 2024), years.ge(2025)],
            ["train_2019_2022", "validation_2023_2024", "test_2025_2026"],
            default="outside_scope",
        ),
        index=signal_dates.index,
        dtype="object",
    )


def summarize_chronological_folds(
    aligned: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    random_seed: int = 20260829,
    return_column: str = "net_forward_return_pct",
    cluster_column: str | None = None,
) -> pd.DataFrame:
    if aligned.empty or "fold" not in aligned:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for offset, (fold, group) in enumerate(aligned.groupby("fold", sort=True)):
        if fold == "outside_scope":
            continue
        summary = summarize_event_returns(
            group,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + offset * 1_000,
            return_column=return_column,
            cluster_column=cluster_column,
        )
        if not summary.empty:
            summary.insert(0, "fold", fold)
            parts.append(summary)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate_daily_bars(frame: pd.DataFrame, *, equity_session: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    bars = _numeric_ohlc(frame)
    if equity_session:
        local = bars.tz_convert("America/New_York")
        local = local.between_time("09:30", "16:00", inclusive="both")
        session_dates = pd.Series(local.index.date, index=local.index)
    else:
        local = bars.tz_convert("UTC")
        session_dates = pd.Series(local.index.date, index=local.index)
    daily = local.groupby(session_dates).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "session_date"
    return daily.dropna().sort_index()


def daily_forward_returns(
    daily_macro: pd.DataFrame,
    daily_assets: dict[str, pd.DataFrame],
    horizons_sessions: Iterable[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Measure returns from the next session open after a daily macro signal."""

    output: list[dict[str, object]] = []
    if daily_macro.empty:
        return pd.DataFrame()
    for asset, raw in daily_assets.items():
        if raw.empty:
            continue
        daily = raw.sort_index()
        dates = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
        for signal_date, macro in daily_macro.iterrows():
            signal = pd.Timestamp(signal_date).tz_localize(None).normalize()
            # Never map a macro observation from before the asset's history to
            # the first available bar. That would repeat the same future day
            # across thousands of signals and manufacture an impossible hit
            # rate. A signal must lie inside the observed asset date range.
            if signal < dates[0] or signal > dates[-1]:
                continue
            entry_position = int(dates.searchsorted(signal, side="right"))
            if entry_position >= len(daily):
                continue
            entry = float(daily.iloc[entry_position]["open"])
            for horizon in horizons_sessions:
                exit_position = entry_position + int(horizon) - 1
                if exit_position >= len(daily):
                    continue
                exit_price = float(daily.iloc[exit_position]["close"])
                segment = daily.iloc[entry_position : exit_position + 1]
                output.append(
                    {
                        "asset": asset,
                        "signal_date": signal,
                        "entry_session": dates[entry_position],
                        "horizon_sessions": int(horizon),
                        "forward_return_pct": (exit_price / entry - 1.0) * 100.0,
                        "mae_pct": (float(segment["low"].min()) / entry - 1.0) * 100.0,
                        "mfe_pct": (float(segment["high"].max()) / entry - 1.0) * 100.0,
                        **macro.to_dict(),
                    }
                )
    return pd.DataFrame(output)
