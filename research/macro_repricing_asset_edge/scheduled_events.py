from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .event_study import summarize_event_returns


UTC = "UTC"
BEIJING = "Asia/Shanghai"


# Curated release anchors already used by CoinBacktest's free macro downloader.
# Beijing time is retained for presentation; all calculations convert to UTC.
EVENTS_2026_BJT: tuple[tuple[str, str], ...] = (
    ("2026-03-19 02:00", "FOMC"),
    ("2026-04-03 20:30", "NFP+Unemployment"),
    ("2026-04-10 20:30", "CPI"),
    ("2026-04-30 02:00", "FOMC"),
    ("2026-04-30 20:30", "Core PCE"),
    ("2026-05-01 22:00", "ISM Manufacturing"),
    ("2026-05-08 20:30", "NFP+Unemployment"),
    ("2026-05-12 20:30", "CPI"),
    ("2026-05-13 20:30", "PPI"),
    ("2026-05-28 20:30", "Core PCE"),
    ("2026-06-05 20:30", "NFP+Unemployment"),
    ("2026-06-10 20:30", "CPI"),
    ("2026-06-11 20:30", "PPI"),
    ("2026-06-17 20:30", "Retail Sales"),
    ("2026-06-18 02:00", "FOMC"),
    ("2026-06-25 20:30", "Core PCE"),
    ("2026-07-01 22:00", "ISM Manufacturing"),
    ("2026-07-02 20:30", "NFP+Unemployment"),
    ("2026-07-14 20:30", "CPI"),
    ("2026-07-15 20:30", "PPI"),
    ("2026-07-16 20:30", "Retail Sales"),
    ("2026-07-30 02:00", "FOMC"),
    ("2026-07-30 20:30", "Core PCE"),
    ("2026-08-03 22:00", "ISM Manufacturing"),
    ("2026-08-07 20:30", "NFP+Unemployment"),
    ("2026-08-12 20:30", "CPI"),
    ("2026-08-13 20:30", "PPI"),
    ("2026-08-14 20:30", "Retail Sales"),
    ("2026-08-26 20:30", "Core PCE"),
)


@dataclass(frozen=True)
class ScheduledThresholds:
    zq_post_fomc_implied_rate_bp: float = 2.0
    zt_tightening_price_bp: float = 4.0
    us2y_yield_bp: float = 3.0
    us10y_yield_bp: float = 3.0
    dxy_pct: float = 0.15


def macro_event_calendar(
    start_utc: str | pd.Timestamp | None = None,
    end_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(EVENTS_2026_BJT, columns=["event_time_bjt", "event_type"])
    frame["event_time_bjt"] = pd.to_datetime(frame["event_time_bjt"]).dt.tz_localize(BEIJING)
    frame["event_time_utc"] = frame["event_time_bjt"].dt.tz_convert(UTC)
    frame["event_time_ny"] = frame["event_time_bjt"].dt.tz_convert("America/New_York")
    frame["event_id"] = (
        frame["event_time_bjt"].dt.strftime("%Y%m%d_%H%M_")
        + frame["event_type"].str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_")
    )
    if start_utc is not None:
        start = pd.Timestamp(start_utc)
        start = start.tz_localize(UTC) if start.tzinfo is None else start.tz_convert(UTC)
        frame = frame.loc[frame["event_time_utc"] >= start]
    if end_utc is not None:
        end = pd.Timestamp(end_utc)
        end = end.tz_localize(UTC) if end.tzinfo is None else end.tz_convert(UTC)
        frame = frame.loc[frame["event_time_utc"] <= end]
    return frame.reset_index(drop=True)


def _close_strictly_before(
    frame: pd.DataFrame,
    target: pd.Timestamp,
    *,
    tolerance_minutes: float,
) -> float:
    if frame.empty or "close" not in frame:
        return float("nan")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("intraday macro bars must have timezone-aware indexes")
    position = int(index.searchsorted(target, side="left")) - 1
    if position < 0:
        return float("nan")
    age = target - index[position]
    if age < pd.Timedelta(0) or age > pd.Timedelta(minutes=tolerance_minutes):
        return float("nan")
    return float(pd.to_numeric(frame.iloc[position]["close"], errors="coerce"))


def _paired_prices(
    frame: pd.DataFrame,
    event_time: pd.Timestamp,
    decision_time: pd.Timestamp,
) -> tuple[float, float]:
    baseline = _close_strictly_before(frame, event_time, tolerance_minutes=20)
    current = _close_strictly_before(frame, decision_time, tolerance_minutes=10)
    return baseline, current


def _finite_change(baseline: float, current: float, *, kind: str) -> float:
    if not np.isfinite(baseline) or not np.isfinite(current) or baseline == 0:
        return float("nan")
    if kind == "yield_bp":
        return (current - baseline) * 100.0
    if kind == "price_return_bp":
        return (current / baseline - 1.0) * 10_000.0
    if kind == "percent":
        return (current / baseline - 1.0) * 100.0
    raise ValueError(f"unsupported change kind: {kind}")


def _classify_proxy_row(
    row: Mapping[str, object],
    thresholds: ScheduledThresholds,
    *,
    threshold_multiplier: float,
) -> dict[str, object]:
    components: list[dict[str, object]] = []

    def add(name: str, column: str, threshold: float, *, primary: bool) -> None:
        raw = row.get(column)
        value = float(raw) if raw is not None else float("nan")
        scaled_threshold = float(threshold) * threshold_multiplier
        if not np.isfinite(value) or abs(value) < scaled_threshold:
            return
        components.append(
            {
                "name": name,
                "value": value,
                "threshold": scaled_threshold,
                "standardized": abs(value) / scaled_threshold,
                "direction": 1 if value > 0 else -1,  # positive = hawkish
                "primary": primary,
            }
        )

    add(
        "ZQ Oct implied rate",
        "zq_post_fomc_implied_rate_change_bp",
        thresholds.zq_post_fomc_implied_rate_bp,
        primary=True,
    )
    exact_us2y_available = np.isfinite(float(row.get("us2y_exact_change_bp", np.nan)))
    add("US2Y exact", "us2y_exact_change_bp", thresholds.us2y_yield_bp, primary=True)
    add(
        "ZT price tightening proxy",
        "zt_tightening_price_proxy_bp",
        thresholds.zt_tightening_price_bp,
        primary=not exact_us2y_available,
    )
    add("US10Y exact", "us10y_exact_change_bp", thresholds.us10y_yield_bp, primary=False)
    if not np.isfinite(float(row.get("us10y_exact_change_bp", np.nan))):
        add("US10Y Yahoo", "us10y_yahoo_change_bp", thresholds.us10y_yield_bp, primary=False)
    add("DXY", "dxy_change_pct", thresholds.dxy_pct, primary=False)

    primary_components = [item for item in components if bool(item["primary"])]
    if not primary_components:
        return {
            "regime": "stable",
            "severity": 0,
            "score": 0.0,
            "primary_count": 0,
            "confirmation_count": 0,
            "drivers": "",
        }
    signed_primary = sum(
        float(item["direction"]) * float(item["standardized"])
        for item in primary_components
    )
    direction = int(np.sign(signed_primary))
    if direction == 0:
        regime = "mixed"
    else:
        regime = "proxy_hawkish" if direction > 0 else "proxy_dovish"
    aligned = [item for item in components if int(item["direction"]) == direction]
    opposed = [item for item in components if int(item["direction"]) == -direction]
    score = sum(float(item["standardized"]) for item in aligned) - 0.5 * sum(
        float(item["standardized"]) for item in opposed
    )
    primary_names = {str(item["name"]) for item in aligned if bool(item["primary"])}
    confirmation_names = {str(item["name"]) for item in aligned if not bool(item["primary"])}
    severity = 1
    if score >= 2.0:
        severity = 2
    if len(primary_names) >= 2 or (score >= 3.0 and confirmation_names):
        severity = 3
    return {
        "regime": regime,
        "severity": severity,
        "score": float(score),
        "primary_count": len(primary_names),
        "confirmation_count": len(confirmation_names),
        "drivers": "; ".join(
            f"{item['name']}={float(item['value']):+.2f}/{float(item['threshold']):.2f}"
            for item in sorted(aligned, key=lambda value: float(value["standardized"]), reverse=True)
        ),
    }


def build_scheduled_macro_signals(
    events: pd.DataFrame,
    macro_bars: Mapping[str, pd.DataFrame],
    signal_delays_minutes: Iterable[int] = (5, 10, 15),
    *,
    thresholds: ScheduledThresholds | None = None,
    threshold_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Create post-release proxy signals using only bars known by decision time."""

    thresholds = thresholds or ScheduledThresholds()
    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        event_time = pd.Timestamp(event["event_time_utc"])
        for delay in signal_delays_minutes:
            decision_time = event_time + pd.Timedelta(minutes=int(delay))
            values: dict[str, object] = {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "event_time_utc": event_time,
                "event_time_bjt": event["event_time_bjt"],
                "signal_delay_minutes": int(delay),
                "timestamp_utc": decision_time,
                "threshold_multiplier": float(threshold_multiplier),
            }

            zq_base, zq_current = _paired_prices(
                macro_bars.get("ZQ_POST_FOMC", pd.DataFrame()), event_time, decision_time
            )
            # Fed Funds futures price is 100 minus implied effective rate.
            values["zq_post_fomc_implied_rate_change_bp"] = -_finite_change(
                zq_base, zq_current, kind="price_return_bp"
            ) * (zq_base / 100.0 if np.isfinite(zq_base) else np.nan)
            values["zq_post_fomc_price"] = zq_current

            zt_base, zt_current = _paired_prices(
                macro_bars.get("ZT", pd.DataFrame()), event_time, decision_time
            )
            values["zt_price_return_bp"] = _finite_change(
                zt_base, zt_current, kind="price_return_bp"
            )
            values["zt_tightening_price_proxy_bp"] = -float(values["zt_price_return_bp"])

            us2_base, us2_current = _paired_prices(
                macro_bars.get("US2Y_EXACT", pd.DataFrame()), event_time, decision_time
            )
            values["us2y_exact_change_bp"] = _finite_change(
                us2_base, us2_current, kind="yield_bp"
            )
            values["us2y_exact_yield_pct"] = us2_current

            us10_base, us10_current = _paired_prices(
                macro_bars.get("US10Y_EXACT", pd.DataFrame()), event_time, decision_time
            )
            values["us10y_exact_change_bp"] = _finite_change(
                us10_base, us10_current, kind="yield_bp"
            )
            values["us10y_exact_yield_pct"] = us10_current

            us10_yahoo_base, us10_yahoo_current = _paired_prices(
                macro_bars.get("US10Y_YAHOO", pd.DataFrame()), event_time, decision_time
            )
            values["us10y_yahoo_change_bp"] = _finite_change(
                us10_yahoo_base, us10_yahoo_current, kind="yield_bp"
            )

            dxy_base, dxy_current = _paired_prices(
                macro_bars.get("DXY", pd.DataFrame()), event_time, decision_time
            )
            values["dxy_change_pct"] = _finite_change(dxy_base, dxy_current, kind="percent")
            values["dxy_index"] = dxy_current
            values.update(
                _classify_proxy_row(
                    values,
                    thresholds,
                    threshold_multiplier=threshold_multiplier,
                )
            )
            rows.append(values)
    return pd.DataFrame(rows).sort_values(["event_time_utc", "signal_delay_minutes"])


def align_scheduled_asset_responses(
    signals: pd.DataFrame,
    asset_bars: Mapping[str, pd.DataFrame],
    *,
    execution_delays_minutes: Iterable[int] = (0, 5, 10),
    horizons_minutes: Iterable[int] = (5, 15, 30, 60),
) -> pd.DataFrame:
    """Enter strictly after the decision plus execution delay; never at event time."""

    output: list[dict[str, object]] = []
    selected = signals.loc[signals["regime"].isin(["proxy_hawkish", "proxy_dovish"])].copy()
    for asset, raw in asset_bars.items():
        if raw.empty:
            continue
        bars = raw.copy().sort_index()
        for column in ("open", "high", "low", "close"):
            bars[column] = pd.to_numeric(bars[column], errors="coerce")
        bars = bars.dropna(subset=["open", "high", "low", "close"])
        index = pd.DatetimeIndex(bars.index)
        if index.tz is None:
            raise ValueError(f"{asset} bars must have timezone-aware indexes")
        for _, signal in selected.iterrows():
            decision_time = pd.Timestamp(signal["timestamp_utc"])
            for execution_delay in execution_delays_minutes:
                entry_target = decision_time + pd.Timedelta(minutes=int(execution_delay))
                entry_position = int(index.searchsorted(entry_target, side="right"))
                if entry_position >= len(bars):
                    continue
                entry_time = index[entry_position]
                if entry_time - entry_target > pd.Timedelta(minutes=10):
                    continue
                entry_price = float(bars.iloc[entry_position]["open"])
                for horizon in horizons_minutes:
                    exit_target = entry_time + pd.Timedelta(minutes=int(horizon))
                    exit_position = int(index.searchsorted(exit_target, side="left")) - 1
                    if exit_position < entry_position:
                        continue
                    if exit_target - index[exit_position] > pd.Timedelta(minutes=10):
                        continue
                    segment = bars.iloc[entry_position : exit_position + 1]
                    exit_price = float(segment.iloc[-1]["close"])
                    gross = (exit_price / entry_price - 1.0) * 100.0
                    output.append(
                        {
                            "asset": asset,
                            "event_id": signal["event_id"],
                            "event_type": signal["event_type"],
                            "event_time_utc": signal["event_time_utc"],
                            "timestamp_utc": decision_time,
                            "regime": signal["regime"],
                            "severity": int(signal["severity"]),
                            "score": float(signal["score"]),
                            "drivers": signal["drivers"],
                            "signal_delay_minutes": int(signal["signal_delay_minutes"]),
                            "execution_delay_minutes": int(execution_delay),
                            "entry_time_utc": entry_time,
                            "entry_price": entry_price,
                            "horizon_minutes": int(horizon),
                            "exit_time_utc": segment.index[-1],
                            "exit_price": exit_price,
                            "forward_return_pct": gross,
                            "net_return_5bp_pct": gross - 0.05,
                            "net_return_10bp_pct": gross - 0.10,
                            "mae_pct": (float(segment["low"].min()) / entry_price - 1.0) * 100.0,
                            "mfe_pct": (float(segment["high"].max()) / entry_price - 1.0) * 100.0,
                        }
                    )
    return pd.DataFrame(output)


def summarize_scheduled_responses(
    responses: pd.DataFrame,
    *,
    return_column: str = "forward_return_pct",
    by_event_type: bool = False,
    bootstrap_samples: int = 2_000,
    random_seed: int = 20260829,
) -> pd.DataFrame:
    if responses.empty:
        return pd.DataFrame()
    grouping = ["signal_delay_minutes", "execution_delay_minutes"]
    if by_event_type:
        grouping.insert(0, "event_type")
    parts: list[pd.DataFrame] = []
    for offset, (keys, group) in enumerate(responses.groupby(grouping, sort=True)):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        summary = summarize_event_returns(
            group,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed + offset * 1_000,
            return_column=return_column,
            cluster_column="event_id",
        )
        if summary.empty:
            continue
        for column, value in reversed(list(zip(grouping, keys_tuple))):
            summary.insert(0, column, value)
        parts.append(summary)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def leave_one_event_out_summary(
    responses: pd.DataFrame,
    *,
    return_column: str = "net_return_5bp_pct",
) -> pd.DataFrame:
    if responses.empty:
        return pd.DataFrame()
    group_columns = [
        "asset",
        "regime",
        "signal_delay_minutes",
        "execution_delay_minutes",
        "horizon_minutes",
    ]
    records: list[dict[str, object]] = []
    for keys, group in responses.groupby(group_columns, sort=True):
        events = list(group["event_id"].dropna().unique())
        if len(events) < 3:
            continue
        full_mean = float(pd.to_numeric(group[return_column], errors="coerce").mean())
        means = []
        for event_id in events:
            remaining = pd.to_numeric(
                group.loc[group["event_id"] != event_id, return_column], errors="coerce"
            ).dropna()
            if len(remaining):
                means.append(float(remaining.mean()))
        if not means:
            continue
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "events": len(events),
                "full_mean_pct": full_mean,
                "loo_min_mean_pct": min(means),
                "loo_max_mean_pct": max(means),
                "same_sign_all_loo": bool(
                    (full_mean > 0 and min(means) > 0) or (full_mean < 0 and max(means) < 0)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def intraday_quality_profile(
    source_frames: Mapping[str, pd.DataFrame],
    *,
    expected_interval_minutes: int = 5,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset, frame in source_frames.items():
        if frame.empty:
            records.append(
                {
                    "dataset": dataset,
                    "rows": 0,
                    "start_utc": None,
                    "end_utc": None,
                    "duplicate_timestamps": 0,
                    "null_close_pct": np.nan,
                    "median_interval_minutes": np.nan,
                    "expected_interval_share_pct": np.nan,
                    "status": "missing",
                }
            )
            continue
        index = pd.DatetimeIndex(frame.index)
        deltas = index.to_series().diff().dt.total_seconds().div(60.0).dropna()
        duplicate_count = int(index.duplicated().sum())
        expected_share = float((deltas.eq(expected_interval_minutes)).mean() * 100.0) if len(deltas) else np.nan
        null_close = float(pd.to_numeric(frame["close"], errors="coerce").isna().mean() * 100.0)
        status = "ok"
        if duplicate_count or null_close > 0:
            status = "review"
        records.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "start_utc": index.min(),
                "end_utc": index.max(),
                "duplicate_timestamps": duplicate_count,
                "null_close_pct": null_close,
                "median_interval_minutes": float(deltas.median()) if len(deltas) else np.nan,
                "expected_interval_share_pct": expected_share,
                "status": status,
            }
        )
    return pd.DataFrame(records)

