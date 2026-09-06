from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import SignalThresholds


def asof_change(
    series: pd.Series,
    minutes: int,
    *,
    tolerance_minutes: float | None = None,
) -> pd.Series:
    """Return current minus the last observation at/before each target time.

    This intentionally avoids interpolation. A window is unavailable when no
    real observation exists close enough to the requested lookback.
    """

    values = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if values.empty:
        return result
    tolerance = tolerance_minutes if tolerance_minutes is not None else max(2.0, minutes * 0.20)
    current = pd.DataFrame(
        {
            "timestamp_utc": values.index,
            "target_utc": values.index - pd.Timedelta(minutes=minutes),
            "current": values.to_numpy(dtype=float),
        }
    ).sort_values("target_utc")
    history = pd.DataFrame(
        {"history_utc": values.index, "previous": values.to_numpy(dtype=float)}
    ).sort_values("history_utc")
    matched = pd.merge_asof(
        current,
        history,
        left_on="target_utc",
        right_on="history_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance),
    )
    changes = matched["current"] - matched["previous"]
    result.loc[matched["timestamp_utc"]] = changes.to_numpy()
    return result


def build_intraday_features(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    result = panel.copy().sort_index()
    definitions = {
        "fedwatch_policy_bias_pct": ((15, 60), 1.0, "fedwatch_bias_change_{minutes}m_pct"),
        "fedwatch_expected_rate_pct": ((15, 60), 100.0, "expected_rate_change_{minutes}m_bp"),
        "us2y_yield_pct": ((5, 15, 60), 100.0, "us2y_change_{minutes}m_bp"),
        "us10y_yield_pct": ((5, 15, 60), 100.0, "us10y_change_{minutes}m_bp"),
        "curve_10y_2y_pct": ((15, 60), 100.0, "curve_change_{minutes}m_bp"),
    }
    for column, (windows, scale, template) in definitions.items():
        if column not in result:
            continue
        for minutes in windows:
            result[template.format(minutes=minutes)] = asof_change(result[column], minutes) * scale
    if "dxy_index" in result:
        for minutes in (5, 15, 60):
            absolute = asof_change(result["dxy_index"], minutes)
            previous = result["dxy_index"] - absolute
            result[f"dxy_change_{minutes}m_pct"] = absolute / previous * 100.0
    return result


def _component(
    row: pd.Series,
    column: str,
    threshold: float,
    *,
    positive_direction: int,
    name: str,
) -> dict[str, object] | None:
    value = row.get(column)
    if value is None or not np.isfinite(value) or abs(float(value)) < float(threshold):
        return None
    raw_sign = 1 if float(value) > 0 else -1
    # direction: +1 dovish, -1 hawkish
    direction = raw_sign * positive_direction
    return {
        "name": name,
        "change": float(value),
        "threshold": float(threshold),
        "standardized": abs(float(value)) / float(threshold),
        "direction": int(direction),
    }


def classify_intraday_row(row: pd.Series, thresholds: SignalThresholds) -> dict[str, object]:
    """Classify repricing. FedWatch/US2Y are primary; others confirm only."""

    components: list[dict[str, object]] = []
    primary: list[dict[str, object]] = []
    threshold_map = asdict(thresholds)
    definitions = (
        ("fedwatch_bias_change_15m_pct", threshold_map["fedwatch_15m_pct"], 1, "FedWatch Bias 15m", True),
        ("fedwatch_bias_change_60m_pct", threshold_map["fedwatch_60m_pct"], 1, "FedWatch Bias 60m", True),
        ("expected_rate_change_15m_bp", threshold_map["expected_rate_15m_bp"], -1, "Expected Rate 15m", False),
        ("expected_rate_change_60m_bp", threshold_map["expected_rate_60m_bp"], -1, "Expected Rate 60m", False),
        ("us2y_change_5m_bp", threshold_map["us2y_5m_bp"], -1, "US2Y 5m", True),
        ("us2y_change_15m_bp", threshold_map["us2y_15m_bp"], -1, "US2Y 15m", True),
        ("us2y_change_60m_bp", threshold_map["us2y_60m_bp"], -1, "US2Y 60m", True),
        ("us10y_change_5m_bp", threshold_map["us10y_5m_bp"], -1, "US10Y 5m", False),
        ("us10y_change_15m_bp", threshold_map["us10y_15m_bp"], -1, "US10Y 15m", False),
        ("us10y_change_60m_bp", threshold_map["us10y_60m_bp"], -1, "US10Y 60m", False),
        ("dxy_change_5m_pct", threshold_map["dxy_5m_pct"], -1, "DXY 5m", False),
        ("dxy_change_15m_pct", threshold_map["dxy_15m_pct"], -1, "DXY 15m", False),
        ("dxy_change_60m_pct", threshold_map["dxy_60m_pct"], -1, "DXY 60m", False),
    )
    for column, threshold, positive_direction, name, is_primary in definitions:
        item = _component(
            row,
            column,
            threshold,
            positive_direction=positive_direction,
            name=name,
        )
        if item:
            components.append(item)
            if is_primary:
                primary.append(item)

    if not primary:
        return {
            "regime": "stable",
            "severity": 0,
            "score": 0.0,
            "primary_count": 0,
            "confirmation_count": 0,
            "drivers": "",
        }
    primary_direction = int(np.sign(sum(float(item["direction"]) * float(item["standardized"]) for item in primary)))
    if primary_direction == 0:
        regime = "mixed"
    else:
        regime = "dovish" if primary_direction > 0 else "hawkish"
    aligned = [item for item in components if item["direction"] == primary_direction]
    opposed = [item for item in components if item["direction"] == -primary_direction]
    score = sum(float(item["standardized"]) for item in aligned) - sum(
        0.5 * float(item["standardized"]) for item in opposed
    )
    distinct_primary = {str(item["name"]).split()[0] for item in primary if item["direction"] == primary_direction}
    distinct_confirmers = {
        str(item["name"]).split()[0]
        for item in aligned
        if str(item["name"]).split()[0] not in distinct_primary
    }
    severity = 1
    if score >= 2.0:
        severity = 2
    if len(distinct_primary) >= 2 or (score >= 3.0 and distinct_confirmers):
        severity = 3
    return {
        "regime": regime,
        "severity": severity,
        "score": float(score),
        "primary_count": len(distinct_primary),
        "confirmation_count": len(distinct_confirmers),
        "drivers": "; ".join(
            f"{item['name']}={item['change']:+.2f}/{item['threshold']:.2f}"
            for item in sorted(aligned, key=lambda x: float(x["standardized"]), reverse=True)
        ),
    }


def select_intraday_events(
    features: pd.DataFrame,
    thresholds: SignalThresholds,
    *,
    cooldown_minutes: int = 30,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for timestamp, row in features.iterrows():
        classification = classify_intraday_row(row, thresholds)
        if classification["regime"] == "stable":
            continue
        payload = {"timestamp_utc": timestamp, **classification}
        for column in (
            "fedwatch_cut_pct",
            "fedwatch_hold_pct",
            "fedwatch_hike_pct",
            "fedwatch_policy_bias_pct",
            "fedwatch_expected_rate_pct",
            "us2y_yield_pct",
            "us10y_yield_pct",
            "curve_10y_2y_pct",
            "dxy_index",
            "fedwatch_bias_change_15m_pct",
            "fedwatch_bias_change_60m_pct",
            "expected_rate_change_15m_bp",
            "expected_rate_change_60m_bp",
            "us2y_change_5m_bp",
            "us2y_change_15m_bp",
            "us2y_change_60m_bp",
        ):
            payload[column] = row.get(column, np.nan)
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows).sort_values("timestamp_utc")
    kept: list[pd.Series] = []
    for _, candidate in candidates.iterrows():
        if not kept:
            kept.append(candidate)
            continue
        previous = kept[-1]
        elapsed = candidate["timestamp_utc"] - previous["timestamp_utc"]
        same_regime = candidate["regime"] == previous["regime"]
        if same_regime and elapsed < pd.Timedelta(minutes=cooldown_minutes):
            # Preserve a later continuation only when severity genuinely
            # breaks through, mirroring the live email cooldown behavior.
            if int(candidate["severity"]) > int(previous["severity"]):
                kept.append(candidate)
            continue
        kept.append(candidate)
    return pd.DataFrame(kept).reset_index(drop=True)


def build_daily_proxy_panel(
    fred_yields: pd.DataFrame,
    dxy_daily: pd.DataFrame,
    fed_funds_futures_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Build a daily proxy panel. This is explicitly not FedWatch history."""

    yields = fred_yields.copy()
    yields.index = pd.DatetimeIndex(yields.index).tz_localize(None).normalize()
    panel = yields[[column for column in ("DGS2", "DGS10") if column in yields]].copy()
    if "DGS2" in panel:
        panel["us2y_change_bp"] = panel["DGS2"].diff() * 100.0
    if "DGS10" in panel:
        panel["us10y_change_bp"] = panel["DGS10"].diff() * 100.0
    if {"DGS2", "DGS10"}.issubset(panel):
        panel["curve_10y_2y_bp"] = (panel["DGS10"] - panel["DGS2"]) * 100.0
        panel["curve_change_bp"] = panel["curve_10y_2y_bp"].diff()

    if not dxy_daily.empty:
        dxy = dxy_daily.copy()
        dxy.index = pd.DatetimeIndex(dxy.index).tz_convert("UTC").tz_localize(None).normalize()
        panel = panel.join(dxy[["close"]].rename(columns={"close": "dxy_close"}), how="outer")
        panel["dxy_change_pct"] = panel["dxy_close"].pct_change(fill_method=None) * 100.0
    if not fed_funds_futures_daily.empty:
        zq = fed_funds_futures_daily.copy()
        zq.index = pd.DatetimeIndex(zq.index).tz_convert("UTC").tz_localize(None).normalize()
        panel = panel.join(zq[["close"]].rename(columns={"close": "zq_front_close"}), how="outer")
        panel["fed_funds_implied_rate_proxy_pct"] = 100.0 - panel["zq_front_close"]
        panel["fed_funds_implied_change_proxy_bp"] = panel["fed_funds_implied_rate_proxy_pct"].diff() * 100.0
    return panel.sort_index()


def select_daily_proxy_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Select daily rate/DXY repricing events from clearly labelled proxies.

    US2Y and front 30-Day Fed Funds Futures implied-rate changes are primary.
    US10Y and DXY can confirm but cannot create an event by themselves.
    """

    if panel.empty:
        return pd.DataFrame()
    definitions = (
        ("fed_funds_implied_change_proxy_bp", 2.0, -1, "ZQ implied rate", True),
        ("us2y_change_bp", 5.0, -1, "US2Y", True),
        ("us10y_change_bp", 5.0, -1, "US10Y", False),
        ("dxy_change_pct", 0.25, -1, "DXY", False),
    )
    output: list[dict[str, object]] = []
    for date, row in panel.iterrows():
        components: list[dict[str, object]] = []
        primary: list[dict[str, object]] = []
        for column, threshold, positive_direction, name, is_primary in definitions:
            item = _component(
                row,
                column,
                threshold,
                positive_direction=positive_direction,
                name=name,
            )
            if item:
                components.append(item)
                if is_primary:
                    primary.append(item)
        if not primary:
            continue
        signed_primary = sum(float(item["direction"]) * float(item["standardized"]) for item in primary)
        direction = int(np.sign(signed_primary))
        if direction == 0:
            regime = "mixed"
        else:
            regime = "dovish" if direction > 0 else "hawkish"
        aligned = [item for item in components if item["direction"] == direction]
        opposed = [item for item in components if item["direction"] == -direction]
        score = sum(float(item["standardized"]) for item in aligned) - sum(
            0.5 * float(item["standardized"]) for item in opposed
        )
        aligned_names = {str(item["name"]) for item in aligned}
        if {"US2Y", "ZQ implied rate"}.issubset(aligned_names):
            primary_driver = "us2y_and_zq_confirmed"
        elif "US2Y" in aligned_names:
            primary_driver = "us2y"
        elif "ZQ implied rate" in aligned_names:
            primary_driver = "zq_front_proxy"
        else:
            primary_driver = "mixed_primary"
        output.append(
            {
                "signal_date": pd.Timestamp(date).normalize(),
                "regime": regime,
                "severity": 3 if len(aligned) >= 3 else (2 if len(aligned) >= 2 or score >= 2 else 1),
                "score": float(score),
                "primary_driver": primary_driver,
                "drivers": "; ".join(
                    f"{item['name']}={item['change']:+.2f}/{item['threshold']:.2f}"
                    for item in sorted(aligned, key=lambda x: float(x["standardized"]), reverse=True)
                ),
                **row.to_dict(),
            }
        )
    if not output:
        return pd.DataFrame()
    return pd.DataFrame(output).set_index("signal_date").sort_index()


def select_daily_static_background(panel: pd.DataFrame, *, min_periods: int = 126) -> pd.DataFrame:
    """Classify a slow-moving level background using trailing-only proxies.

    This is a transparent rates/DXY background, not a historical FedWatch
    stance. A hawkish background requires at least two available inputs above
    their trailing reference; a dovish background requires at least two below.
    """

    if panel.empty:
        return pd.DataFrame()
    work = panel.copy().sort_index()
    votes = pd.DataFrame(index=work.index)
    labels: dict[str, pd.Series] = {}
    if "DGS2" in work:
        reference = work["DGS2"].rolling(252, min_periods=min_periods).median()
        labels["US2Y vs 1y median"] = work["DGS2"] - reference
    if "dxy_close" in work:
        reference = work["dxy_close"].rolling(200, min_periods=min_periods).mean()
        labels["DXY vs 200d mean"] = (work["dxy_close"] / reference - 1.0) * 100.0
    if "fed_funds_implied_rate_proxy_pct" in work:
        reference = work["fed_funds_implied_rate_proxy_pct"].rolling(126, min_periods=min_periods).median()
        labels["ZQ implied vs 6m median"] = work["fed_funds_implied_rate_proxy_pct"] - reference
    for name, values in labels.items():
        votes[name] = np.sign(values)

    output: list[dict[str, object]] = []
    for date, row in votes.iterrows():
        available = row.dropna()
        if len(available) < 2:
            continue
        hawkish_votes = int((available > 0).sum())
        dovish_votes = int((available < 0).sum())
        if hawkish_votes >= 2:
            regime = "hawkish"
            direction = 1
        elif dovish_votes >= 2:
            regime = "dovish"
            direction = -1
        else:
            continue
        drivers = "; ".join(
            f"{name}={float(row[name]):+.3f}" for name in row.index if pd.notna(row[name]) and int(np.sign(row[name])) == direction
        )
        output.append(
            {
                "signal_date": pd.Timestamp(date).normalize(),
                "regime": regime,
                "severity": 2 if max(hawkish_votes, dovish_votes) >= 3 else 1,
                "score": float(max(hawkish_votes, dovish_votes)),
                "drivers": drivers,
                "available_votes": int(len(available)),
                "aligned_votes": int(max(hawkish_votes, dovish_votes)),
            }
        )
    if not output:
        return pd.DataFrame()
    return pd.DataFrame(output).set_index("signal_date").sort_index()
