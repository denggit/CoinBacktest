#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature/context builders for ETH MF Low Sweep."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import range_code
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader, price_step_code
from src.edge_lib.mf_low_sweep.events import build_canonical_events, build_features, build_low_sweep_events, parse_number_list


def split_csv_names(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def safe_num(s: pd.Series | object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(s, index=index, dtype="float64")


def to_float_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def timeframe_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("H"):
        return int(tf[:-1]) * 60
    if tf.endswith("D"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe for rolling-day conversion: {timeframe}")


def rolling_window_bars(args: Any, days: float) -> int:
    minutes = timeframe_to_minutes(str(args.timeframe))
    return max(1, int(round(float(days) * 1440.0 / minutes)))


def tag_number(value: float, *, scale: int = 100) -> str:
    return f"{int(round(float(value) * scale)):0{4 if scale == 10000 else 2}d}"


def fixed_threshold_labels(series: pd.Series, *, thresholds: Sequence[float], labels: Sequence[str]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(labels[-1], index=s.index, dtype="object")
    prev = -np.inf
    for threshold, label in zip(thresholds, labels):
        out.loc[(s > prev) & (s <= float(threshold))] = label
        prev = float(threshold)
    out.loc[s.isna()] = "NA"
    return out


def _add_fixed_threshold_flags(out: pd.DataFrame, args: Any) -> None:
    specs = [
        ("taker_buy_ratio", "buy_ratio", parse_number_list(args.buy_ratio_thresholds, cast=float, name="buy_ratio_thresholds"), 100),
        ("buy_pressure", "buy_pressure", parse_number_list(args.buy_pressure_thresholds, cast=float, name="buy_pressure_thresholds"), 100),
        ("delta_pressure", "delta_pressure", parse_number_list(args.delta_pressure_thresholds, cast=float, name="delta_pressure_thresholds", allow_zero=True), 100),
    ]
    for col, prefix, thresholds, scale in specs:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for threshold in thresholds:
            out[f"{prefix}_ge_{tag_number(float(threshold), scale=scale)}"] = series >= float(threshold)

    for w in parse_number_list(args.cvd_windows, cast=int, name="cvd_windows"):
        col = f"cvd_pressure_{int(w)}"
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for threshold in parse_number_list(args.cvd_pressure_thresholds, cast=float, name="cvd_pressure_thresholds", allow_zero=True):
            out[f"{col}_ge_{tag_number(float(threshold), scale=10000)}"] = series >= float(threshold)


def _add_historical_rolling_quantile_flags(out: pd.DataFrame, args: Any) -> pd.DataFrame:
    days_list = parse_number_list(args.rolling_quantile_days, cast=float, name="rolling_quantile_days")
    quantiles = parse_number_list(args.rolling_quantiles, cast=float, name="rolling_quantiles")
    source_cols = [
        ("atr_pct", "atr"),
        ("volume_ratio", "volume"),
        ("trades_count_ratio", "trades"),
        ("taker_buy_ratio", "buy_ratio"),
        ("buy_pressure", "buy_pressure"),
        ("delta_pressure", "delta_pressure"),
        ("large_trade_share", "large_share"),
        ("large_delta_pressure", "large_delta"),
        ("cvd_pressure_15", "cvdp15"),
        ("cvd_pressure_60", "cvdp60"),
    ]
    new_cols: dict[str, pd.Series] = {}
    for col, prefix in source_cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for days in days_list:
            window = rolling_window_bars(args, float(days))
            min_periods = min(window, max(100, window // 3))
            if window < 10:
                min_periods = max(3, window)
            for q in quantiles:
                q = float(q)
                if not 0.0 < q < 1.0:
                    continue
                threshold = series.shift(1).rolling(window, min_periods=min_periods).quantile(q)
                day_tag = str(int(days)) if float(days).is_integer() else str(days).replace(".", "p")
                q_tag = tag_number(q, scale=100)
                base = f"{prefix}_rq{q_tag}_{day_tag}d"
                new_cols[f"{base}_threshold"] = threshold
                new_cols[base] = series >= threshold
    if not new_cols:
        return out
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def build_enriched_features(bars: pd.DataFrame, args: Any) -> pd.DataFrame:
    out = build_features(bars, args).copy()

    buy_notional = to_float_series(out, "buy_notional")
    sell_notional = to_float_series(out, "sell_notional")
    total_notional = (buy_notional + sell_notional).replace(0.0, np.nan)
    out["total_notional"] = total_notional
    out["buy_pressure"] = buy_notional / total_notional
    out["sell_pressure"] = sell_notional / total_notional
    out["notional_imbalance"] = (buy_notional - sell_notional) / total_notional

    buy_volume = to_float_series(out, "buy_volume")
    sell_volume = to_float_series(out, "sell_volume")
    total_volume = (buy_volume + sell_volume).replace(0.0, np.nan)
    out["buy_volume_pressure"] = buy_volume / total_volume
    out["volume_imbalance"] = (buy_volume - sell_volume) / total_volume

    buy_trades = to_float_series(out, "buy_trades_count")
    sell_trades = to_float_series(out, "sell_trades_count")
    total_trades = (buy_trades + sell_trades).replace(0.0, np.nan)
    out["buy_trade_count_pressure"] = buy_trades / total_trades
    out["trade_count_imbalance"] = (buy_trades - sell_trades) / total_trades

    large_buy = to_float_series(out, "large_buy_notional", 0.0)
    large_sell = to_float_series(out, "large_sell_notional", 0.0)
    large_delta = to_float_series(out, "large_delta_notional", 0.0)
    out["large_trade_share"] = (large_buy + large_sell) / total_notional
    out["large_delta_pressure"] = large_delta / total_notional

    delta_notional = to_float_series(out, "delta_notional")
    out["delta_pressure"] = delta_notional / total_notional

    cvd = to_float_series(out, "cvd_notional")
    for w in parse_number_list(args.cvd_windows, cast=int, name="cvd_windows"):
        w = int(w)
        delta_col = f"cvd_delta_{w}"
        pressure_col = f"cvd_pressure_{w}"
        z_col = f"cvd_delta_z_{w}"
        out[delta_col] = cvd - cvd.shift(w)
        notional_sum = total_notional.rolling(w, min_periods=max(3, min(w, max(3, w // 3)))).sum().replace(0.0, np.nan)
        out[pressure_col] = out[delta_col] / notional_sum
        hist = out[delta_col].shift(1).rolling(max(int(args.cvd_window), w), min_periods=max(10, min(int(args.cvd_window), w))).agg(["mean", "std"])
        out[z_col] = (out[delta_col] - hist["mean"]) / hist["std"].replace(0.0, np.nan)

    if "cvd_delta_5" in out.columns and "cvd_delta_30" in out.columns:
        out["cvd_short_turn_up_after_dump"] = (out["cvd_delta_30"] < 0) & (out["cvd_delta_5"] > 0)
    elif "cvd_delta_15" in out.columns and "cvd_delta_60" in out.columns:
        out["cvd_short_turn_up_after_dump"] = (out["cvd_delta_60"] < 0) & (out["cvd_delta_15"] > 0)
    else:
        out["cvd_short_turn_up_after_dump"] = False

    out["buy_pressure_high"] = out["buy_pressure"] >= out["buy_pressure"].quantile(0.75)
    out["sell_pressure_high"] = out["sell_pressure"] >= out["sell_pressure"].quantile(0.75)
    out["large_trade_share_high"] = out["large_trade_share"] >= out["large_trade_share"].quantile(0.75)
    out["large_buy_absorption"] = (out["large_delta_pressure"] > 0) & (out["down_spike_pct"] >= 0.008)

    _add_fixed_threshold_flags(out, args)
    return _add_historical_rolling_quantile_flags(out, args)


def attach_extra_features_to_events(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    extra_cols = [
        "total_notional",
        "buy_pressure",
        "sell_pressure",
        "notional_imbalance",
        "buy_volume_pressure",
        "volume_imbalance",
        "buy_trade_count_pressure",
        "trade_count_imbalance",
        "large_trade_share",
        "large_delta_pressure",
        "delta_pressure",
        "cvd_short_turn_up_after_dump",
        "buy_pressure_high",
        "sell_pressure_high",
        "large_trade_share_high",
        "large_buy_absorption",
    ]
    for col in features.columns:
        if (
            col.startswith("cvd_delta_")
            or col.startswith("cvd_pressure_")
            or col.startswith("cvd_delta_z_")
            or col.startswith("buy_ratio_ge_")
            or col.startswith("buy_pressure_ge_")
            or col.startswith("delta_pressure_ge_")
            or col.startswith("atr_rq")
            or col.startswith("volume_rq")
            or col.startswith("trades_rq")
            or col.startswith("buy_ratio_rq")
            or col.startswith("buy_pressure_rq")
            or col.startswith("delta_pressure_rq")
            or col.startswith("large_share_rq")
            or col.startswith("large_delta_rq")
            or col.startswith("cvdp15_rq")
            or col.startswith("cvdp60_rq")
        ):
            extra_cols.append(col)
    extra_cols = [c for c in dict.fromkeys(extra_cols) if c in features.columns]
    if not extra_cols:
        return events.copy()
    right = features[extra_cols].copy().reset_index()
    right = right.rename(columns={right.columns[0]: "signal_time"})
    out = events.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"])
    return out.merge(right, on="signal_time", how="left", suffixes=("", "_extra"))


def add_filter_bins(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["spike_bucket"] = fixed_threshold_labels(
        out["down_spike_pct"],
        thresholds=[0.006, 0.008, 0.010, 0.012, 0.016],
        labels=["SPIKE_LT_60", "SPIKE_60_80", "SPIKE_80_100", "SPIKE_100_120", "SPIKE_120_160", "SPIKE_GT_160"],
    )
    out["age_bucket"] = fixed_threshold_labels(
        out["swing_age"],
        thresholds=[6, 12, 24, 48],
        labels=["AGE_0_6", "AGE_7_12", "AGE_13_24", "AGE_25_48", "AGE_GT_48"],
    )
    out["close_pos_bucket"] = fixed_threshold_labels(
        out["close_pos_in_bar"],
        thresholds=[0.2, 0.4, 0.6, 0.8],
        labels=["CLOSE_0_20", "CLOSE_20_40", "CLOSE_40_60", "CLOSE_60_80", "CLOSE_80_100"],
    )
    out["delta_positive"] = pd.to_numeric(out.get("delta_notional", np.nan), errors="coerce") > 0
    out["delta_pressure_positive"] = pd.to_numeric(out.get("delta_pressure", np.nan), errors="coerce") > 0
    out["strong_volume_spike"] = pd.to_numeric(out["volume_ratio"], errors="coerce") >= 2.0
    out["deep_close"] = pd.to_numeric(out["close_pos_in_bar"], errors="coerce") <= 0.30
    out["large_lower_wick"] = pd.to_numeric(out["lower_wick_frac"], errors="coerce") >= 0.45
    return out


def build_fixed_candidate_masks(events: pd.DataFrame) -> dict[str, dict[str, object]]:
    spike = safe_num(events.get("down_spike_pct", np.nan), events.index)
    close_pos = safe_num(events.get("close_pos_in_bar", np.nan), events.index)
    session = events.get("session_bucket", pd.Series("", index=events.index)).astype("object")
    large_q80 = events.get("large_share_rq80_90d", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    atr_q80 = events.get("atr_rq80_90d", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    a = (spike >= 0.0080) & (close_pos <= 0.30) & large_q80
    b = session.eq("S0_00_07") & (spike >= 0.0080) & atr_q80
    c = session.eq("S0_00_07") & (spike >= 0.0120)
    union = a | b | c
    return {
        "A_spike_close_large_share": {"mask": a, "bucket_type": "candidate"},
        "B_session_spike_atr": {"mask": b, "bucket_type": "candidate"},
        "C_session_extreme_spike": {"mask": c, "bucket_type": "candidate"},
        "ABC_union": {"mask": union, "bucket_type": "union_bucket"},
        "A_only": {"mask": a & ~b & ~c, "bucket_type": "overlap_bucket"},
        "B_only": {"mask": b & ~a & ~c, "bucket_type": "overlap_bucket"},
        "C_only": {"mask": c & ~a & ~b, "bucket_type": "overlap_bucket"},
        "AB_overlap_only": {"mask": a & b & ~c, "bucket_type": "overlap_bucket"},
        "AC_overlap_only": {"mask": a & c & ~b, "bucket_type": "overlap_bucket"},
        "BC_overlap_only": {"mask": b & c & ~a, "bucket_type": "overlap_bucket"},
        "ABC_overlap": {"mask": a & b & c, "bucket_type": "overlap_bucket"},
    }


def attach_support_zone_metrics(events: pd.DataFrame, bars: pd.DataFrame, args: Any) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy().sort_values("signal_time").reset_index(drop=True)
    out["cluster_touch_count_020"] = 1
    out["cluster_touch_count_030"] = 1
    out["cluster_zone_low_020"] = out.get("swing_level", np.nan)
    out["cluster_zone_low_030"] = out.get("swing_level", np.nan)
    out["cluster_oldest_age_bars_020"] = out.get("swing_age", np.nan)
    out["cluster_oldest_age_bars_030"] = out.get("swing_age", np.nan)
    return out


def merge_asof_event_context(events: pd.DataFrame, ctx: pd.DataFrame, prefix: str, columns: Sequence[str]) -> pd.DataFrame:
    if events.empty or ctx.empty:
        return events
    left = events.sort_values("signal_time").copy()
    right = ctx.copy().sort_index()
    right = right[~right.index.duplicated(keep="last")]
    use_cols = [c for c in columns if c in right.columns]
    if not use_cols:
        return events
    r = right[use_cols].copy().add_prefix(prefix)
    r["ctx_time"] = r.index
    r = r.reset_index(drop=True).sort_values("ctx_time")
    merged = pd.merge_asof(
        left,
        r,
        left_on="signal_time",
        right_on="ctx_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns=["ctx_time"], errors="ignore").sort_values("signal_time").reset_index(drop=True)


def attach_footprint_context(events: pd.DataFrame, args: Any) -> pd.DataFrame:
    out = events.copy()
    if "footprint" not in set(split_csv_names(args.context_sources)):
        return out
    tag = range_code(float(args.footprint_range_pct))
    step_tag = price_step_code(float(args.footprint_price_step))
    try:
        print(f"[mf] load footprint context {tag}_{step_tag}", flush=True)
        fp = OKXRangeFootprintLoader(
            symbol=args.symbol,
            range_pct=float(args.footprint_range_pct),
            price_step=float(args.footprint_price_step),
        ).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
        if fp.empty:
            print("[mf] footprint context empty; skip", flush=True)
            return out
        fp = fp.copy().sort_index()
        denom = (
            pd.to_numeric(fp.get("buy_notional", np.nan), errors="coerce")
            + pd.to_numeric(fp.get("sell_notional", np.nan), errors="coerce")
        ).replace(0.0, np.nan)
        fp["bucket_delta_pressure"] = pd.to_numeric(fp.get("delta_notional", np.nan), errors="coerce") / denom
        agg = fp.groupby("bar_id", dropna=False).agg(
            end_ts=("end_ts", "last") if "end_ts" in fp.columns else ("bucket_delta_pressure", "last"),
            fp_notional=("notional", "sum"),
            fp_delta_notional=("delta_notional", "sum"),
            fp_max_bucket_abs_delta_pressure=("bucket_delta_pressure", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            fp_low_bucket_delta_pressure=("bucket_delta_pressure", "first"),
            fp_high_bucket_delta_pressure=("bucket_delta_pressure", "last"),
        ).reset_index()
        if "end_ts" not in agg or pd.to_datetime(agg["end_ts"], errors="coerce").isna().all():
            return out
        agg["end_ts"] = pd.to_datetime(agg["end_ts"])
        agg = agg.set_index("end_ts").sort_index()
        denom2 = pd.to_numeric(agg["fp_notional"], errors="coerce").replace(0.0, np.nan)
        agg[f"fp_{tag}_{step_tag}_delta_pressure"] = pd.to_numeric(agg["fp_delta_notional"], errors="coerce") / denom2
        return merge_asof_event_context(
            out,
            agg,
            "",
            [f"fp_{tag}_{step_tag}_delta_pressure", "fp_max_bucket_abs_delta_pressure", "fp_low_bucket_delta_pressure", "fp_high_bucket_delta_pressure"],
        )
    except Exception as exc:
        print(f"[mf] footprint context skipped: {exc}", flush=True)
        return out


def prepare_studied_events(bars: pd.DataFrame, args: Any) -> pd.DataFrame:
    features = build_enriched_features(bars, args)
    events = build_low_sweep_events(features, args)
    if events.empty:
        return events
    events = attach_extra_features_to_events(events, features)
    events = add_filter_bins(events)
    events = build_canonical_events(events)
    events["signal_time"] = pd.to_datetime(events["signal_time"])
    events["year"] = events["signal_time"].dt.year
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    return events.loc[(events["signal_time"] >= start) & (events["signal_time"] <= end)].reset_index(drop=True)


def prepare_events_and_context(bars: pd.DataFrame, args: Any) -> pd.DataFrame:
    events = prepare_studied_events(bars, args)
    print(f"[mf] canonical events={len(events):,}", flush=True)
    if events.empty:
        return events
    events = attach_support_zone_metrics(events, bars, args)
    events = attach_footprint_context(events, args)
    return events

