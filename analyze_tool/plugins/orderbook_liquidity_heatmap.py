#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CoinGlass-style offline OKX order-book liquidity heatmap.

The display layer is built from reconstructed historical Books.  Raw Trades are
used only for causal removal attribution (consumed/cancelled/replenished).  The
same compact feature artifacts are exposed through
``src.data_feed.okx_liquidity_map_loader`` for later event studies/backtests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import (
    PluginParam,
    PluginRunContext,
    PluginRunResult,
    PriceHeatmapCell,
    PriceRegion,
)
from src.data_feed.okx_liquidity_map_loader import OKXLiquidityMapLoader
from src.liquidity_map.aggregation import (
    aggregate_heatmap_cells,
    infer_heatmap_seconds,
    seconds_to_timeframe,
    timeframe_to_seconds,
)
from src.liquidity_map.depth_scale import (
    CausalDepthScaleConfig,
    attach_causal_depth_scale,
)
from src.liquidity_map.wall_detector import (
    PersistentWallConfig,
    detect_persistent_liquidity_walls,
)

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"


def _values(series: pd.Series, length: int | None = None) -> list[Any]:
    if length is not None and len(series) != length:
        series = series.reindex(range(length))
    out: list[Any] = []
    for value in series:
        if value is None or pd.isna(value):
            out.append(None)
        elif isinstance(value, (np.integer, int)):
            out.append(int(value))
        elif isinstance(value, (np.floating, float)):
            number = float(value)
            out.append(number if math.isfinite(number) else None)
        else:
            out.append(str(value))
    return out


def _categorical(values: list[Any]) -> dict[str, Any]:
    categories: dict[str, str] = {}
    code_by_text: dict[str, int] = {}
    codes: list[int | None] = []
    for value in values:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            codes.append(None)
            continue
        text = str(value)
        code = code_by_text.get(text)
        if code is None:
            code = len(code_by_text) + 1
            code_by_text[text] = code
            categories[str(code)] = text
        codes.append(code)
    return {"values": codes, "categories": categories}


def _fmt_number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _timezone_offset() -> pd.Timedelta:
    text = str(TIMEZONE).strip()
    if text.startswith("+"):
        return pd.Timedelta(hours=float(text[1:] or 0))
    if text.startswith("-"):
        return -pd.Timedelta(hours=float(text[1:] or 0))
    return pd.Timedelta(0)


def _aggregate_for_display(
    frame: pd.DataFrame,
    *,
    source_price_step: float,
    display_price_step: float,
    render_seconds: int,
    depth_unit: str,
) -> pd.DataFrame:
    grouped = aggregate_heatmap_cells(
        frame,
        target_seconds=render_seconds,
        source_price_step=source_price_step,
        target_price_step=display_price_step,
    )
    if grouped.empty:
        return grouped
    metric = "depth_usd" if depth_unit == "usd" else "depth_base"
    grouped["display_depth"] = pd.to_numeric(grouped[metric], errors="coerce").fillna(0.0)
    grouped["display_order_count"] = pd.to_numeric(grouped["order_count"], errors="coerce").fillna(0.0)
    side_max = grouped.groupby(["bucket_start_ms", "side"], observed=True)["display_depth"].transform("max")
    grouped["local_ratio"] = np.where(side_max > 0, grouped["display_depth"] / side_max, 0.0)
    grouped.attrs["display_price_step"] = float(grouped.attrs.get("price_step", display_price_step))
    grouped.attrs["render_seconds"] = int(grouped.attrs.get("heatmap_seconds", render_seconds))
    return grouped



def _aggregate_snapshot_for_wall(
    frame: pd.DataFrame,
    *,
    source_price_step: float,
    display_price_step: float,
    target_seconds: int,
) -> pd.DataFrame:
    """Build exact period-end wall snapshots at a coarser causal cadence.

    Generic heatmap aggregation time-averages depth, which is correct for the
    visual heatmap but wrong for a real-time wall detector.  This helper first
    selects the globally latest completed 5s source snapshot inside each target
    bucket, then merges only that snapshot's price levels.  A level that was
    cancelled before the target boundary is therefore absent rather than
    carried forward from an earlier source row.
    """

    required = {
        "bucket_start_ms", "bucket_end_ms", "side_code", "price_low",
        "end_depth_base", "end_depth_usd", "end_order_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"wall snapshot aggregation requires exact end-state fields {missing}; "
            "please force-rebuild the affected liquidity-map day"
        )
    source_seconds = infer_heatmap_seconds(frame)
    target_seconds = max(int(target_seconds), source_seconds)
    if target_seconds % source_seconds:
        target_seconds = int(math.ceil(target_seconds / source_seconds)) * source_seconds
    target_ms = target_seconds * 1000
    source_step = float(source_price_step)
    target_step = max(source_step, float(display_price_step))
    ratio = target_step / source_step
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("wall target price step must be an integer multiple of source price step")

    work = frame.copy()
    work["bucket_start_ms"] = pd.to_numeric(work["bucket_start_ms"], errors="coerce")
    work["bucket_end_ms"] = pd.to_numeric(work["bucket_end_ms"], errors="coerce")
    work = work.dropna(subset=["bucket_start_ms", "bucket_end_ms", "price_low", "side_code"]).copy()
    if work.empty:
        return work
    work["bucket_start_ms"] = work["bucket_start_ms"].astype("int64")
    work["bucket_end_ms"] = work["bucket_end_ms"].astype("int64")
    work["target_start_ms"] = (work["bucket_start_ms"] // target_ms) * target_ms
    latest_end = work.groupby("target_start_ms", observed=True)["bucket_end_ms"].transform("max")
    work = work.loc[work["bucket_end_ms"] == latest_end].copy()
    work["price_low"] = pd.to_numeric(work["price_low"], errors="coerce")
    work = work.dropna(subset=["price_low"]).copy()
    work["price_index_out"] = np.floor(work["price_low"] / target_step + 1e-12).astype("int64")
    for column in ("end_depth_base", "end_depth_usd", "end_order_count"):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    grouped = (
        work.groupby(["target_start_ms", "price_index_out", "side_code"], sort=True, observed=True)
        .agg(
            end_depth_base=("end_depth_base", "sum"),
            end_depth_usd=("end_depth_usd", "sum"),
            end_order_count=("end_order_count", "sum"),
            source_bucket_end_ms=("bucket_end_ms", "max"),
        )
        .reset_index()
        .rename(columns={"target_start_ms": "bucket_start_ms", "price_index_out": "price_index"})
    )
    grouped["bucket_start_ms"] = grouped["bucket_start_ms"].astype("int64")
    grouped["bucket_end_ms"] = grouped["bucket_start_ms"] + target_ms
    grouped["price_index"] = grouped["price_index"].astype("int64")
    grouped["side_code"] = grouped["side_code"].astype("int8")
    grouped["side"] = grouped["side_code"].map({1: "bid", -1: "ask"}).fillna("unknown")
    grouped["price_low"] = grouped["price_index"] * target_step
    grouped["price_high"] = grouped["price_low"] + target_step
    grouped["depth_base"] = grouped["end_depth_base"]
    grouped["depth_usd"] = grouped["end_depth_usd"]
    grouped["order_count"] = grouped["end_order_count"].round().clip(lower=0).astype("int64")
    grouped["end_order_count"] = grouped["order_count"]
    side_max = grouped.groupby(["bucket_start_ms", "side_code"], observed=True)["end_depth_base"].transform("max")
    grouped["local_depth_ratio"] = np.where(side_max > 0, grouped["end_depth_base"] / side_max, 0.0)
    grouped = grouped.sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)
    grouped.attrs.update(frame.attrs)
    grouped.attrs["source_heatmap_seconds"] = source_seconds
    grouped.attrs["heatmap_seconds"] = target_seconds
    grouped.attrs["price_step"] = target_step
    return grouped

def _time_axis_column(frame: pd.DataFrame) -> str | None:
    for name in ("bar_start_ms", "bucket_start_ms"):
        if name in frame.columns:
            return name
    return None


def _attach_distance_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach causal snapshot-mid distance bands used by display baselines.

    The midpoint is reconstructed independently at each completed display
    timestamp from the best visible bid/ask.  No later snapshot participates.
    """

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    time_column = _time_axis_column(out)
    required = {"side", "price_low", "price_high", "price_mid"}
    if time_column is None or not required.issubset(out.columns):
        if "distance_bps" not in out:
            out["distance_bps"] = np.nan
        if "distance_band" not in out:
            out["distance_band"] = pd.Series("all", index=out.index, dtype="string")
        return out

    if "snapshot_mid" not in out or pd.to_numeric(out["snapshot_mid"], errors="coerce").isna().all():
        best_bid = (
            out.loc[out["side"] == "bid"]
            .groupby(time_column, observed=True)["price_high"]
            .max()
        )
        best_ask = (
            out.loc[out["side"] == "ask"]
            .groupby(time_column, observed=True)["price_low"]
            .min()
        )
        mid = ((best_bid + best_ask) / 2.0).rename("snapshot_mid")
        out = out.drop(columns=["snapshot_mid"], errors="ignore").merge(
            mid, left_on=time_column, right_index=True, how="left"
        )

    snapshot_mid = pd.to_numeric(out["snapshot_mid"], errors="coerce")
    price_mid = pd.to_numeric(out["price_mid"], errors="coerce")
    out["distance_bps"] = np.where(
        snapshot_mid > 0,
        (price_mid - snapshot_mid).abs() / snapshot_mid * 10_000.0,
        np.nan,
    )
    out["distance_band"] = pd.cut(
        out["distance_bps"], bins=_DISTANCE_BINS_BPS, labels=_DISTANCE_LABELS, right=True
    ).astype("string")
    out["distance_band"] = out["distance_band"].fillna("all")
    return out


def _causal_rolling_depth_cap(
    frame: pd.DataFrame,
    *,
    window_hours: int,
    percentile: float,
    fallback_cap: float,
) -> pd.Series:
    """Return a per-row color cap using only the current/past book state.

    Production heatmap rows are grouped by side and distance-from-mid band.  At
    each timestamp we first take the cross-sectional depth percentile inside
    that band, then take the same percentile across the *previous* rolling
    window (``closed='left'``).  The current timestamp therefore cannot raise
    its own historical cap, and extending the query into the future cannot
    recolor already rendered history.

    The first observation of a group has no prior history.  It bootstraps from
    that timestamp's own cross-sectional percentile, which is causal because
    the snapshot is already complete.  Frames without temporal metadata use
    the explicit fallback cap rather than a query-wide quantile.
    """

    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    depth = pd.to_numeric(frame["display_depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    time_column = _time_axis_column(frame)
    if time_column is None:
        return pd.Series(max(float(fallback_cap), 1e-12), index=frame.index, dtype=float)

    work = _attach_distance_context(frame)
    work = work.assign(_row_index=frame.index, _depth=depth.to_numpy(dtype=float))
    work[time_column] = pd.to_numeric(work[time_column], errors="coerce")
    work = work.dropna(subset=[time_column]).copy()
    if work.empty:
        return pd.Series(max(float(fallback_cap), 1e-12), index=frame.index, dtype=float)
    work[time_column] = work[time_column].astype("int64")
    work["side"] = work.get("side", pd.Series("all", index=work.index)).astype("string").fillna("all")
    work["distance_band"] = work.get(
        "distance_band", pd.Series("all", index=work.index, dtype="string")
    ).astype("string").fillna("all")

    quantile = min(0.999, max(0.5, float(percentile)))
    references = (
        work.groupby([time_column, "side", "distance_band"], observed=True)["_depth"]
        .quantile(quantile)
        .rename("snapshot_depth_reference")
        .reset_index()
        .sort_values(time_column)
    )
    window = pd.Timedelta(hours=max(1, int(window_hours)))
    cap_frames: list[pd.DataFrame] = []
    for (_, _), group in references.groupby(["side", "distance_band"], observed=True, sort=False):
        ordered = group.sort_values(time_column).copy()
        times = pd.to_datetime(ordered[time_column], unit="ms", utc=True)
        values = pd.Series(
            ordered["snapshot_depth_reference"].to_numpy(dtype=float), index=times
        )
        historical = values.rolling(
            window=window,
            closed="left",
            min_periods=1,
        ).quantile(quantile)
        bootstrap = ordered["snapshot_depth_reference"].to_numpy(dtype=float)
        cap_values = historical.to_numpy(dtype=float)
        cap_values = np.where(np.isfinite(cap_values) & (cap_values > 0), cap_values, bootstrap)
        cap_values = np.where(
            np.isfinite(cap_values) & (cap_values > 0),
            cap_values,
            max(float(fallback_cap), 1e-12),
        )
        ordered["causal_color_cap"] = cap_values
        cap_frames.append(
            ordered[[time_column, "side", "distance_band", "causal_color_cap"]]
        )

    if not cap_frames:
        return pd.Series(max(float(fallback_cap), 1e-12), index=frame.index, dtype=float)
    caps = pd.concat(cap_frames, ignore_index=True)
    keyed = work.merge(
        caps,
        on=[time_column, "side", "distance_band"],
        how="left",
        validate="many_to_one",
    )
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    result.loc[keyed["_row_index"].to_numpy()] = pd.to_numeric(
        keyed["causal_color_cap"], errors="coerce"
    ).to_numpy(dtype=float)
    return result.fillna(max(float(fallback_cap), 1e-12)).clip(lower=1e-12)


def _normalize_depth(
    frame: pd.DataFrame,
    *,
    mode: str,
    manual_max: float,
    rolling_window_hours: int = 24,
    rolling_percentile: float = 0.99,
    display_contrast_gamma: float = 2.8,
    return_caps: bool = False,
) -> tuple[pd.Series, float] | tuple[pd.Series, float, pd.Series]:
    """Normalize heatmap depth without using the full requested interval.

    ``causal_max_ratio`` uses the rolling global maximum shared with the
    wall detector. ``log_depth``, ``auto_window`` and ``salience`` retain their
    older causal percentile caps for comparison. ``manual`` and ``local_ratio``
    remain explicit modes.
    """

    depth = pd.to_numeric(frame["display_depth"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if mode == "causal_max_ratio":
        scaled = attach_causal_depth_scale(
            frame,
            depth_column="display_depth",
            config=CausalDepthScaleConfig(
                window_hours=rolling_window_hours,
                snapshot_reference_quantile=rolling_percentile,
            ),
            ratio_column="_causal_amount_ratio",
            reference_column="_causal_amount_reference",
            snapshot_max_column="_snapshot_side_max_depth",
        )
        intensity = pd.to_numeric(scaled["_causal_amount_ratio"], errors="coerce").fillna(0.0).clip(0, 1)
        caps = pd.to_numeric(scaled["_causal_amount_reference"], errors="coerce").fillna(max(float(manual_max), 1e-12)).clip(lower=1e-12)
        cap = float(caps.median()) if not caps.empty else max(float(manual_max), 1e-12)
        result = (intensity, cap, caps) if return_caps else (intensity, cap)
        return result
    if mode == "local_ratio":
        intensity = pd.to_numeric(frame["local_ratio"], errors="coerce").fillna(0.0).clip(0, 1)
        caps = pd.Series(1.0, index=frame.index, dtype=float)
        result = (intensity, 1.0, caps) if return_caps else (intensity, 1.0)
        return result
    if mode == "manual":
        cap = max(float(manual_max), 1e-12)
        intensity = (depth / cap).clip(0, 1)
        caps = pd.Series(cap, index=frame.index, dtype=float)
        result = (intensity, cap, caps) if return_caps else (intensity, cap)
        return result

    caps = _causal_rolling_depth_cap(
        frame,
        window_hours=rolling_window_hours,
        percentile=rolling_percentile,
        fallback_cap=manual_max,
    )
    safe_caps = pd.to_numeric(caps, errors="coerce").fillna(max(float(manual_max), 1e-12)).clip(lower=1e-12)
    cap_summary = float(safe_caps.median()) if not safe_caps.empty else max(float(manual_max), 1e-12)

    if mode == "log_depth":
        clipped = np.minimum(depth.to_numpy(dtype=float), safe_caps.to_numpy(dtype=float))
        denominator = np.log1p(safe_caps.to_numpy(dtype=float))
        intensity_values = np.divide(
            np.log1p(clipped),
            denominator,
            out=np.zeros_like(clipped, dtype=float),
            where=denominator > 0,
        )
        # Keep the causal rolling cap, but restore human-readable contrast.
        # The raw logarithmic ratio packs ordinary levels into red/purple.
        # A monotone display-only gamma curve spreads them back across the
        # classic pale/yellow/orange/red/purple palette without changing rank,
        # detector inputs, or any backtest feature.
        gamma = max(1.0, float(display_contrast_gamma))
        intensity_values = np.power(np.clip(intensity_values, 0.0, 1.0), gamma)
        intensity = pd.Series(intensity_values, index=frame.index, dtype=float).clip(0, 1)
    else:
        linear = (depth / safe_caps).clip(0, 1)
        if mode == "salience":
            local_ratio = pd.to_numeric(frame.get("local_ratio", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
            # Causal exceptional-liquidity view: absolute rolling percentile
            # strength plus current same-side prominence.  No visible-window
            # median/quantile is consulted.
            intensity = (linear.pow(0.8) * (0.35 + 0.65 * local_ratio.pow(0.5))).clip(0, 1)
        else:
            intensity = linear

    result = (intensity, cap_summary, safe_caps) if return_caps else (intensity, cap_summary)
    return result

def _sample_evenly(frame: pd.DataFrame, count: int, *, order_by: str = "price_low") -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0]
    if len(frame) <= count:
        return frame
    ordered = frame.sort_values(order_by)
    positions = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return ordered.iloc[np.unique(positions)]


def _reduce_cells(frame: pd.DataFrame, *, min_intensity: float, max_cells: int) -> tuple[pd.DataFrame, float]:
    """Bound browser payload without erasing shallow last-snapshot liquidity.

    If the payload fits, every positive cell survives. If it does not, the
    reducer always keeps the strongest cell for each timestamp/side and then
    distributes the remaining budget across weak, medium and strong cells.
    Sampling is global and deterministic on time/price order, avoiding the
    expensive per-bar Python loop used by the first V2.5.2 draft.
    """

    if frame is None or frame.empty:
        return frame, float(min_intensity)
    threshold = max(0.0, float(min_intensity))
    max_cells = max(1, int(max_cells))
    time_column = "bar_start_ms" if "bar_start_ms" in frame.columns else "bucket_start_ms"
    work = frame.loc[pd.to_numeric(frame["display_depth"], errors="coerce").fillna(0.0) > 0].copy()
    if work.empty:
        return work, threshold
    work["_intensity"] = pd.to_numeric(work["intensity"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    group_columns = [time_column] + (["side"] if "side" in work.columns else [])
    sort_columns = group_columns + ["_intensity", "price_low"]
    ascending = [True] * len(group_columns) + [False, True]
    mandatory = (
        work.sort_values(sort_columns, ascending=ascending)
        .groupby(group_columns, sort=False, observed=True)
        .head(1)
    )
    visible = work if threshold <= 0 else work.loc[work["_intensity"] + 1e-12 >= threshold]
    candidate = work.loc[mandatory.index.union(visible.index)].copy()
    if len(candidate) <= max_cells:
        selected = candidate
    elif len(mandatory) >= max_cells:
        selected = mandatory.nlargest(max_cells, "_intensity")
    else:
        selected_parts: list[pd.DataFrame] = [mandatory]
        selected_indices = set(int(value) for value in mandatory.index)
        pool = candidate.loc[~candidate.index.isin(selected_indices)].copy()
        remaining = max_cells - len(mandatory)
        intensity = pool["_intensity"]
        strata = [
            ("weak", pool.loc[intensity < 0.15], 0.40),
            ("medium", pool.loc[(intensity >= 0.15) & (intensity < 0.50)], 0.30),
            ("strong", pool.loc[intensity >= 0.50], 0.30),
        ]
        for name, bucket, share in strata:
            target = min(len(bucket), max(0, int(round(remaining * share))))
            if target <= 0:
                continue
            if name == "strong":
                part = bucket.nlargest(target, "_intensity")
            else:
                part = _sample_evenly(
                    bucket.sort_values([time_column, "side", "price_low"] if "side" in bucket.columns else [time_column, "price_low"]),
                    target,
                    order_by=time_column,
                )
            selected_parts.append(part)
            selected_indices.update(int(value) for value in part.index)
        used = sum(len(part) for part in selected_parts)
        if used < max_cells:
            leftovers = pool.loc[~pool.index.isin(selected_indices)]
            selected_parts.append(leftovers.nlargest(max_cells - used, "_intensity"))
        selected = pd.concat(selected_parts, axis=0).drop_duplicates()
        if len(selected) > max_cells:
            # Preserve mandatory rows; trim only extras.
            extra_budget = max_cells - len(mandatory)
            extras = selected.loc[~selected.index.isin(mandatory.index)].nlargest(max(0, extra_budget), "_intensity")
            selected = pd.concat([mandatory, extras], axis=0)

    order = [time_column, "side", "price_low"] if "side" in selected.columns else [time_column, "price_low"]
    return (
        selected.drop(columns=["_intensity"], errors="ignore")
        .drop_duplicates()
        .sort_values(order)
        .reset_index(drop=True),
        threshold,
    )



def _compact_heatmap_payload(
    frame: pd.DataFrame,
    *,
    depth_unit: str,
    color_mode: str,
    display_price_step: float,
) -> dict[str, Any]:
    """Encode a large heatmap as compact columnar arrays.

    The former object-per-cell JSON repeats timestamps, labels, colors and a
    large ``fields`` mapping hundreds of thousands of times.  At 800k cells
    that response can exceed several hundred MB and the HTTP connection may be
    terminated before the browser receives valid JSON.  This representation
    keeps the exact same cells and intensities while storing per-column values
    once and using short numeric arrays for each cell.
    """

    if frame is None or frame.empty:
        return {"v": 1, "starts": [], "ends": [], "c": [], "p": [], "i": [], "s": [], "d": [], "o": []}

    work = frame.sort_values(["bar_start_ms", "side_code", "price_low"]).reset_index(drop=True)
    column_table = (
        work[[
            "bar_start_ms",
            "bar_start",
            "bar_end",
            "source_bucket_start_ms",
            "source_bucket_end_ms",
            "source_lag_ms",
        ]]
        .drop_duplicates("bar_start_ms", keep="last")
        .sort_values("bar_start_ms")
        .reset_index(drop=True)
    )
    column_map = pd.Series(np.arange(len(column_table), dtype=np.int32), index=column_table["bar_start_ms"].astype("int64"))
    column_indices = work["bar_start_ms"].astype("int64").map(column_map).astype("int32").to_numpy(copy=False)

    step = max(float(display_price_step), 1e-12)
    price_indices = np.rint(pd.to_numeric(work["price_low"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / step).astype(np.int32)
    intensity_scale = 10_000
    intensities = np.rint(
        pd.to_numeric(work["intensity"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        * intensity_scale
    ).astype(np.uint16)
    side_codes = np.where(work["side"].astype(str).to_numpy() == "bid", 1, -1).astype(np.int8)
    depths = np.round(pd.to_numeric(work["display_depth"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 6)
    orders = np.rint(pd.to_numeric(work["display_order_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)).astype(np.int32)
    large_source = work["is_large_rolling"] if "is_large_rolling" in work.columns else pd.Series(False, index=work.index)
    large = pd.to_numeric(large_source, errors="coerce").fillna(0).astype(bool).to_numpy(dtype=np.uint8)

    def stamps(column: str) -> list[str]:
        return [pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S") for value in column_table[column]]

    source_starts = [
        (pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(None) + _timezone_offset()).strftime("%Y-%m-%d %H:%M:%S")
        for value in column_table["source_bucket_start_ms"]
    ]
    source_ends = [
        (pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(None) + _timezone_offset()).strftime("%Y-%m-%d %H:%M:%S")
        for value in column_table["source_bucket_end_ms"]
    ]

    return {
        "v": 1,
        "step": step,
        "scale": intensity_scale,
        "unit": "USD" if depth_unit == "usd" else "ETH",
        "color_mode": str(color_mode),
        "starts": stamps("bar_start"),
        "ends": stamps("bar_end"),
        "source_starts": source_starts,
        "source_ends": source_ends,
        "source_lags": pd.to_numeric(column_table["source_lag_ms"], errors="coerce").fillna(0).astype("int64").tolist(),
        "c": column_indices.tolist(),
        "p": price_indices.tolist(),
        "i": intensities.astype(np.int32).tolist(),
        "s": side_codes.astype(np.int32).tolist(),
        "d": depths.tolist(),
        "o": orders.tolist(),
        "l": large.astype(np.int32).tolist(),
    }

def _select_dominant_walls(walls: list[Any], *, max_regions: int) -> list[Any]:
    """Cap wall lifecycles without merging distinct real-time tracks.

    V1 used hindsight rectangles and needed visual non-maximum suppression.
    V2 already keeps exact per-snapshot ranges; suppressing overlapping tracks
    can hide a real point wall beside a real wall zone.  We therefore only cap
    by lifecycle strength and keep chronological order.
    """

    if not walls:
        return []
    strongest = sorted(
        walls,
        key=lambda wall: (
            float(getattr(wall, "strength_score", 0.0)),
            float(getattr(wall, "duration_minutes", 0.0)),
        ),
        reverse=True,
    )[: max(1, int(max_regions))]
    return sorted(strongest, key=lambda wall: (wall.confirmed_at_ms, wall.price_low))


def _choose_auto_render_seconds(*, duration_seconds: float, source_seconds: int, max_columns: int) -> int:
    """Choose the finest source-multiple that fits the visible time span."""
    source = max(1, int(source_seconds))
    columns = max(100, int(max_columns))
    minimum = max(source, int(math.ceil(max(duration_seconds, 1.0) / columns)))
    multiples = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1800, 3600)
    for candidate in multiples:
        seconds = source * candidate
        if seconds >= minimum:
            return seconds
    return int(math.ceil(minimum / source)) * source


_ANALYZE_FEATURE_COLUMNS = (
    "bucket_end_ms",
    "available_time_ms",
    "book_valid",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread_bps",
    "depth_imbalance_25bps",
    "top_bid_wall_price",
    "top_ask_wall_price",
    "top_bid_wall_depth_base",
    "top_ask_wall_depth_base",
    "nearest_large_bid_price",
    "nearest_large_ask_price",
    "nearest_large_bid_depth_base",
    "nearest_large_ask_depth_base",
    "large_bid_depth_base",
    "large_ask_depth_base",
    "aggressive_buy_base",
    "aggressive_sell_base",
    "estimated_bid_cancel_base",
    "estimated_ask_cancel_base",
    "estimated_bid_consumed_base",
    "estimated_ask_consumed_base",
    "estimated_bid_replenished_base",
    "estimated_ask_replenished_base",
)


def _align_features_to_bars(loader: OKXLiquidityMapLoader, bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(index=bars.index)
    direct = getattr(loader, "align_features_to_times", None)
    if callable(direct):
        aligned = direct(
            pd.DatetimeIndex(pd.to_datetime(bars.index)),
            project_time=True,
            tolerance="5m",
            columns=list(_ANALYZE_FEATURE_COLUMNS),
        )
        aligned.index = pd.DatetimeIndex(pd.to_datetime(bars.index))
        return aligned
    start, end = pd.Timestamp(bars.index.min()), pd.Timestamp(bars.index.max())
    features = loader.load_features(start - pd.Timedelta(minutes=5), end, project_time=True, index_mode="none")
    left = pd.DataFrame({"bar_timestamp": pd.DatetimeIndex(pd.to_datetime(bars.index))}).sort_values("bar_timestamp")
    if features.empty:
        return pd.DataFrame(index=left["bar_timestamp"])
    right = features.sort_values("available_time").copy()
    aligned = pd.merge_asof(
        left,
        right,
        left_on="bar_timestamp",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(minutes=5),
    )
    aligned.index = pd.DatetimeIndex(left["bar_timestamp"])
    return aligned



_DISTANCE_BINS_BPS = (-1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0, math.inf)
_DISTANCE_LABELS = ("0-10", "10-25", "25-50", "50-100", "100-200", "200-500", "500-1000", "1000+")


def _project_naive_to_utc_ms(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return int(ts.tz_convert("UTC").timestamp() * 1000)
    return int((ts - _timezone_offset()).tz_localize("UTC").timestamp() * 1000)


def _bar_intervals(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    warmup_hours: int = 0,
) -> pd.DataFrame:
    """Build exact chart-bar intervals in exchange UTC milliseconds.

    Time-based bars use their declared timeframe rather than the next returned
    row.  This prevents a missing candle or a request boundary from stretching a
    heatmap column across an unrelated gap.  Warmup intervals are synthetic and
    exist only to seed the causal rolling wall threshold.
    """

    if bars is None or bars.empty:
        return pd.DataFrame(columns=["bar_start", "bar_end", "bar_start_ms", "bar_end_ms", "visible"])
    seconds = timeframe_to_seconds(timeframe or "1m")
    step = pd.Timedelta(seconds=seconds)
    visible_starts = pd.DatetimeIndex(pd.to_datetime(bars.index)).sort_values().unique()
    starts = list(visible_starts)
    if warmup_hours > 0 and len(visible_starts):
        count = int(math.ceil(warmup_hours * 3600 / seconds))
        first = pd.Timestamp(visible_starts[0])
        starts = [first - step * i for i in range(count, 0, -1)] + starts
    visible_set = {pd.Timestamp(value) for value in visible_starts}
    rows = []
    for value in starts:
        start = pd.Timestamp(value)
        end = start + step
        rows.append(
            {
                "bar_start": start,
                "bar_end": end,
                "bar_start_ms": _project_naive_to_utc_ms(start),
                "bar_end_ms": _project_naive_to_utc_ms(end),
                "visible": start in visible_set,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("bar_start_ms").sort_values("bar_start_ms").reset_index(drop=True)


def _period_end_snapshot_for_bars(
    frame: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    source_price_step: float,
    display_price_step: float,
    depth_unit: str,
) -> pd.DataFrame:
    """Select the last completed source snapshot inside each chart bar.

    This is the CoinGlass/TradingLite-style display semantic: the current bar
    can be overwritten by newer snapshots, while a completed bar freezes the
    last snapshot available before its end.  No earlier price level is carried
    forward when it is absent from the selected snapshot.
    """

    if frame is None or frame.empty or bars is None or bars.empty:
        return pd.DataFrame()
    source_step = float(source_price_step)
    effective_step = max(source_step, float(display_price_step))
    ratio = effective_step / source_step
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"显示价格格 ${effective_step:g} 必须是基础价格格 ${source_step:g} 的整数倍"
        )

    required_end_fields = {
        "end_depth_base", "end_depth_usd", "end_order_count", "end_local_depth_ratio"
    }
    missing_end_fields = sorted(required_end_fields.difference(frame.columns))
    if missing_end_fields:
        raise ValueError(
            "当前 heatmap 派生文件是旧版平均深度结构，缺少精确周期末盘口字段 "
            f"{missing_end_fields}；请使用 V1.8 代码加 --force-rebuild 重建对应日期。"
        )

    work = frame.copy()
    work["bucket_start_ms"] = pd.to_numeric(work["bucket_start_ms"], errors="coerce")
    work["bucket_end_ms"] = pd.to_numeric(work["bucket_end_ms"], errors="coerce")
    work = work.dropna(subset=["bucket_start_ms", "bucket_end_ms"]).copy()
    if work.empty:
        return work
    work["bucket_start_ms"] = work["bucket_start_ms"].astype("int64")
    work["bucket_end_ms"] = work["bucket_end_ms"].astype("int64")

    snapshots = (
        work[["bucket_start_ms", "bucket_end_ms"]]
        .drop_duplicates()
        .sort_values(["bucket_end_ms", "bucket_start_ms"])
        .reset_index(drop=True)
    )
    source_ends = snapshots["bucket_end_ms"].to_numpy(dtype="int64")
    source_starts = snapshots["bucket_start_ms"].to_numpy(dtype="int64")
    mappings: list[dict[str, Any]] = []
    for bar in bars.itertuples(index=False):
        position = int(np.searchsorted(source_ends, int(bar.bar_end_ms), side="right") - 1)
        if position < 0:
            continue
        source_end = int(source_ends[position])
        source_start = int(source_starts[position])
        if source_end <= int(bar.bar_start_ms):
            continue
        mappings.append(
            {
                "source_bucket_start_ms": source_start,
                "source_bucket_end_ms": source_end,
                "bar_start_ms": int(bar.bar_start_ms),
                "bar_end_ms": int(bar.bar_end_ms),
                "bar_start": pd.Timestamp(bar.bar_start),
                "bar_end": pd.Timestamp(bar.bar_end),
                "visible": bool(bar.visible),
                "source_lag_ms": int(bar.bar_end_ms) - source_end,
            }
        )
    if not mappings:
        return pd.DataFrame()
    mapping = pd.DataFrame(mappings).drop_duplicates("bar_start_ms")
    selected_starts = set(mapping["source_bucket_start_ms"].astype("int64").tolist())
    selected = work.loc[work["bucket_start_ms"].isin(selected_starts)].copy()
    selected = selected.merge(
        mapping,
        left_on="bucket_start_ms",
        right_on="source_bucket_start_ms",
        how="inner",
        validate="many_to_one",
    )
    if selected.empty:
        return selected

    if "price_low" in selected.columns:
        price_low = pd.to_numeric(selected["price_low"], errors="coerce")
    else:
        price_low = pd.to_numeric(selected["price_index"], errors="coerce") * source_step
    selected = selected.loc[price_low.notna()].copy()
    price_low = price_low.loc[selected.index]
    selected["price_index_out"] = np.floor(price_low / effective_step + 1e-12).astype("int64")
    selected["side"] = selected.get("side", selected["side_code"].map({1: "bid", -1: "ask"}))

    metric = "end_depth_usd" if depth_unit == "usd" else "end_depth_base"
    selected["display_depth"] = pd.to_numeric(selected[metric], errors="coerce").fillna(0.0)
    selected["display_order_count"] = pd.to_numeric(
        selected["end_order_count"], errors="coerce"
    ).fillna(0.0)
    # Period-end means exactly the reconstructed book state at the boundary.
    # Rows retained only for within-bucket activity/average depth must not leak
    # into the frozen historical column after the level has disappeared.
    selected = selected.loc[selected["display_depth"] > 0].copy()
    if selected.empty:
        return selected
    for column in ("added_base", "removed_base", "executed_base", "cancelled_base", "consumed_base", "replenished_base"):
        if column not in selected:
            selected[column] = 0.0
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0.0)

    keys = [
        "bar_start_ms", "bar_end_ms", "bar_start", "bar_end", "visible",
        "source_bucket_start_ms", "source_bucket_end_ms", "source_lag_ms",
        "price_index_out", "side_code", "side",
    ]
    grouped = selected.groupby(keys, sort=True, observed=True).agg(
        display_depth=("display_depth", "sum"),
        display_order_count=("display_order_count", "sum"),
        end_depth_base=("end_depth_base", "sum"),
        end_depth_usd=("end_depth_usd", "sum"),
        end_order_count=("end_order_count", "sum"),
        added_base=("added_base", "sum"),
        removed_base=("removed_base", "sum"),
        executed_base=("executed_base", "sum"),
        cancelled_base=("cancelled_base", "sum"),
        consumed_base=("consumed_base", "sum"),
        replenished_base=("replenished_base", "sum"),
    ).reset_index()
    grouped["price_index"] = grouped["price_index_out"].astype("int64")
    grouped["price_low"] = grouped["price_index"] * effective_step
    grouped["price_high"] = grouped["price_low"] + effective_step
    grouped["price_mid"] = (grouped["price_low"] + grouped["price_high"]) / 2.0
    grouped["display_order_count"] = grouped["display_order_count"].round().clip(lower=0)
    side_max = grouped.groupby(["bar_start_ms", "side"], observed=True)["display_depth"].transform("max")
    grouped["local_ratio"] = np.where(side_max > 0, grouped["display_depth"] / side_max, 0.0)
    grouped.attrs["display_price_step"] = effective_step
    grouped.attrs["render_seconds"] = int(round((bars.iloc[0].bar_end - bars.iloc[0].bar_start).total_seconds()))
    return grouped.sort_values(["bar_start_ms", "side_code", "price_index"]).reset_index(drop=True)


def _period_end_cached_for_bars(
    frame: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    depth_unit: str,
) -> pd.DataFrame:
    """Decorate an already cached period-end frame for Analyze Tool display.

    The cache already contains exactly one final source snapshot per chart
    bucket.  This helper only intersects it with the requested chart bars and
    adds display fields; it performs no source heatmap scan or groupby.
    """

    if frame is None or frame.empty or bars is None or bars.empty:
        return pd.DataFrame()
    mapping = bars[["bar_start", "bar_end", "bar_start_ms", "bar_end_ms", "visible"]].copy()
    selected = frame.merge(
        mapping,
        left_on=["bucket_start_ms", "bucket_end_ms"],
        right_on=["bar_start_ms", "bar_end_ms"],
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    if selected.empty:
        return selected
    metric = "end_depth_usd" if depth_unit == "usd" else "end_depth_base"
    selected["display_depth"] = pd.to_numeric(selected[metric], errors="coerce").fillna(0.0)
    selected["display_order_count"] = pd.to_numeric(
        selected["end_order_count"], errors="coerce"
    ).fillna(0.0)
    selected = selected.loc[selected["display_depth"] > 0].copy()
    if selected.empty:
        return selected
    selected["source_lag_ms"] = (
        pd.to_numeric(selected["bar_end_ms"], errors="coerce")
        - pd.to_numeric(selected["source_bucket_end_ms"], errors="coerce")
    ).astype("int64")
    selected["price_mid"] = (selected["price_low"] + selected["price_high"]) / 2.0
    side_max = selected.groupby(["bar_start_ms", "side"], observed=True)["display_depth"].transform("max")
    selected["local_ratio"] = np.where(side_max > 0, selected["display_depth"] / side_max, 0.0)
    for column in ("added_base", "removed_base", "executed_base", "cancelled_base", "consumed_base", "replenished_base"):
        if column not in selected:
            selected[column] = 0.0
        selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
    selected.attrs.update(frame.attrs)
    selected.attrs["display_price_step"] = float(frame.attrs.get("price_step", 1.0))
    selected.attrs["render_seconds"] = int(frame.attrs.get("heatmap_seconds", 60))
    return selected.sort_values(["bar_start_ms", "side_code", "price_index"]).reset_index(drop=True)


def _add_rolling_large_threshold(
    frame: pd.DataFrame,
    *,
    window_hours: int,
    percentile: float,
) -> pd.DataFrame:
    """Attach a causal rolling wall threshold by side and distance band.

    The comparison series is the maximum depth observed in each side/distance
    band at each completed display snapshot.  The current snapshot is excluded
    (`closed='left'`), so a wall cannot raise its own threshold.
    """

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    best_bid = out.loc[out["side"] == "bid"].groupby("bar_start_ms", observed=True)["price_high"].max()
    best_ask = out.loc[out["side"] == "ask"].groupby("bar_start_ms", observed=True)["price_low"].min()
    mid = ((best_bid + best_ask) / 2.0).rename("snapshot_mid")
    out = out.merge(mid, left_on="bar_start_ms", right_index=True, how="left")
    out["distance_bps"] = np.where(
        out["snapshot_mid"] > 0,
        (out["price_mid"] - out["snapshot_mid"]).abs() / out["snapshot_mid"] * 10_000.0,
        np.nan,
    )
    out["distance_band"] = pd.cut(
        out["distance_bps"], bins=_DISTANCE_BINS_BPS, labels=_DISTANCE_LABELS, right=True
    ).astype("string")
    maxima = (
        out.groupby(["bar_start_ms", "side", "distance_band"], observed=True)["display_depth"]
        .max()
        .rename("band_max_depth")
        .reset_index()
        .sort_values("bar_start_ms")
    )
    threshold_frames: list[pd.DataFrame] = []
    quantile = min(0.999, max(0.5, float(percentile)))
    window = pd.Timedelta(hours=max(1, int(window_hours)))
    min_periods = 4
    for (_, _), group in maxima.groupby(["side", "distance_band"], observed=True, sort=False):
        ordered = group.sort_values("bar_start_ms").copy()
        times = pd.to_datetime(ordered["bar_start_ms"], unit="ms", utc=True)
        series = pd.Series(ordered["band_max_depth"].to_numpy(dtype=float), index=times)
        ordered["rolling_large_threshold"] = (
            series.rolling(window=window, closed="left", min_periods=min_periods).quantile(quantile).to_numpy()
        )
        threshold_frames.append(ordered[["bar_start_ms", "side", "distance_band", "rolling_large_threshold"]])
    if threshold_frames:
        thresholds = pd.concat(threshold_frames, ignore_index=True)
        out = out.merge(thresholds, on=["bar_start_ms", "side", "distance_band"], how="left")
    else:
        out["rolling_large_threshold"] = np.nan
    out["is_large_rolling"] = (
        out["rolling_large_threshold"].notna()
        & (out["display_depth"] >= out["rolling_large_threshold"])
    )
    out["rolling_window_hours"] = int(window_hours)
    out["rolling_percentile"] = float(percentile) * 100.0
    return out


def _alignment_audit(
    grouped: pd.DataFrame,
    bars: pd.DataFrame,
    loader: OKXLiquidityMapLoader,
    *,
    source_seconds: int,
    display_price_step: float,
) -> dict[str, Any]:
    """Audit time and price placement without trusting the browser renderer."""

    base = {
        "status": "unavailable",
        "checked_bars": 0,
        "time_alignment_ok": False,
        "price_alignment_ok": False,
        "market_price_alignment_ok": False,
        "max_allowed_price_error": float(display_price_step),
    }
    if grouped is None or grouped.empty or bars is None or bars.empty:
        return base
    visible = grouped.loc[grouped["visible"].astype(bool)].copy()
    if visible.empty:
        return base

    per_bar_all = visible[[
        "bar_start_ms", "bar_end_ms", "source_bucket_end_ms", "source_lag_ms"
    ]].drop_duplicates("bar_start_ms").sort_values("bar_start_ms").reset_index(drop=True)
    # A deterministic uniform sample keeps a month-long 1m request from turning
    # the diagnostic itself into the dominant workload while still checking the
    # full date span, including both ends.
    if len(per_bar_all) > 2000:
        positions = np.linspace(0, len(per_bar_all) - 1, 2000).round().astype(int)
        per_bar = per_bar_all.iloc[np.unique(positions)].copy()
        visible = visible.loc[visible["bar_start_ms"].isin(set(per_bar["bar_start_ms"].tolist()))].copy()
    else:
        per_bar = per_bar_all
    lags = pd.to_numeric(per_bar["source_lag_ms"], errors="coerce").dropna()
    base["checked_bars"] = int(len(per_bar))
    base["time_lag_max_ms"] = int(lags.max()) if not lags.empty else None
    base["time_lag_p95_ms"] = float(lags.quantile(0.95)) if not lags.empty else None
    base["time_alignment_ok"] = bool(
        not lags.empty and (lags >= 0).all() and (lags <= source_seconds * 1000 + 1).all()
    )

    start = pd.Timestamp(bars.index.min()) - pd.Timedelta(minutes=5)
    end = pd.Timestamp(bars.index.max()) + pd.Timedelta(minutes=10)
    try:
        features = loader.load_features(start, end, project_time=True, index_mode="none")
    except Exception:
        features = pd.DataFrame()
    required = {"bucket_end_ms", "book_valid", "best_bid", "best_ask", "mid_price"}
    if features.empty or not required.issubset(features.columns):
        base["status"] = "time_only_pass" if base["time_alignment_ok"] else "warning"
        return base

    features = features.loc[pd.to_numeric(features["book_valid"], errors="coerce").fillna(0) > 0].copy()
    if features.empty:
        base["status"] = "time_only_pass" if base["time_alignment_ok"] else "warning"
        return base
    features["bucket_end_ms"] = pd.to_numeric(features["bucket_end_ms"], errors="coerce")
    features = features.dropna(subset=["bucket_end_ms"]).sort_values("bucket_end_ms")
    sample = pd.merge_asof(
        per_bar.sort_values("source_bucket_end_ms"),
        features[["bucket_end_ms", "best_bid", "best_ask", "mid_price"]],
        left_on="source_bucket_end_ms",
        right_on="bucket_end_ms",
        direction="backward",
        tolerance=max(source_seconds * 1000, 10_000),
    )
    if sample.empty:
        base["status"] = "time_only_pass" if base["time_alignment_ok"] else "warning"
        return base

    bid_top = (
        visible.loc[visible["side"] == "bid"]
        .sort_values(["bar_start_ms", "price_low"])
        .groupby("bar_start_ms", observed=True).tail(1)
        .set_index("bar_start_ms")
    )
    ask_top = (
        visible.loc[visible["side"] == "ask"]
        .sort_values(["bar_start_ms", "price_low"])
        .groupby("bar_start_ms", observed=True).head(1)
        .set_index("bar_start_ms")
    )
    price_checks = []
    mismatch_examples: list[dict[str, Any]] = []
    step = float(display_price_step)
    for row in sample.itertuples(index=False):
        start_ms = int(row.bar_start_ms)
        if start_ms not in bid_top.index or start_ms not in ask_top.index:
            continue
        bid = bid_top.loc[start_ms]
        ask = ask_top.loc[start_ms]
        best_bid_value = float(row.best_bid)
        best_ask_value = float(row.best_ask)
        if not math.isfinite(best_bid_value) or not math.isfinite(best_ask_value):
            continue
        expected_bid_low = math.floor(best_bid_value / step + 1e-12) * step
        expected_ask_low = math.floor(best_ask_value / step + 1e-12) * step
        bid_ok = math.isclose(float(bid.price_low), expected_bid_low, rel_tol=0.0, abs_tol=1e-8)
        ask_ok = math.isclose(float(ask.price_low), expected_ask_low, rel_tol=0.0, abs_tol=1e-8)
        passed = bool(bid_ok and ask_ok)
        price_checks.append(passed)
        if not passed and len(mismatch_examples) < 8:
            mismatch_examples.append(
                {
                    "bar_start_project": (
                        pd.to_datetime(start_ms, unit="ms", utc=True).tz_convert(None) + _timezone_offset()
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "best_bid": best_bid_value,
                    "display_bid_bin": [float(bid.price_low), float(bid.price_high)],
                    "display_bid_depth": float(bid.display_depth),
                    "expected_bid_bin": [expected_bid_low, expected_bid_low + step],
                    "best_ask": best_ask_value,
                    "display_ask_bin": [float(ask.price_low), float(ask.price_high)],
                    "display_ask_depth": float(ask.display_depth),
                    "expected_ask_bin": [expected_ask_low, expected_ask_low + step],
                }
            )
    base["price_checks"] = int(len(price_checks))
    base["price_mismatches"] = int(sum(not value for value in price_checks))
    base["price_mismatch_examples"] = mismatch_examples
    base["price_alignment_ok"] = bool(price_checks and all(price_checks))

    close_by_ms = {
        _project_naive_to_utc_ms(index): float(value)
        for index, value in pd.to_numeric(bars["close"], errors="coerce").items()
        if pd.notna(value)
    }
    deviations = []
    for row in sample.itertuples(index=False):
        close = close_by_ms.get(int(row.bar_start_ms))
        mid_value = float(row.mid_price) if pd.notna(row.mid_price) else math.nan
        if close is None or not math.isfinite(mid_value) or mid_value <= 0:
            continue
        deviations.append(abs(close - mid_value) / mid_value * 10_000.0)
    if deviations:
        series = pd.Series(deviations, dtype=float)
        base["close_mid_median_bps"] = float(series.median())
        base["close_mid_p95_bps"] = float(series.quantile(0.95))
        base["close_mid_max_bps"] = float(series.max())
        # A bar-close and the order-book mid sampled at that bar's end should
        # be very close.  These deliberately generous limits catch an 8-hour
        # or whole-column shift without failing on ordinary spread/no-trade
        # differences between the candle and book feeds.
        base["market_price_alignment_ok"] = bool(
            base["close_mid_median_bps"] <= 5.0 and base["close_mid_p95_bps"] <= 20.0
        )
    else:
        base["close_mid_median_bps"] = None
        base["close_mid_p95_bps"] = None
        base["close_mid_max_bps"] = None
        base["market_price_alignment_ok"] = False

    if base["time_alignment_ok"] and base["price_alignment_ok"] and base["market_price_alignment_ok"]:
        base["status"] = "pass"
    else:
        base["status"] = "warning"
    return base


class OrderBookLiquidityHeatmapPlugin:
    plugin_id = "offline_orderbook_liquidity_heatmap_v1"
    name = "离线订单簿流动性热力图 V2.5.4.1 天蓝矩形墙"
    description = (
        "V2.5.4.1 保持周期末/当前最新盘口矩阵、紧凑JSON与原墙算法，仅把墙框改为高对比天蓝色："
        "墙区必须在时间×价格矩阵中具有足够填充度，不能包含大量白色空洞；买墙必须始终位于市场下方，"
        "卖墙必须始终位于市场上方，价格触碰或穿入后墙立即结束。"
    )
    params = [
        PluginParam(
            name="books_depth",
            label="订单簿深度",
            kind="select",
            default="5000",
            choices=[
                {"value": "5000", "label": "5000档 · 宽范围地图（推荐）"},
                {"value": "400", "label": "400档 · 近端盘口"},
            ],
        ),
        PluginParam(
            name="display_mode",
            label="热力列模式",
            kind="select",
            default="period_end",
            choices=[
                {"value": "period_end", "label": "一根K线一列 · 周期末盘口（推荐）"},
                {"value": "time_weighted", "label": "一根K线一列 · 时间加权均值"},
                {"value": "micro_detail", "label": "微结构细节 · 保留5秒/自动列"},
            ],
        ),
        PluginParam(
            name="normalization",
            label="颜色归一化",
            kind="select",
            default="causal_max_ratio",
            choices=[
                {"value": "causal_max_ratio", "label": "过去24h稳健高位挂单比例（推荐，墙检测同口径）"},
                {"value": "local_ratio", "label": "当前截面同侧最厚=100%（临时观察）"},
                {"value": "auto_window", "label": "过去窗口因果滚动 P99 · 线性"},
                {"value": "log_depth", "label": "过去窗口因果滚动 P99 · 对数"},
                {"value": "salience", "label": "因果滚动显著性（深度+同侧突出度）"},
                {"value": "manual", "label": "手动最大值封顶"},
            ],
        ),
        PluginParam(
            name="depth_unit",
            label="深度单位",
            kind="select",
            default="base",
            choices=[{"value": "base", "label": "ETH"}, {"value": "usd", "label": "USD名义价值"}],
        ),
        PluginParam("manual_max", "手动最深颜色阈值", default=100.0, min_value=0.01, max_value=1000000000, step=1.0),
        PluginParam("color_window_hours", "挂单量参考窗口（小时）", default=24, min_value=1, max_value=168, step=1),
        PluginParam("color_percentile", "每帧高位参考分位数", default=99, min_value=90, max_value=100, step=0.5),
        PluginParam("display_contrast_gamma", "对数模式显示对比度", default=1.4, min_value=1.0, max_value=4.0, step=0.1),
        PluginParam("large_window_hours", "大流动性滚动窗口（小时）", default=24, min_value=1, max_value=168, step=1),
        PluginParam("large_percentile", "大流动性滚动分位数", default=95, min_value=50, max_value=99.9, step=0.5),
        PluginParam(
            name="wall_resolution",
            label="墙识别输入",
            kind="select",
            default="chart",
            choices=[{"value": "chart", "label": "跟随K线周期 · 历史末快照/当前最新（固定）"}],
        ),
        PluginParam("wall_lookback_hours", "墙二维观察窗口（小时）", default=4, min_value=1, max_value=24, step=1),
        PluginParam("wall_min_history_bars", "开始判断前最少K线数", default=4, min_value=2, max_value=96, step=1),
        PluginParam("wall_support_time_coverage_pct", "15%+连接层时间覆盖率（%）", default=70, min_value=20, max_value=100, step=5),
        PluginParam("wall_zone_time_coverage_pct", "30%+主体层时间覆盖率（%）", default=55, min_value=10, max_value=100, step=5),
        PluginParam("wall_core_time_coverage_pct", "50%+核心层时间覆盖率（%）", default=20, min_value=0, max_value=100, step=5),
        PluginParam("wall_min_average_ratio_pct", "区域历史平均挂单比例（%）", default=16, min_value=1, max_value=80, step=1),
        PluginParam("wall_min_current_ratio_pct", "当前列最低保留挂单比例（%）", default=12, min_value=0, max_value=50, step=1),
        PluginParam("wall_strong_ratio_pct", "最深核心阈值（占24h高位参考%）", default=50, min_value=20, max_value=100, step=1),
        PluginParam("wall_zone_ratio_pct", "主墙主体阈值（占24h高位参考%）", default=30, min_value=10, max_value=95, step=1),
        PluginParam("wall_support_ratio_pct", "主墙连接阈值（占24h高位参考%）", default=15, min_value=1, max_value=80, step=1),
        PluginParam("wall_isolated_point_ratio_pct", "单点小阻力/支撑阈值（%）", default=50, min_value=20, max_value=100, step=1),
        PluginParam("wall_min_zone_band_points", "主墙至少30%+价格格数", default=3, min_value=2, max_value=20, step=1),
        PluginParam("wall_min_zone_support_points", "主墙至少总价格格数", default=4, min_value=2, max_value=40, step=1),
        PluginParam("wall_min_zone_density_mass", "主墙最小历史挂单质量", default=1.2, min_value=0.5, max_value=20, step=0.1),
        PluginParam("wall_min_zone_price_coverage_pct", "主墙价格范围最小覆盖率（%）", default=70, min_value=10, max_value=100, step=5),
        PluginParam("wall_rectangle_support_occupancy_pct", "矩形内15%+色块总体占比（%）", default=65, min_value=20, max_value=100, step=5),
        PluginParam("wall_rectangle_zone_occupancy_pct", "矩形内30%+色块总体占比（%）", default=30, min_value=5, max_value=100, step=5),
        PluginParam("wall_current_support_occupancy_pct", "当前列矩形内15%+占比（%）", default=55, min_value=10, max_value=100, step=5),
        PluginParam("wall_rectangle_price_persistence_pct", "固定矩形价格格复现率（%）", default=60, min_value=20, max_value=100, step=5),
        PluginParam("wall_min_zone_strong_points", "主墙最少50%核心格数", default=1, min_value=0, max_value=20, step=1),
        PluginParam("wall_strongless_min_band_points", "无50%核心时至少30%主体格数", default=4, min_value=3, max_value=30, step=1),
        PluginParam("wall_min_absolute_depth", "单格最小绝对挂单量（当前深度单位）", default=0, min_value=0, max_value=1000000000, step=1),
        PluginParam("wall_min_zone_total_depth", "墙区最小绝对总挂单量（当前深度单位）", default=0, min_value=0, max_value=1000000000, step=1),
        PluginParam("wall_max_distance_bps", "墙识别最大距离（bps）", default=500, min_value=25, max_value=3000, step=25),
        PluginParam("wall_market_clearance_bins", "墙与当前价格最少间隔格", default=1, min_value=0, max_value=20, step=1),
        PluginParam("wall_max_missing_price_bins", "墙内允许浅色/空白价格格", default=1, min_value=0, max_value=10, step=1),
        PluginParam("wall_max_cluster_span_bins", "单个墙区最大跨度格", default=18, min_value=1, max_value=100, step=1),
        PluginParam("wall_history_price_tolerance_bins", "历史墙允许价格漂移格", default=2, min_value=0, max_value=10, step=1),
        PluginParam("wall_min_confirm_bars", "满足二维条件后确认K线数", default=2, min_value=1, max_value=20, step=1),
        PluginParam("wall_persistent_after_minutes", "长期墙升级时间（分钟）", default=60, min_value=1, max_value=360, step=1),
        PluginParam("wall_major_after_minutes", "主力墙升级时间（分钟）", default=240, min_value=5, max_value=1440, step=5),
        PluginParam("wall_max_missing_bars", "允许暂时变浅K线数", default=1, min_value=0, max_value=20, step=1),
        PluginParam("wall_min_match_overlap_pct", "跨帧最小价格区间重叠（%）", default=35, min_value=0, max_value=100, step=5),
        PluginParam("wall_max_center_drift_bins", "跨帧最大中心移动格", default=2, min_value=0, max_value=20, step=1),
        PluginParam("wall_min_strength_score", "墙最小显示强度", default=35, min_value=0, max_value=100, step=1),
        PluginParam("wall_max_overlay_regions", "最多显示墙生命周期", default=150, min_value=1, max_value=2000, step=10),
        PluginParam(
            name="time_aggregation",
            label="微结构模式时间精度",
            kind="select",
            default="auto_detail",
            choices=[
                {"value": "auto_detail", "label": "按可视区自动保持细节"},
                {"value": "5s", "label": "5秒"},
                {"value": "10s", "label": "10秒"},
                {"value": "15s", "label": "15秒"},
                {"value": "30s", "label": "30秒"},
                {"value": "1m", "label": "1分钟"},
                {"value": "5m", "label": "5分钟"},
                {"value": "15m", "label": "15分钟"},
            ],
        ),
        PluginParam("max_time_columns", "微结构模式最大时间列", default=2400, min_value=400, max_value=10000, step=200),
        PluginParam(
            name="display_price_step",
            label="价格合并",
            kind="select",
            default="1",
            choices=[
                {"value": "0.5", "label": "$0.5（基础数据允许时）"},
                {"value": "1", "label": "$1（推荐）"},
                {"value": "2", "label": "$2"},
                {"value": "5", "label": "$5"},
            ],
        ),
        PluginParam(
            name="color_mode",
            label="颜色方案",
            kind="select",
            default="single",
            choices=[
                {"value": "single", "label": "CoinGlass暖色单色"},
                {"value": "split", "label": "买卖盘分色"},
            ],
        ),
        PluginParam("min_intensity_pct", "后端最浅显示阈值", default=0.0, min_value=0, max_value=95, step=0.5),
        PluginParam("max_render_cells", "前端最大热力格", default=400000, min_value=10000, max_value=800000, step=10000),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        return self.run_with_context(PluginRunContext(display_df=df, visible_df=df), params)

    def run_with_context(self, context: PluginRunContext, params: dict[str, Any] | None) -> PluginRunResult:
        bars = context.visible_df
        if bars is None or bars.empty:
            return PluginRunResult(markers=[], summary={"input_rows": 0, "matched": 0, "display": "无K线数据"})
        p = params or {}
        normalization = str(p.get("normalization", "causal_max_ratio"))
        depth_unit = str(p.get("depth_unit", "base"))
        manual_max = float(p.get("manual_max", 100.0))
        color_window_hours = int(float(p.get("color_window_hours", 24)))
        color_percentile = float(p.get("color_percentile", 99.0)) / 100.0
        display_contrast_gamma = float(p.get("display_contrast_gamma", 1.4))
        books_depth = int(float(p.get("books_depth", 5000)))
        display_mode = str(p.get("display_mode", "period_end"))
        time_aggregation = str(p.get("time_aggregation", "auto_detail"))
        max_time_columns = int(float(p.get("max_time_columns", 2400)))
        display_price_step = float(p.get("display_price_step", 1.0))
        color_mode = str(p.get("color_mode", "single"))
        min_intensity = float(p.get("min_intensity_pct", 0.0)) / 100.0
        max_cells = int(float(p.get("max_render_cells", 400000)))
        large_window_hours = int(float(p.get("large_window_hours", 24)))
        large_percentile = float(p.get("large_percentile", 95.0)) / 100.0
        wall_resolution = "chart"
        wall_lookback_hours = float(p.get("wall_lookback_hours", 4.0))
        wall_min_history_bars = int(float(p.get("wall_min_history_bars", 4)))
        wall_support_time_coverage = float(p.get("wall_support_time_coverage_pct", 70.0)) / 100.0
        wall_zone_time_coverage = float(p.get("wall_zone_time_coverage_pct", 55.0)) / 100.0
        wall_core_time_coverage = float(p.get("wall_core_time_coverage_pct", 20.0)) / 100.0
        wall_min_average_ratio = float(p.get("wall_min_average_ratio_pct", 16.0)) / 100.0
        wall_min_current_ratio = float(p.get("wall_min_current_ratio_pct", 12.0)) / 100.0
        wall_strong_ratio = float(p.get("wall_strong_ratio_pct", 50.0)) / 100.0
        wall_zone_ratio = float(p.get("wall_zone_ratio_pct", 30.0)) / 100.0
        wall_support_ratio = float(p.get("wall_support_ratio_pct", 15.0)) / 100.0
        wall_isolated_point_ratio = float(p.get("wall_isolated_point_ratio_pct", 50.0)) / 100.0
        wall_min_zone_band_points = int(float(p.get("wall_min_zone_band_points", 3)))
        wall_min_zone_support_points = int(float(p.get("wall_min_zone_support_points", 4)))
        wall_min_zone_density_mass = float(p.get("wall_min_zone_density_mass", 1.2))
        wall_min_zone_price_coverage = float(p.get("wall_min_zone_price_coverage_pct", 70.0)) / 100.0
        wall_rectangle_support_occupancy = float(p.get("wall_rectangle_support_occupancy_pct", 65.0)) / 100.0
        wall_rectangle_zone_occupancy = float(p.get("wall_rectangle_zone_occupancy_pct", 30.0)) / 100.0
        wall_current_support_occupancy = float(p.get("wall_current_support_occupancy_pct", 55.0)) / 100.0
        wall_rectangle_price_persistence = float(p.get("wall_rectangle_price_persistence_pct", 60.0)) / 100.0
        wall_min_zone_strong_points = int(float(p.get("wall_min_zone_strong_points", 1)))
        wall_strongless_min_band_points = int(float(p.get("wall_strongless_min_band_points", 4)))
        wall_min_absolute_depth = float(p.get("wall_min_absolute_depth", 0.0))
        wall_min_zone_total_depth = float(p.get("wall_min_zone_total_depth", 0.0))
        wall_max_distance_bps = float(p.get("wall_max_distance_bps", 500.0))
        wall_market_clearance_bins = int(float(p.get("wall_market_clearance_bins", 1)))
        wall_max_missing_price_bins = int(float(p.get("wall_max_missing_price_bins", 1)))
        wall_max_cluster_span_bins = int(float(p.get("wall_max_cluster_span_bins", 18)))
        wall_history_price_tolerance_bins = int(float(p.get("wall_history_price_tolerance_bins", 2)))
        wall_min_confirm_bars = int(float(p.get("wall_min_confirm_bars", 2)))
        wall_persistent_after_minutes = int(float(p.get("wall_persistent_after_minutes", 60)))
        wall_major_after_minutes = int(float(p.get("wall_major_after_minutes", 240)))
        wall_max_missing_bars = int(float(p.get("wall_max_missing_bars", 1)))
        wall_min_match_overlap = float(p.get("wall_min_match_overlap_pct", 35.0)) / 100.0
        wall_max_center_drift_bins = int(float(p.get("wall_max_center_drift_bins", 2)))
        wall_min_strength_score = float(p.get("wall_min_strength_score", 35.0))
        wall_max_overlay_regions = int(float(p.get("wall_max_overlay_regions", 150)))
        timeframe = context.timeframe or "1m"
        timeframe_seconds = timeframe_to_seconds(timeframe)
        wall_resolution_seconds = timeframe_seconds
        symbol = str(context.meta.get("symbol") or context.request.get("symbol") or "ETH-USDT-SWAP")
        loader = OKXLiquidityMapLoader(symbol=symbol, books_depth=books_depth)

        start = pd.Timestamp(bars.index.min())
        end = pd.Timestamp(bars.index.max()) + pd.Timedelta(seconds=timeframe_seconds)
        visible_duration_seconds = max((end - start).total_seconds(), 1.0)
        wall_reference_warmup = pd.Timedelta(hours=max(color_window_hours, wall_lookback_hours))
        warmup_hours = max(
            large_window_hours if display_mode == "period_end" else 0,
            color_window_hours,
            int(math.ceil(wall_lookback_hours)),
        )
        bar_intervals = _bar_intervals(bars, timeframe=timeframe, warmup_hours=warmup_hours)
        load_start = start - pd.Timedelta(hours=warmup_hours)

        raw_count = 0
        coverage_days: list[str] = []
        grouped_frames: list[pd.DataFrame] = []
        wall_frames: list[pd.DataFrame] = []
        source_steps: set[float] = set()
        source_intervals: set[int] = set()
        render_seconds: int | None = None
        if display_mode == "micro_detail":
            render_seconds = None if time_aggregation == "auto_detail" else timeframe_to_seconds(time_aggregation)
        else:
            render_seconds = timeframe_seconds

        period_end_cache_hits = 0
        period_end_cache_misses = 0
        period_end_cache_paths: list[str] = []
        period_end_iterator = getattr(loader, "iter_period_end_snapshot_days", None)
        use_period_end_cache = display_mode == "period_end" and callable(period_end_iterator)

        if use_period_end_cache:
            cached_frames = period_end_iterator(
                load_start,
                end,
                timeframe=timeframe_seconds,
                price_step=display_price_step,
                project_time=True,
            )
            valid_bar_starts = set(bar_intervals["bar_start_ms"].astype("int64").tolist())
            for cached_day in cached_frames:
                if cached_day is None or cached_day.empty:
                    continue
                source_step = float(cached_day.attrs.get("source_price_step", cached_day.attrs.get("price_step", 1.0)))
                source_seconds = int(cached_day.attrs.get("source_heatmap_seconds", 60))
                source_steps.add(source_step)
                source_intervals.add(source_seconds)
                compact_day = cached_day.loc[
                    pd.to_numeric(cached_day["bucket_start_ms"], errors="coerce").astype("int64").isin(valid_bar_starts)
                ].copy()
                if compact_day.empty:
                    continue
                compact_day.attrs.update(cached_day.attrs)
                wall_frames.append(compact_day)
                grouped_day = _period_end_cached_for_bars(
                    compact_day,
                    bar_intervals,
                    depth_unit=depth_unit,
                )
                raw_count += int(cached_day.attrs.get("source_row_count", len(cached_day)))
                cache_hit = bool(cached_day.attrs.get("cache_hit"))
                period_end_cache_hits += int(cache_hit)
                period_end_cache_misses += int(not cache_hit)
                cache_path = cached_day.attrs.get("cache_path")
                if cache_path:
                    period_end_cache_paths.append(str(cache_path))
                day_label = cached_day.attrs.get("utc_day")
                if day_label:
                    coverage_days.append(str(day_label))
                if grouped_day is not None and not grouped_day.empty:
                    grouped_frames.append(grouped_day)
            render_seconds = timeframe_seconds
        else:
            iterator = getattr(loader, "iter_heatmap_days", None)
            raw_frames = iterator(load_start, end, project_time=True) if callable(iterator) else [loader.load_heatmap(load_start, end, project_time=True)]
            for raw_day in raw_frames:
                if raw_day is None or raw_day.empty:
                    continue
                source_step = float(raw_day.attrs.get("price_step", 1.0))
                source_seconds = infer_heatmap_seconds(raw_day, fallback=60)
                source_steps.add(source_step)
                source_intervals.add(source_seconds)
                effective_wall_seconds = max(source_seconds, timeframe_seconds)
                if effective_wall_seconds % source_seconds:
                    effective_wall_seconds = int(math.ceil(effective_wall_seconds / source_seconds)) * source_seconds
                wall_day = _aggregate_snapshot_for_wall(
                    raw_day,
                    source_price_step=source_step,
                    display_price_step=display_price_step,
                    target_seconds=effective_wall_seconds,
                )
                if wall_day is not None and not wall_day.empty:
                    wall_day.attrs["heatmap_seconds"] = effective_wall_seconds
                    wall_day.attrs["price_step"] = float(wall_day.attrs.get("price_step", max(source_step, display_price_step)))
                    wall_frames.append(wall_day)
                if display_mode == "period_end":
                    if timeframe_seconds < source_seconds:
                        raise ValueError(
                            f"周期末盘口显示要求K线周期不小于基础热力周期：{timeframe} < {seconds_to_timeframe(source_seconds)}"
                        )
                    grouped_day = _period_end_snapshot_for_bars(
                        raw_day,
                        bar_intervals,
                        source_price_step=source_step,
                        display_price_step=display_price_step,
                        depth_unit=depth_unit,
                    )
                else:
                    if display_mode == "time_weighted":
                        effective_seconds = timeframe_seconds
                    else:
                        effective_seconds = render_seconds
                        if effective_seconds is None:
                            effective_seconds = _choose_auto_render_seconds(
                                duration_seconds=visible_duration_seconds,
                                source_seconds=source_seconds,
                                max_columns=max_time_columns,
                            )
                            render_seconds = effective_seconds
                    if effective_seconds < source_seconds:
                        effective_seconds = source_seconds
                        render_seconds = effective_seconds
                    grouped_day = _aggregate_for_display(
                        raw_day,
                        source_price_step=source_step,
                        display_price_step=display_price_step,
                        render_seconds=int(effective_seconds),
                        depth_unit=depth_unit,
                    )
                    grouped_day["source_bucket_start_ms"] = grouped_day["bucket_start_ms"]
                    grouped_day["source_bucket_end_ms"] = grouped_day["bucket_end_ms"]
                    grouped_day["source_lag_ms"] = 0
                    grouped_day["bar_start_ms"] = grouped_day["bucket_start_ms"]
                    grouped_day["bar_end_ms"] = grouped_day["bucket_end_ms"]
                    grouped_day["bar_start"] = pd.to_datetime(grouped_day["bar_start_ms"], unit="ms", utc=True).dt.tz_convert(None) + _timezone_offset()
                    grouped_day["bar_end"] = pd.to_datetime(grouped_day["bar_end_ms"], unit="ms", utc=True).dt.tz_convert(None) + _timezone_offset()
                    grouped_day["visible"] = (
                        (grouped_day["bar_start"] >= start)
                        & (grouped_day["bar_start"] < end)
                    )
                    grouped_day["price_mid"] = (grouped_day["price_low"] + grouped_day["price_high"]) / 2.0
                raw_count += int(len(raw_day))
                day_label = raw_day.attrs.get("utc_day")
                if day_label:
                    coverage_days.append(str(day_label))
                if grouped_day is not None and not grouped_day.empty:
                    grouped_frames.append(grouped_day)

        if not grouped_frames:
            coverage = [item.__dict__ for item in loader.coverage()]
            return PluginRunResult(
                markers=[],
                summary={
                    "input_rows": int(len(bars)),
                    "matched": 0,
                    "display": f"缺少该区间的 {books_depth} 档离线流动性衍生数据",
                    "books_depth": books_depth,
                    "coverage": coverage,
                    "ui": {"compact": True, "brief_available": False, "advanced_collapsed": True, "heatmap_label": "订单簿流动性格"},
                },
            )
        if len(source_steps) > 1:
            raise ValueError(f"请求区间混用了不同 price_step: {sorted(source_steps)}")
        if len(source_intervals) > 1:
            raise ValueError(f"请求区间混用了不同基础热力周期: {sorted(source_intervals)}")

        grouped_all = pd.concat(grouped_frames, ignore_index=True)
        effective_price_step = max(display_price_step, next(iter(source_steps)))
        if display_mode == "period_end":
            # A bar spanning two UTC day files can be seen by both daily
            # iterators. Keep only the globally latest completed source
            # snapshot for that bar. Filtering by snapshot first is essential:
            # deduplicating per price would incorrectly carry a level from an
            # earlier snapshot after it had disappeared in the final snapshot.
            latest_source_end = grouped_all.groupby("bar_start_ms", observed=True)["source_bucket_end_ms"].transform("max")
            grouped_all = grouped_all.loc[grouped_all["source_bucket_end_ms"] == latest_source_end].copy()
            grouped_all = (
                grouped_all.sort_values(["bar_start_ms", "source_bucket_end_ms", "side_code", "price_index"])
                .drop_duplicates(["bar_start_ms", "source_bucket_end_ms", "side_code", "price_index"], keep="last")
                .reset_index(drop=True)
            )
            grouped_all = _add_rolling_large_threshold(
                grouped_all,
                window_hours=large_window_hours,
                percentile=large_percentile,
            )
            grouped = grouped_all.loc[grouped_all["visible"].astype(bool)].copy()
        else:
            grouped_all = _attach_distance_context(grouped_all)
            grouped_all["rolling_large_threshold"] = np.nan
            grouped_all["is_large_rolling"] = False
            grouped = grouped_all.loc[grouped_all["visible"].astype(bool)].copy()
        grouped.attrs["display_price_step"] = effective_price_step
        grouped.attrs["render_seconds"] = int(render_seconds or timeframe_seconds)

        # Calculate the causal colour/depth scale once. The period-end wall
        # detector reuses these exact arrays instead of repeating the 24h
        # normalisation over the same rows.
        intensity_all, cap, color_caps = _normalize_depth(
            grouped_all,
            mode=normalization,
            manual_max=manual_max,
            rolling_window_hours=color_window_hours,
            rolling_percentile=color_percentile,
            display_contrast_gamma=display_contrast_gamma,
            return_caps=True,
        )
        grouped_all["intensity"] = intensity_all
        grouped_all["causal_color_cap"] = color_caps

        wall_regions: list[PriceRegion] = []
        detected_walls = []
        if display_mode == "period_end" or wall_frames:
            if display_mode == "period_end":
                # Same final/latest snapshot matrix as the display, including
                # the already computed causal ratio/reference.
                wall_input = grouped_all.copy()
                if "bucket_start_ms" not in wall_input.columns:
                    wall_input["bucket_start_ms"] = wall_input["bar_start_ms"]
                if "bucket_end_ms" not in wall_input.columns:
                    wall_input["bucket_end_ms"] = wall_input["bar_end_ms"]
                if normalization == "causal_max_ratio":
                    wall_input["depth_ratio"] = pd.to_numeric(wall_input["intensity"], errors="coerce").fillna(0.0)
                    wall_input["reference_depth"] = pd.to_numeric(wall_input["causal_color_cap"], errors="coerce").fillna(1e-12)
                    wall_input["snapshot_high_depth"] = wall_input["reference_depth"]
                wall_input.attrs["heatmap_seconds"] = timeframe_seconds
                wall_input.attrs["price_step"] = effective_price_step
            else:
                wall_input = (
                    pd.concat(wall_frames, ignore_index=True)
                    .sort_values(["bucket_start_ms", "side_code", "price_index"])
                    .drop_duplicates(["bucket_start_ms", "side_code", "price_index"], keep="last")
                    .reset_index(drop=True)
                )
                wall_input.attrs["heatmap_seconds"] = int(wall_frames[0].attrs.get("heatmap_seconds", timeframe_seconds))
                wall_input.attrs["price_step"] = float(wall_frames[0].attrs.get("price_step", effective_price_step))
            wall_start_ms = _project_naive_to_utc_ms(start - wall_reference_warmup)
            wall_input = wall_input.loc[wall_input["bucket_end_ms"] >= wall_start_ms].copy()
            wall_config = PersistentWallConfig(
                reference_window_hours=color_window_hours,
                reference_snapshot_quantile=color_percentile,
                strong_depth_ratio=wall_strong_ratio,
                zone_depth_ratio=wall_zone_ratio,
                support_depth_ratio=wall_support_ratio,
                isolated_point_ratio=wall_isolated_point_ratio,
                lookback_hours=wall_lookback_hours,
                minimum_history_bars=wall_min_history_bars,
                minimum_support_time_coverage=wall_support_time_coverage,
                minimum_zone_time_coverage=wall_zone_time_coverage,
                minimum_core_time_coverage=wall_core_time_coverage,
                minimum_average_depth_ratio=wall_min_average_ratio,
                minimum_current_depth_ratio=wall_min_current_ratio,
                minimum_zone_band_points=wall_min_zone_band_points,
                minimum_zone_support_points=wall_min_zone_support_points,
                minimum_zone_density_mass=wall_min_zone_density_mass,
                minimum_zone_price_coverage=wall_min_zone_price_coverage,
                minimum_rectangle_support_occupancy=wall_rectangle_support_occupancy,
                minimum_rectangle_zone_occupancy=wall_rectangle_zone_occupancy,
                minimum_current_support_occupancy=wall_current_support_occupancy,
                rectangle_price_persistence=wall_rectangle_price_persistence,
                minimum_zone_strong_points=wall_min_zone_strong_points,
                strongless_zone_min_band_points=wall_strongless_min_band_points,
                minimum_absolute_depth=wall_min_absolute_depth,
                minimum_zone_total_depth=wall_min_zone_total_depth,
                maximum_distance_bps=wall_max_distance_bps,
                minimum_market_clearance_bins=wall_market_clearance_bins,
                maximum_missing_price_bins=wall_max_missing_price_bins,
                maximum_cluster_span_bins=wall_max_cluster_span_bins,
                history_price_tolerance_bins=wall_history_price_tolerance_bins,
                minimum_confirm_bars=wall_min_confirm_bars,
                persistent_after_minutes=wall_persistent_after_minutes,
                major_after_minutes=wall_major_after_minutes,
                maximum_missing_bars=wall_max_missing_bars,
                minimum_match_overlap=wall_min_match_overlap,
                maximum_center_drift_bins=wall_max_center_drift_bins,
                minimum_strength_score=wall_min_strength_score,
                maximum_walls=max(wall_max_overlay_regions * 4, wall_max_overlay_regions),
            )
            wall_depth_column = "end_depth_usd" if depth_unit == "usd" else "end_depth_base"
            detected_walls = detect_persistent_liquidity_walls(
                wall_input,
                depth_column=wall_depth_column,
                config=wall_config,
            )
            detected_walls = _select_dominant_walls(
                detected_walls,
                max_regions=wall_max_overlay_regions,
            )
            visible_start_ms = _project_naive_to_utc_ms(start)
            visible_end_ms = _project_naive_to_utc_ms(end)
            for wall in detected_walls:
                side_label = "买" if wall.side == "bid" else "卖"
                type_label = "单点墙口" if wall.wall_type == "POINT" else "主墙"
                rectangle_low = float(wall.fields.get("rectangle_price_low", wall.price_low))
                rectangle_high = float(wall.fields.get("rectangle_price_high", wall.price_high))
                rectangle_start_ms = max(int(wall.confirmed_at_ms), visible_start_ms)
                rectangle_end_ms = min(int(wall.end_ms), visible_end_ms)
                if rectangle_end_ms <= rectangle_start_ms or rectangle_high <= rectangle_low:
                    continue

                rectangle_start = pd.to_datetime(
                    rectangle_start_ms, unit="ms", utc=True
                ).tz_convert(None) + _timezone_offset()
                rectangle_end = pd.to_datetime(
                    rectangle_end_ms, unit="ms", utc=True
                ).tz_convert(None) + _timezone_offset()

                # A resting wall is a barrier ahead of price, never a box that
                # wraps around candles.  End it at the first chart bar whose
                # traded range touches/enters the rectangle.  Historical bars
                # use completed OHLC; the live bar uses its current high/low.
                price_path = bars.loc[
                    (bars.index >= rectangle_start) & (bars.index < rectangle_end)
                ]
                if not price_path.empty:
                    if wall.side == "bid":
                        touched = pd.to_numeric(price_path["low"], errors="coerce") <= rectangle_high + 1e-12
                    else:
                        touched = pd.to_numeric(price_path["high"], errors="coerce") >= rectangle_low - 1e-12
                    if bool(touched.any()):
                        first_touch = pd.Timestamp(touched[touched].index[0])
                        rectangle_end = min(rectangle_end, first_touch)
                if rectangle_end <= rectangle_start:
                    continue

                lifecycle_stage = str(wall.fields.get("lifecycle_stage") or "STABLE")
                wall_regions.append(
                    PriceRegion(
                        start_timestamp=rectangle_start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_timestamp=rectangle_end.strftime("%Y-%m-%d %H:%M:%S"),
                        price_low=rectangle_low,
                        price_high=rectangle_high,
                        label=f"{side_label}{type_label}",
                        color="#00AEEF",
                        opacity=0.028,
                        border_width=2.4 if wall.wall_type == "MAIN" else 2.0,
                        side=wall.side,
                        status=lifecycle_stage,
                        fields={
                            "wall_id": wall.wall_id,
                            "wall_type": wall.wall_type,
                            "detector_version": wall.fields.get("detector_version"),
                            "first_seen_ms": wall.first_seen_ms,
                            "confirmed_at_ms": wall.confirmed_at_ms,
                            "duration_minutes": wall.duration_minutes,
                            "time_coverage": wall.time_coverage,
                            "price_coverage": wall.price_coverage,
                            "strength_score": wall.strength_score,
                            "rectangle_price_low": rectangle_low,
                            "rectangle_price_high": rectangle_high,
                            "rectangle_price_indices": wall.fields.get("rectangle_price_indices"),
                            "rectangle_price_persistence": wall.fields.get("rectangle_price_persistence"),
                            "minimum_market_clearance_bins": wall.fields.get("minimum_market_clearance_bins"),
                            "rectangular_wall": True,
                            "terminated_at_price_touch": rectangle_end < pd.to_datetime(
                                rectangle_end_ms, unit="ms", utc=True
                            ).tz_convert(None) + _timezone_offset(),
                            "show_label": False,
                        },
                    )
                )


        grouped = grouped_all.loc[grouped_all["visible"].astype(bool)].copy()
        grouped.attrs["display_price_step"] = effective_price_step
        grouped.attrs["render_seconds"] = int(render_seconds or timeframe_seconds)
        grouped, applied_threshold = _reduce_cells(grouped, min_intensity=min_intensity, max_cells=max_cells)
        unit_label = "USD" if depth_unit == "usd" else "ETH"
        warm_color = "#f97316"
        compact_heatmap: dict[str, Any] | None = None
        heatmap: list[PriceHeatmapCell] = []
        # Large object-per-cell payloads can exceed hundreds of MB because every
        # cell repeats timestamps, labels and a fields dictionary.  Keep the
        # legacy objects for small responses/tests, and use compact columnar
        # JSON for real multi-day charts.
        if len(grouped) > 50_000:
            compact_heatmap = _compact_heatmap_payload(
                grouped,
                depth_unit=depth_unit,
                color_mode=color_mode,
                display_price_step=effective_price_step,
            )
        else:
            for row in grouped.itertuples(index=False):
                side = str(row.side)
                color = warm_color if color_mode == "single" else ("#22d3ee" if side == "bid" else "#fb7185")
                strength = float(row.intensity)
                rolling_threshold = float(row.rolling_large_threshold) if pd.notna(row.rolling_large_threshold) else math.nan
                is_large = bool(row.is_large_rolling)
                label = "买盘流动性" if side == "bid" else "卖盘流动性"
                if is_large:
                    label += f" · 过去{large_window_hours}h P{large_percentile * 100:g}+"
                start_ts = pd.Timestamp(row.bar_start) if hasattr(row, "bar_start") else pd.to_datetime(int(row.bucket_start_ms), unit="ms", utc=True).tz_convert(None) + _timezone_offset()
                end_ts = pd.Timestamp(row.bar_end) if hasattr(row, "bar_end") else pd.to_datetime(int(row.bucket_end_ms), unit="ms", utc=True).tz_convert(None) + _timezone_offset()
                heatmap.append(
                    PriceHeatmapCell(
                        start_timestamp=start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        end_timestamp=end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        price_low=float(row.price_low),
                        price_high=float(row.price_high),
                        intensity=float(row.intensity),
                        side=side,
                        color=color,
                        label=label,
                        confidence=1.0,
                        fields={
                            "depth": float(row.display_depth),
                            "unit": unit_label,
                            "local_ratio": float(row.local_ratio),
                            "causal_depth_ratio": strength,
                            "order_count": float(row.display_order_count),
                            "price_mid": float(row.price_mid),
                            "distance_bps": None if pd.isna(row.distance_bps) else float(row.distance_bps),
                            "distance_band": None if pd.isna(row.distance_band) else str(row.distance_band),
                            "causal_color_cap": float(row.causal_color_cap),
                            "color_window_hours": color_window_hours,
                            "color_percentile": color_percentile * 100.0,
                            "color_reference_semantics": "rolling max of per-snapshot robust high quantile" if normalization == "causal_max_ratio" else "legacy percentile/manual",
                            "rolling_large_threshold": None if not math.isfinite(rolling_threshold) else rolling_threshold,
                            "rolling_window_hours": large_window_hours,
                            "rolling_percentile": large_percentile * 100.0,
                            "is_large_rolling": is_large,
                            "source_snapshot_start": (pd.to_datetime(int(row.source_bucket_start_ms), unit="ms", utc=True).tz_convert(None) + _timezone_offset()).strftime("%Y-%m-%d %H:%M:%S"),
                            "source_snapshot_end": (pd.to_datetime(int(row.source_bucket_end_ms), unit="ms", utc=True).tz_convert(None) + _timezone_offset()).strftime("%Y-%m-%d %H:%M:%S"),
                            "source_lag_ms": int(row.source_lag_ms),
                            "added_base": float(row.added_base),
                            "removed_base": float(row.removed_base),
                            "executed_base": float(row.executed_base),
                            "cancelled_base": float(row.cancelled_base),
                            "consumed_base": float(row.consumed_base),
                            "replenished_base": float(row.replenished_base),
                        },
                    )
                )

        aligned = _align_features_to_bars(loader, bars)
        n = len(bars)
        if aligned.empty:
            aligned = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(bars.index)))
        def col(name: str, default: float = np.nan) -> pd.Series:
            if name not in aligned:
                return pd.Series(default, index=aligned.index, dtype=float)
            return pd.to_numeric(aligned[name], errors="coerce")

        imbalance = col("depth_imbalance_25bps")
        valid = col("book_valid", 0).fillna(0).astype(int)
        bid_price = col("nearest_large_bid_price").combine_first(col("top_bid_wall_price"))
        ask_price = col("nearest_large_ask_price").combine_first(col("top_ask_wall_price"))
        bid_depth = col("nearest_large_bid_depth_base", 0) if "nearest_large_bid_depth_base" in aligned.columns else col("top_bid_wall_depth_base", 0)
        ask_depth = col("nearest_large_ask_depth_base", 0) if "nearest_large_ask_depth_base" in aligned.columns else col("top_ask_wall_depth_base", 0)
        buy_flow, sell_flow = col("aggressive_buy_base", 0), col("aggressive_sell_base", 0)
        bid_cancel, ask_cancel = col("estimated_bid_cancel_base", 0), col("estimated_ask_cancel_base", 0)
        bid_consume, ask_consume = col("estimated_bid_consumed_base", 0), col("estimated_ask_consumed_base", 0)
        bid_replenish, ask_replenish = col("estimated_bid_replenished_base", 0), col("estimated_ask_replenished_base", 0)

        directions, phases, processes, reason1, reason2, reason3, advice = [], [], [], [], [], [], []
        for i in range(n):
            if i >= len(aligned) or int(valid.iloc[i] or 0) == 0:
                directions.append("盘口数据暂不可用")
                phases.append("等待有效快照")
                processes.append("无可用状态")
                reason1.append("该时刻没有满足 available_time 的有效盘口状态")
                reason2.append("热力图展示层与策略可用时间分开处理")
                reason3.append("不使用未来快照修复缺口")
                advice.append("跳过该时刻的流动性条件")
                continue
            im = float(imbalance.iloc[i]) if math.isfinite(float(imbalance.iloc[i])) else 0.0
            directions.append("买盘更厚" if im > 0.12 else "卖盘更厚" if im < -0.12 else "近端深度均衡")
            phases.append(
                f"买墙 {_fmt_number(bid_price.iloc[i])} / {_fmt_number(bid_depth.iloc[i])} ETH；"
                f"卖墙 {_fmt_number(ask_price.iloc[i])} / {_fmt_number(ask_depth.iloc[i])} ETH"
            )
            cancel_total = float(bid_cancel.iloc[i] + ask_cancel.iloc[i])
            consume_total = float(bid_consume.iloc[i] + ask_consume.iloc[i])
            replenish_total = float(bid_replenish.iloc[i] + ask_replenish.iloc[i])
            if consume_total > cancel_total * 1.5 and consume_total > 0:
                process = "挂单以真实成交消耗为主"
            elif cancel_total > consume_total * 1.5 and cancel_total > 0:
                process = "挂单撤走为主"
            elif replenish_total > max(consume_total, cancel_total) and replenish_total > 0:
                process = "成交后补单较强"
            else:
                process = "暂无主导流动性过程"
            processes.append(process)
            reason1.append(f"25bps深度不平衡 {im:+.3f}")
            reason2.append(f"主动买/卖 {_fmt_number(buy_flow.iloc[i])}/{_fmt_number(sell_flow.iloc[i])} ETH")
            reason3.append(f"撤单/消耗/补单 {_fmt_number(cancel_total)}/{_fmt_number(consume_total)}/{_fmt_number(replenish_total)} ETH")
            advice.append("显示默认按精确周期末盘口；墙识别读取每根K线的末快照；实盘当前列读取最新盘口。")

        row_fields = {
            "brief_direction": _categorical(directions),
            "brief_phase": _categorical(phases),
            "brief_process": _categorical(processes),
            "brief_reason_1": _categorical(reason1),
            "brief_reason_2": _categorical(reason2),
            "brief_reason_3": _categorical(reason3),
            "brief_advice": _categorical(advice),
            "brief_context_detail": _categorical(directions),
            "book_valid": _values(valid.reset_index(drop=True), n),
            "spread_bps": _values(col("spread_bps").reset_index(drop=True), n),
            "depth_imbalance_25bps": _values(imbalance.reset_index(drop=True), n),
            "top_bid_wall_price": _values(col("top_bid_wall_price").reset_index(drop=True), n),
            "top_bid_wall_depth_base": _values(col("top_bid_wall_depth_base", 0).reset_index(drop=True), n),
            "top_ask_wall_price": _values(col("top_ask_wall_price").reset_index(drop=True), n),
            "top_ask_wall_depth_base": _values(col("top_ask_wall_depth_base", 0).reset_index(drop=True), n),
            "aggressive_buy_base": _values(buy_flow.reset_index(drop=True), n),
            "aggressive_sell_base": _values(sell_flow.reset_index(drop=True), n),
            "estimated_bid_cancel_base": _values(bid_cancel.reset_index(drop=True), n),
            "estimated_ask_cancel_base": _values(ask_cancel.reset_index(drop=True), n),
            "estimated_bid_consumed_base": _values(bid_consume.reset_index(drop=True), n),
            "estimated_ask_consumed_base": _values(ask_consume.reset_index(drop=True), n),
            "estimated_bid_replenished_base": _values(bid_replenish.reset_index(drop=True), n),
            "estimated_ask_replenished_base": _values(ask_replenish.reset_index(drop=True), n),
        }

        source_seconds = int(next(iter(source_intervals)))
        if display_mode == "period_end":
            alignment = _alignment_audit(
                grouped_all,
                bars,
                loader,
                source_seconds=source_seconds,
                display_price_step=effective_price_step,
            )
        else:
            alignment = {
                "status": "not_applicable",
                "reason": "price/bin alignment audit applies to exact period-end display mode",
            }
        normalization_text = {
            "causal_max_ratio": (
                f"过去{color_window_hours}h因果最高挂单量比例；"
                f"典型100%参考={cap:,.2f} {unit_label}"
            ),
            "log_depth": (
                f"过去{color_window_hours}h因果滚动P{color_percentile * 100:g}对数·经典对比{display_contrast_gamma:g}；"
                f"典型封顶={cap:,.2f} {unit_label}"
            ),
            "salience": (
                f"过去{color_window_hours}h因果滚动显著性；"
                f"典型封顶={cap:,.2f} {unit_label}"
            ),
            "auto_window": (
                f"过去{color_window_hours}h因果滚动P{color_percentile * 100:g}线性；"
                f"典型封顶={cap:,.2f} {unit_label}"
            ),
            "manual": f"手动封顶={cap:,.2f} {unit_label}",
            "local_ratio": "每个时刻同侧最厚挂单=100%（临时观察）",
        }.get(normalization, normalization)
        mode_text = {
            "period_end": "一根K线一列·周期末盘口",
            "time_weighted": "一根K线一列·时间加权均值",
            "micro_detail": f"微结构 {seconds_to_timeframe(int(render_seconds or source_seconds))}",
        }.get(display_mode, display_mode)
        coverage_days = sorted(set(coverage_days))
        return PluginRunResult(
            markers=[],
            heatmap=heatmap,
            heatmap_compact=compact_heatmap,
            price_regions=wall_regions,
            row_fields=row_fields,
            summary={
                "input_rows": int(len(bars)),
                "matched": int(valid.sum()),
                "display": (
                    f"{books_depth}档 · {mode_text} · 热力格 {len(grouped):,}；{normalization_text}；"
                    f"价格格 [low, high)=${effective_price_step:g}；墙 {len(wall_regions)} 个；对齐审计={alignment.get('status')}"
                ),
                "source_heatmap_cells": int(raw_count),
                "display_heatmap_cells": int(len(grouped)),
                "heatmap_payload_mode": "compact_v1" if compact_heatmap is not None else "objects",
                "period_end_cache_enabled": bool(use_period_end_cache),
                "period_end_cache_hits": int(period_end_cache_hits),
                "period_end_cache_misses": int(period_end_cache_misses),
                "period_end_cache_paths": sorted(set(period_end_cache_paths)),
                "applied_min_intensity": float(applied_threshold),
                "display_mode": display_mode,
                "aggregation_mode": time_aggregation,
                "books_depth": books_depth,
                "source_heatmap_seconds": source_seconds,
                "effective_render_seconds": int(render_seconds or timeframe_seconds),
                "effective_timeframe": seconds_to_timeframe(int(render_seconds or timeframe_seconds)),
                "display_price_step": float(effective_price_step),
                "per_timeframe_artifact_required": False,
                "color_window_hours": color_window_hours,
                "color_percentile": color_percentile * 100.0,
                "color_reference_semantics": "rolling max of per-snapshot robust high quantile" if normalization == "causal_max_ratio" else "legacy percentile/manual",
                "color_normalization_causal": normalization in {"causal_max_ratio", "log_depth", "auto_window", "salience"},
                "color_uses_full_request_interval": False,
                "large_window_hours": large_window_hours,
                "large_percentile": large_percentile * 100.0,
                "persistent_wall_count": int(len(detected_walls)),
                "persistent_zone_wall_count": int(sum(wall.wall_type == "MAIN" for wall in detected_walls)),
                "persistent_point_wall_count": int(sum(wall.wall_type == "POINT" for wall in detected_walls)),
                "persistent_wall_overlay_regions": int(len(wall_regions)),
                "wall_detector_version": "v2_5_4_strict_rectangular_market_side",
                "wall_strong_ratio_pct": wall_strong_ratio * 100.0,
                "wall_zone_ratio_pct": wall_zone_ratio * 100.0,
                "wall_support_ratio_pct": wall_support_ratio * 100.0,
                "wall_isolated_point_ratio_pct": wall_isolated_point_ratio * 100.0,
                "wall_min_zone_band_points": wall_min_zone_band_points,
                "wall_min_zone_support_points": wall_min_zone_support_points,
                "wall_min_zone_density_mass": wall_min_zone_density_mass,
                "wall_min_zone_price_coverage_pct": wall_min_zone_price_coverage * 100.0,
                "wall_rectangle_support_occupancy_pct": wall_rectangle_support_occupancy * 100.0,
                "wall_rectangle_zone_occupancy_pct": wall_rectangle_zone_occupancy * 100.0,
                "wall_current_support_occupancy_pct": wall_current_support_occupancy * 100.0,
                "wall_rectangle_price_persistence_pct": wall_rectangle_price_persistence * 100.0,
                "wall_market_clearance_bins": wall_market_clearance_bins,
                "wall_shape": "fixed_rectangle",
                "wall_price_relation": "bid below traded range; ask above traded range",
                "wall_min_absolute_depth": wall_min_absolute_depth,
                "wall_min_zone_total_depth": wall_min_zone_total_depth,
                "wall_resolution": timeframe,
                "wall_input_semantics": "historical bar final snapshot; live bar latest snapshot",
                "wall_lookback_hours": wall_lookback_hours,
                "wall_min_history_bars": wall_min_history_bars,
                "wall_min_confirm_bars": wall_min_confirm_bars,
                "wall_max_missing_bars": wall_max_missing_bars,
                "wall_max_missing_price_bins": wall_max_missing_price_bins,
                "alignment_audit": alignment,
                "coverage_days": coverage_days,
                "ui": {
                    "compact": True,
                    "brief_available": True,
                    "advanced_collapsed": True,
                    "heatmap_label": "订单簿流动性格",
                    "heatmap_hover": True,
                    "heatmap_color_controls": True,
                    "heatmap_color_min_pct": 0,
                    "heatmap_color_max_pct": 50,
                    "heatmap_detail_card": True,
                    "wall_overlay_control": True,
                    "wall_overlay_label": "墙",
                    "wall_overlay_default": False,
                    "brief_disclaimer": (
                        "历史热力列和墙检测都只使用每根K线结束前最后一个有效盘口；实盘当前列使用最新盘口持续覆盖。"
                        "颜色和墙共用过去24小时因果稳健高位挂单比例，不使用查询区间未来数据。"
                        "深蓝矩形墙来自连续多根K线的价格×时间矩阵：矩形内部必须具有足够色块填充度；买墙只能位于价格下方，卖墙只能位于价格上方，价格触碰后墙结束。"
                    ),
                },
            },
        )
