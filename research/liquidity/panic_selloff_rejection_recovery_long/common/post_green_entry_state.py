#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal post-green early-path state machines.

The green bar is only an observation checkpoint. Every delayed decision is made
from closed bars after green and executes on the following bar open. The module
never uses the eventual post-green outcome class to generate an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EntryModel:
    name: str
    family: str
    description: str
    mode: str
    max_wait_bars: int
    require_flow: bool


ENTRY_MODELS: tuple[EntryModel, ...] = (
    EntryModel(
        "GREEN_NEXT_OPEN",
        "baseline",
        "Green closed bar -> next open, no post-green confirmation.",
        "baseline",
        0,
        False,
    ),
    EntryModel(
        "CONTINUATION_PRICE_3B",
        "immediate_continuation",
        "Within three closed bars: break green high with positive R progress.",
        "continuation",
        3,
        False,
    ),
    EntryModel(
        "CONTINUATION_FLOW_3B",
        "immediate_continuation",
        "Continuation price trigger plus short ordinary/large flow confirmation.",
        "continuation",
        3,
        True,
    ),
    EntryModel(
        "PULLBACK_RECLAIM_PRICE_10B",
        "pullback_reclaim",
        "Arm on a controlled pullback, then reclaim prior high without breaking purple stop.",
        "pullback",
        10,
        False,
    ),
    EntryModel(
        "PULLBACK_RECLAIM_FLOW_10B",
        "pullback_reclaim",
        "Controlled pullback/reclaim plus order-flow recovery and sell-intensity decay.",
        "pullback",
        10,
        True,
    ),
    EntryModel(
        "ADAPTIVE_PRICE_10B",
        "adaptive",
        "Take the first causal continuation or controlled pullback-reclaim trigger.",
        "adaptive",
        10,
        False,
    ),
    EntryModel(
        "ADAPTIVE_FLOW_10B",
        "adaptive",
        "Take the first causal trigger only when ordinary/large flow confirms.",
        "adaptive",
        10,
        True,
    ),
)


DIAGNOSTIC_WINDOWS: tuple[int, ...] = (1, 3, 5, 10)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-12:
        return np.nan
    return float(num / den)


def _segment_ratio(frame: pd.DataFrame, num_col: str, den_col: str) -> float:
    if frame.empty or num_col not in frame.columns or den_col not in frame.columns:
        return np.nan
    num = pd.to_numeric(frame[num_col], errors="coerce").sum(min_count=1)
    den = pd.to_numeric(frame[den_col], errors="coerce").sum(min_count=1)
    return _safe_ratio(_finite(num), _finite(den))


def _mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    return _finite(pd.to_numeric(frame[col], errors="coerce").mean())


def _max(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    return _finite(pd.to_numeric(frame[col], errors="coerce").max())


def _last(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame.columns:
        return np.nan
    return _finite(pd.to_numeric(frame[col], errors="coerce").iloc[-1])


def _decision_snapshot(
    price_path: pd.DataFrame,
    flow_path: pd.DataFrame,
    *,
    green_close: float,
    green_high: float,
    risk: float,
    pullback_low: float,
    bar_offset: int,
) -> dict[str, Any]:
    close_now = _finite(price_path["close"].iloc[-1])
    high_now = _finite(price_path["high"].iloc[-1])
    low_now = _finite(price_path["low"].iloc[-1])
    min_low = _finite(pd.to_numeric(price_path["low"], errors="coerce").min())
    max_high = _finite(pd.to_numeric(price_path["high"], errors="coerce").max())
    prev_high = _finite(price_path["high"].iloc[-2], green_high) if len(price_path) >= 2 else green_high
    pullback_depth_r = _safe_ratio(green_close - min_low, risk)
    progress_r = _safe_ratio(close_now - green_close, risk)
    rebound_r = _safe_ratio(close_now - pullback_low, risk)
    return {
        "decision_bar_offset": int(bar_offset),
        "decision_close": close_now,
        "decision_high": high_now,
        "decision_low": low_now,
        "decision_close_progress_r": progress_r,
        "decision_max_progress_r": _safe_ratio(max_high - green_close, risk),
        "decision_pullback_depth_r": pullback_depth_r,
        "decision_rebound_from_pullback_r": rebound_r,
        "decision_close_above_green": bool(close_now >= green_close),
        "decision_close_above_green_high": bool(close_now >= green_high),
        "decision_close_above_prev_high": bool(close_now > prev_high),
        "decision_close_pos": _last(flow_path, "close_pos"),
        "decision_delta_ratio": _segment_ratio(flow_path, "delta_notional", "notional"),
        "decision_large_delta_ratio": _segment_ratio(flow_path, "large_delta_notional", "large_notional"),
        "decision_delta_ratio_2": _last(flow_path, "delta_ratio_2"),
        "decision_large_delta_ratio_2": _last(flow_path, "large_delta_ratio_2"),
        "decision_taker_buy_ratio_2": _last(flow_path, "taker_buy_ratio_2"),
        "decision_sell_intensity_last": _last(flow_path, "sell_notional_ratio_base"),
        "decision_sell_intensity_mean": _mean(flow_path, "sell_notional_ratio_base"),
        "decision_sell_intensity_peak": _max(flow_path, "sell_notional_ratio_base"),
        "decision_absorption_last": _last(flow_path, "absorption_score"),
        "decision_large_trade_share_last": _last(flow_path, "large_trade_share"),
        "decision_price_return": _safe_ratio(close_now - green_close, green_close),
    }


def _flow_confirmation(snapshot: dict[str, Any], *, pullback_peak_sell: float = np.nan) -> bool:
    delta2 = _finite(snapshot.get("decision_delta_ratio_2"))
    large2 = _finite(snapshot.get("decision_large_delta_ratio_2"))
    taker2 = _finite(snapshot.get("decision_taker_buy_ratio_2"))
    sell_last = _finite(snapshot.get("decision_sell_intensity_last"))
    sell_ok = True
    if np.isfinite(pullback_peak_sell) and pullback_peak_sell > 0 and np.isfinite(sell_last):
        sell_ok = sell_last <= max(1.10, 0.90 * pullback_peak_sell)
    return bool(
        np.isfinite(delta2)
        and delta2 >= 0.0
        and (not np.isfinite(large2) or large2 >= -0.10)
        and (not np.isfinite(taker2) or taker2 >= 0.50)
        and sell_ok
    )


def _continuation_trigger(snapshot: dict[str, Any], *, require_flow: bool) -> bool:
    price_ok = bool(
        snapshot["decision_close_above_green_high"]
        and _finite(snapshot["decision_close_progress_r"], -np.inf) >= 0.08
        and _finite(snapshot["decision_pullback_depth_r"], np.inf) <= 0.45
        and _finite(snapshot["decision_close_pos"], 0.5) >= 0.55
    )
    return price_ok and (not require_flow or _flow_confirmation(snapshot))


def _pullback_trigger(
    snapshot: dict[str, Any],
    *,
    pullback_armed: bool,
    require_flow: bool,
    pullback_peak_sell: float,
) -> bool:
    price_ok = bool(
        pullback_armed
        and snapshot["decision_close_above_prev_high"]
        and _finite(snapshot["decision_rebound_from_pullback_r"], -np.inf) >= 0.22
        and _finite(snapshot["decision_close_progress_r"], -np.inf) >= -0.18
        and _finite(snapshot["decision_close_pos"], 0.5) >= 0.55
    )
    return price_ok and (
        not require_flow
        or _flow_confirmation(snapshot, pullback_peak_sell=pullback_peak_sell)
    )


def _flow_resurge_abort(snapshot: dict[str, Any], *, bar_offset: int) -> bool:
    if bar_offset < 2:
        return False
    return bool(
        _finite(snapshot.get("decision_close_progress_r"), 0.0) <= -0.45
        and _finite(snapshot.get("decision_delta_ratio_2"), 0.0) <= -0.20
        and _finite(snapshot.get("decision_sell_intensity_last"), 0.0) >= 1.50
    )


def _diagnostic_window_row(
    price: pd.DataFrame,
    flow: pd.DataFrame,
    *,
    green_close: float,
    risk: float,
    window: int,
) -> dict[str, Any]:
    if len(price) < window:
        return {
            f"diag_post_{window}b_complete": False,
            f"diag_post_{window}b_close_r": np.nan,
            f"diag_post_{window}b_mfe_r": np.nan,
            f"diag_post_{window}b_mae_r": np.nan,
            f"diag_post_{window}b_delta_ratio": np.nan,
            f"diag_post_{window}b_large_delta_ratio": np.nan,
            f"diag_post_{window}b_sell_intensity_mean": np.nan,
        }
    p = price.iloc[:window]
    f = flow.iloc[:window]
    close_end = _finite(p["close"].iloc[-1])
    return {
        f"diag_post_{window}b_complete": True,
        f"diag_post_{window}b_close_r": _safe_ratio(close_end - green_close, risk),
        f"diag_post_{window}b_mfe_r": _safe_ratio(_finite(p["high"].max()) - green_close, risk),
        f"diag_post_{window}b_mae_r": _safe_ratio(_finite(p["low"].min()) - green_close, risk),
        f"diag_post_{window}b_delta_ratio": _segment_ratio(f, "delta_notional", "notional"),
        f"diag_post_{window}b_large_delta_ratio": _segment_ratio(f, "large_delta_notional", "large_notional"),
        f"diag_post_{window}b_sell_intensity_mean": _mean(f, "sell_notional_ratio_base"),
    }


def build_post_green_diagnostics_and_decisions(
    bars: pd.DataFrame,
    orderflow: pd.DataFrame,
    green_signals: pd.DataFrame,
    *,
    stop_buffer_pct: float = 0.0005,
    max_wait_bars: int = 10,
    diagnostic_windows: Sequence[int] = DIAGNOSTIC_WINDOWS,
    progress_enabled: bool = True,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return signal diagnostics, causal entries and no-entry funnel rows."""
    if green_signals.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    bars = bars.sort_index()
    orderflow = orderflow.reindex(bars.index)
    index_pos = pd.Series(np.arange(len(bars), dtype=int), index=bars.index)
    diagnostics: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    funnel: list[dict[str, Any]] = []

    reporter = None
    if ProgressReporter is not None:
        reporter = ProgressReporter(
            "[post-green] causal state machines",
            len(green_signals),
            every=max(1, int(progress_every)),
            enabled=progress_enabled,
        )

    for done, signal in enumerate(green_signals.itertuples(index=False), start=1):
        green_time = pd.Timestamp(signal.event_time)
        if green_time not in index_pos.index:
            if reporter is not None:
                reporter.update(done)
            continue
        green_pos = int(index_pos.at[green_time])
        green_close = _finite(bars.at[green_time, "close"])
        green_high = _finite(bars.at[green_time, "high"])
        episode_low = _finite(signal.episode_low)
        stop_price = episode_low * (1.0 - float(stop_buffer_pct))
        risk = green_close - stop_price
        if not np.isfinite(risk) or risk <= green_close * 1e-6:
            if reporter is not None:
                reporter.update(done)
            continue

        end_pos = min(len(bars) - 1, green_pos + int(max_wait_bars))
        post_price = bars.iloc[green_pos + 1 : end_pos + 1]
        post_flow = orderflow.iloc[green_pos + 1 : end_pos + 1]
        diag: dict[str, Any] = {
            "episode_id": int(signal.episode_id),
            "green_time": green_time,
            "green_close": green_close,
            "green_high": green_high,
            "episode_low": episode_low,
            "stop_price": stop_price,
            "green_risk_pct": _safe_ratio(risk, green_close),
            "diagnostic_window_end": post_price.index[-1] if not post_price.empty else pd.NaT,
        }
        for window in diagnostic_windows:
            diag.update(
                _diagnostic_window_row(
                    post_price,
                    post_flow,
                    green_close=green_close,
                    risk=risk,
                    window=int(window),
                )
            )
        diagnostics.append(diag)

        base_fields = signal._asdict()
        base_decision = {
            **base_fields,
            "green_time": green_time,
            "event_time": green_time,
            "decision_time": green_time,
            "feature_window_end": green_time,
            "entry_model": "GREEN_NEXT_OPEN",
            "entry_family": "baseline",
            "decision_reason": "green_closed_bar",
            "decision_bar_offset": 0,
            "decision_close": green_close,
            "decision_stop_known": True,
        }
        decisions.append(base_decision)
        funnel.append(
            {
                "episode_id": int(signal.episode_id),
                "green_time": green_time,
                "entry_model": "GREEN_NEXT_OPEN",
                "status": "entered",
                "decision_time": green_time,
                "decision_bar_offset": 0,
                "reason": "green_closed_bar",
            }
        )

        for model in ENTRY_MODELS[1:]:
            pullback_armed = False
            pullback_low = green_close
            pullback_peak_sell = np.nan
            status = "expired"
            status_reason = "max_wait_without_trigger"
            decision_row: dict[str, Any] | None = None
            model_end = min(len(post_price), int(model.max_wait_bars))

            for offset in range(1, model_end + 1):
                price_path = post_price.iloc[:offset]
                flow_path = post_flow.iloc[:offset]
                current_low = _finite(price_path["low"].iloc[-1])
                if current_low <= stop_price:
                    status = "aborted"
                    status_reason = "purple_stop_broken_before_entry"
                    break
                pullback_low = min(pullback_low, current_low)
                pullback_depth_r = _safe_ratio(green_close - pullback_low, risk)
                if np.isfinite(pullback_depth_r) and 0.12 <= pullback_depth_r <= 0.85:
                    pullback_armed = True
                    pullback_peak_sell = max(
                        _finite(pullback_peak_sell, 0.0),
                        _max(flow_path, "sell_notional_ratio_base"),
                    )

                snapshot = _decision_snapshot(
                    price_path,
                    flow_path,
                    green_close=green_close,
                    green_high=green_high,
                    risk=risk,
                    pullback_low=pullback_low,
                    bar_offset=offset,
                )
                if _flow_resurge_abort(snapshot, bar_offset=offset):
                    status = "aborted"
                    status_reason = "sell_pressure_resurged"
                    break

                continuation = (
                    offset <= 3
                    and _continuation_trigger(snapshot, require_flow=model.require_flow)
                )
                reclaim = _pullback_trigger(
                    snapshot,
                    pullback_armed=pullback_armed,
                    require_flow=model.require_flow,
                    pullback_peak_sell=pullback_peak_sell,
                )
                triggered = (
                    continuation if model.mode == "continuation"
                    else reclaim if model.mode == "pullback"
                    else continuation or reclaim
                )
                if not triggered:
                    continue

                decision_time = pd.Timestamp(price_path.index[-1])
                trigger_kind = "continuation" if continuation else "pullback_reclaim"
                decision_row = {
                    **base_fields,
                    **snapshot,
                    "green_time": green_time,
                    "event_time": decision_time,
                    "decision_time": decision_time,
                    "feature_window_end": decision_time,
                    "entry_model": model.name,
                    "entry_family": model.family,
                    "decision_reason": trigger_kind,
                    "decision_stop_known": True,
                    "pullback_armed": bool(pullback_armed),
                    "pullback_peak_sell_intensity": pullback_peak_sell,
                }
                decisions.append(decision_row)
                status = "entered"
                status_reason = trigger_kind
                break

            funnel.append(
                {
                    "episode_id": int(signal.episode_id),
                    "green_time": green_time,
                    "entry_model": model.name,
                    "status": status,
                    "decision_time": decision_row["decision_time"] if decision_row is not None else pd.NaT,
                    "decision_bar_offset": decision_row["decision_bar_offset"] if decision_row is not None else np.nan,
                    "reason": status_reason,
                }
            )

        if reporter is not None:
            reporter.update(done)
    if reporter is not None:
        reporter.close()

    diag_df = pd.DataFrame(diagnostics).sort_values("green_time").reset_index(drop=True)
    decision_df = pd.DataFrame(decisions).sort_values(["event_time", "entry_model"]).reset_index(drop=True)
    funnel_df = pd.DataFrame(funnel).sort_values(["green_time", "entry_model"]).reset_index(drop=True)
    return diag_df, decision_df, funnel_df


def entry_model_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_model": m.name,
                "family": m.family,
                "mode": m.mode,
                "max_wait_bars": m.max_wait_bars,
                "require_flow": m.require_flow,
                "description": m.description,
            }
            for m in ENTRY_MODELS
        ]
    )


def summarize_funnel(funnel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if funnel.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    green_year = pd.to_datetime(funnel["green_time"]).dt.year
    frame = funnel.assign(year=green_year)
    for model, part in frame.groupby("entry_model", sort=False):
        total = len(part)
        for status, sp in part.groupby("status", dropna=False):
            rows.append(
                {
                    "entry_model": model,
                    "status": status,
                    "count": len(sp),
                    "share": len(sp) / max(1, total),
                    "median_decision_bar_offset": pd.to_numeric(sp["decision_bar_offset"], errors="coerce").median(),
                    "top_reason": sp["reason"].value_counts().index[0] if len(sp) else "",
                }
            )
    for (year, model, status), part in frame.groupby(["year", "entry_model", "status"], dropna=False, sort=False):
        denom = int(((frame["year"] == year) & (frame["entry_model"] == model)).sum())
        yearly.append(
            {
                "year": int(year),
                "entry_model": model,
                "status": status,
                "count": len(part),
                "share": len(part) / max(1, denom),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def post_class_capture(
    funnel: pd.DataFrame,
    signal_classes: pd.DataFrame,
) -> pd.DataFrame:
    if funnel.empty or signal_classes.empty or "post_outcome_class" not in signal_classes.columns:
        return pd.DataFrame()
    classes = signal_classes[["episode_id", "post_outcome_class"]].drop_duplicates("episode_id")
    frame = funnel.merge(classes, on="episode_id", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for (model, status, outcome), part in frame.groupby(
        ["entry_model", "status", "post_outcome_class"], dropna=False, sort=False
    ):
        denom = int(
            ((frame["entry_model"] == model) & (frame["post_outcome_class"] == outcome)).sum()
        )
        rows.append(
            {
                "entry_model": model,
                "status": status,
                "post_outcome_class": outcome,
                "count": len(part),
                "capture_or_reject_rate": len(part) / max(1, denom),
            }
        )
    return pd.DataFrame(rows)
