#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Institutional Flow Regime Lab V1.

This research script looks for larger, persistent microstructure regimes instead
of one-bar events.  It is designed for CoinBacktest local OKX trade bars.

Core idea:
    Large participants usually leave a multi-bar / multi-hour footprint:
    persistent signed flow, participation, low/high impact, absorption, markup,
    distribution, exhaustion, or cross-market confirmation.

Timing:
    regime signal is generated after a closed primary trade bar;
    entry is next primary bar open;
    exits use the last closed bar before the target horizon.

This is a research/event-study script, not a tradable strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

EPS = 1e-12
DEFAULT_HORIZONS = (15, 30, 60, 120, 240, 480, 720)
DEFAULT_WINDOWS = (30, 60, 120, 240, 480)


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    family: str
    side: str
    mask_col: str
    window: int
    description: str


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_list_int(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_list_str(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    gross_profit = vals[vals > 0].sum()
    gross_loss = -vals[vals < 0].sum()
    if gross_loss <= 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def top5_winner_share(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    gains = vals[vals > 0].sort_values(ascending=False)
    total = gains.sum()
    if total <= 0 or gains.empty:
        return 0.0
    return float(gains.head(5).sum() / total)


def rolling_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(20, window // 5)
    mu = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def pct_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or max(20, window // 5)

    def _rank_last(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        valid = x[np.isfinite(x)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= x[-1]).mean())

    return s.rolling(window, min_periods=mp).apply(_rank_last, raw=True)


def load_trade_bars(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    *,
    warmup_start_date: str | None = None,
    build_missing: bool = True,
) -> pd.DataFrame:
    start = warmup_start_date or start_date
    loader = OKXTradeBarLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start, end_date, build_missing=build_missing)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reset_index(drop=True).copy()
    if "timestamp" not in out.columns:
        idx_name = df.index.name or "timestamp"
        out[idx_name] = df.index.to_numpy()
        if idx_name != "timestamp":
            out = out.rename(columns={idx_name: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def add_session_columns(out: pd.DataFrame) -> pd.DataFrame:
    # Timestamps in this project are usually naive +8 local.  For regime studies,
    # rough session labels are only context, not causal trading logic.
    h = out["timestamp"].dt.hour
    out["session"] = np.select(
        [h.between(0, 7), h.between(8, 14), h.between(15, 20), h.between(21, 23)],
        ["asia", "eu_morning", "eu_us_overlap", "us_late"],
        default="other",
    )
    return out


def build_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c != "timestamp":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "notional",
        "delta_notional",
        "large_delta_notional",
        "large_trades_count",
        "max_trade_notional",
        "taker_buy_ratio",
        "trades_count",
        "buy_notional",
        "sell_notional",
        "cvd_notional",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["ret_1"] = out["close"].pct_change().fillna(0.0)
    out["bar_ret"] = out["close"] / out["open"].replace(0, np.nan) - 1.0
    out["hl_range_pct"] = (out["high"] - out["low"]) / out["open"].replace(0, np.nan)
    out["close_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["abs_ret_1"] = out["ret_1"].abs()
    out["abs_delta_notional"] = out["delta_notional"].abs()
    out["abs_large_delta_notional"] = out["large_delta_notional"].abs()
    out["delta_sign"] = np.sign(out["delta_notional"])
    out["ret_sign"] = np.sign(out["ret_1"])

    # Long-term normalizers; values at signal bar only use history up to that closed bar.
    norm_w = max(720, max(windows) * 2)
    for c in ["notional", "abs_delta_notional", "abs_large_delta_notional", "hl_range_pct"]:
        out[f"{c}_z"] = rolling_z(out[c], norm_w)
        out[f"{c}_pct_rank"] = pct_rank(out[c], norm_w)

    # Rolling institutional-flow features.  These are deliberately coarse windows,
    # not a fine parameter grid.
    for w in windows:
        mp = max(10, w // 4)
        ret_w = out["close"] / out["close"].shift(w) - 1.0
        abs_path = out["ret_1"].abs().rolling(w, min_periods=mp).sum()
        signed_delta = out["delta_notional"].rolling(w, min_periods=mp).sum()
        signed_large_delta = out["large_delta_notional"].rolling(w, min_periods=mp).sum()
        notional_sum = out["notional"].rolling(w, min_periods=mp).sum()
        abs_delta_sum = out["abs_delta_notional"].rolling(w, min_periods=mp).sum()
        range_sum = out["hl_range_pct"].rolling(w, min_periods=mp).sum()
        pos_delta_ratio = (out["delta_notional"] > 0).rolling(w, min_periods=mp).mean()
        neg_delta_ratio = (out["delta_notional"] < 0).rolling(w, min_periods=mp).mean()
        pos_ret_ratio = (out["ret_1"] > 0).rolling(w, min_periods=mp).mean()
        neg_ret_ratio = (out["ret_1"] < 0).rolling(w, min_periods=mp).mean()

        out[f"ret_{w}"] = ret_w
        out[f"notional_sum_{w}"] = notional_sum
        out[f"delta_sum_{w}"] = signed_delta
        out[f"large_delta_sum_{w}"] = signed_large_delta
        out[f"abs_delta_sum_{w}"] = abs_delta_sum
        out[f"delta_ratio_{w}"] = signed_delta / notional_sum.replace(0, np.nan)
        out[f"large_delta_ratio_{w}"] = signed_large_delta / notional_sum.replace(0, np.nan)
        out[f"participation_z_{w}"] = rolling_z(notional_sum, norm_w)
        out[f"participation_pct_{w}"] = pct_rank(notional_sum, norm_w)
        out[f"flow_intensity_z_{w}"] = rolling_z(abs_delta_sum, norm_w)
        out[f"flow_intensity_pct_{w}"] = pct_rank(abs_delta_sum, norm_w)
        out[f"delta_consistency_buy_{w}"] = pos_delta_ratio
        out[f"delta_consistency_sell_{w}"] = neg_delta_ratio
        out[f"price_consistency_up_{w}"] = pos_ret_ratio
        out[f"price_consistency_down_{w}"] = neg_ret_ratio
        out[f"trend_efficiency_{w}"] = ret_w.abs() / abs_path.replace(0, np.nan)
        out[f"signed_trend_efficiency_{w}"] = np.sign(ret_w) * out[f"trend_efficiency_{w}"]
        out[f"impact_per_flow_{w}"] = ret_w.abs() / (abs_delta_sum / 1_000_000.0).replace(0, np.nan)
        out[f"signed_impact_{w}"] = ret_w / (signed_delta / 1_000_000.0).replace(0, np.nan)
        out[f"range_sum_{w}"] = range_sum
        # Acceleration: recent flow compared with the first part of the same window.
        short = max(5, w // 4)
        out[f"delta_accel_{w}"] = out["delta_notional"].rolling(short, min_periods=max(3, short // 2)).sum() - out["delta_notional"].rolling(w, min_periods=mp).sum() / max(1, w / short)
        out[f"ret_accel_{w}"] = out["close"].pct_change(short) - ret_w / max(1.0, w / short)

    return add_session_columns(out)


def add_regime_masks(f: pd.DataFrame, windows: list[int]) -> list[RegimeSpec]:
    specs: list[RegimeSpec] = []
    for w in windows:
        # Persistent meta-order: flow is directional, persistent, participates heavily, and price moves in same direction.
        buy = (
            (f[f"delta_ratio_{w}"] > 0.12)
            & (f[f"delta_consistency_buy_{w}"] >= 0.58)
            & (f[f"participation_pct_{w}"] >= 0.70)
            & (f[f"ret_{w}"] > 0)
            & (f[f"trend_efficiency_{w}"] >= 0.22)
        )
        sell = (
            (f[f"delta_ratio_{w}"] < -0.12)
            & (f[f"delta_consistency_sell_{w}"] >= 0.58)
            & (f[f"participation_pct_{w}"] >= 0.70)
            & (f[f"ret_{w}"] < 0)
            & (f[f"trend_efficiency_{w}"] >= 0.22)
        )
        f[f"reg_persistent_buy_{w}"] = buy.fillna(False)
        f[f"reg_persistent_sell_{w}"] = sell.fillna(False)
        specs += [
            RegimeSpec(f"persistent_meta_order_buy_{w}m", "persistent_meta_order", "long", f"reg_persistent_buy_{w}", w, f"{w}m persistent buy flow + price markup."),
            RegimeSpec(f"persistent_meta_order_sell_{w}m", "persistent_meta_order", "short", f"reg_persistent_sell_{w}", w, f"{w}m persistent sell flow + price markdown."),
        ]

        # Stealth accumulation/distribution: directional flow but price not moving much, meaning passive absorption / hidden liquidity.
        accum = (
            (f[f"delta_ratio_{w}"] > 0.12)
            & (f[f"delta_consistency_buy_{w}"] >= 0.56)
            & (f[f"participation_pct_{w}"] >= 0.65)
            & (f[f"ret_{w}"].abs() <= 0.0035)
            & (f[f"trend_efficiency_{w}"] <= 0.18)
        )
        distrib = (
            (f[f"delta_ratio_{w}"] < -0.12)
            & (f[f"delta_consistency_sell_{w}"] >= 0.56)
            & (f[f"participation_pct_{w}"] >= 0.65)
            & (f[f"ret_{w}"].abs() <= 0.0035)
            & (f[f"trend_efficiency_{w}"] <= 0.18)
        )
        f[f"reg_stealth_accum_{w}"] = accum.fillna(False)
        f[f"reg_stealth_distrib_{w}"] = distrib.fillna(False)
        specs += [
            RegimeSpec(f"stealth_accumulation_{w}m", "stealth_accumulation", "long", f"reg_stealth_accum_{w}", w, f"{w}m positive flow absorbed without price markup."),
            RegimeSpec(f"stealth_distribution_{w}m", "stealth_distribution", "short", f"reg_stealth_distrib_{w}", w, f"{w}m negative flow absorbed without price markdown."),
        ]

        # Transition: stealth phase just happened, then short-term displacement starts.
        lookback = max(5, w // 6)
        prior_accum = f[f"reg_stealth_accum_{w}"].rolling(lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
        prior_distrib = f[f"reg_stealth_distrib_{w}"].rolling(lookback, min_periods=1).max().shift(1).fillna(0).astype(bool)
        short_ret_col = f"ret_{max(5, min(30, w // 4))}"
        if short_ret_col not in f.columns:
            short_ret_col = "ret_30" if "ret_30" in f.columns else f"ret_{w}"
        f[f"reg_accum_to_markup_{w}"] = (prior_accum & (f[short_ret_col] > 0.002) & (f[f"delta_accel_{w}"] > 0)).fillna(False)
        f[f"reg_distrib_to_markdown_{w}"] = (prior_distrib & (f[short_ret_col] < -0.002) & (f[f"delta_accel_{w}"] < 0)).fillna(False)
        specs += [
            RegimeSpec(f"accumulation_to_markup_{w}m", "phase_transition", "long", f"reg_accum_to_markup_{w}", w, f"Stealth accumulation followed by upside displacement."),
            RegimeSpec(f"distribution_to_markdown_{w}m", "phase_transition", "short", f"reg_distrib_to_markdown_{w}", w, f"Stealth distribution followed by downside displacement."),
        ]

        # Controlled pullback inside a longer buy/sell regime.
        f[f"reg_buy_regime_controlled_pullback_{w}"] = (
            f[f"reg_persistent_buy_{w}"].shift(1).rolling(lookback, min_periods=1).max().fillna(0).astype(bool)
            & (f["ret_1"] < 0)
            & (f["delta_notional"] >= 0)
            & (f["close_pos"] >= 0.45)
        ).fillna(False)
        f[f"reg_sell_regime_controlled_bounce_{w}"] = (
            f[f"reg_persistent_sell_{w}"].shift(1).rolling(lookback, min_periods=1).max().fillna(0).astype(bool)
            & (f["ret_1"] > 0)
            & (f["delta_notional"] <= 0)
            & (f["close_pos"] <= 0.55)
        ).fillna(False)
        specs += [
            RegimeSpec(f"buy_regime_controlled_pullback_{w}m", "controlled_pullback", "long", f"reg_buy_regime_controlled_pullback_{w}", w, "Pullback against persistent buy regime with flow support."),
            RegimeSpec(f"sell_regime_controlled_bounce_{w}m", "controlled_pullback", "short", f"reg_sell_regime_controlled_bounce_{w}", w, "Bounce against persistent sell regime with flow resistance."),
        ]

        # Exhaustion / failed meta-order: strong flow, low or deteriorating impact, opposite short-term response.
        impact = f[f"impact_per_flow_{w}"]
        impact_low = impact <= impact.rolling(max(120, w), min_periods=max(30, w // 3)).quantile(0.30)
        f[f"reg_buying_exhaustion_{w}"] = (
            (f[f"delta_ratio_{w}"] > 0.15)
            & (f[f"flow_intensity_pct_{w}"] >= 0.70)
            & impact_low
            & (f[f"ret_{w}"] <= 0.001)
            & (f["close_pos"] < 0.50)
        ).fillna(False)
        f[f"reg_selling_exhaustion_{w}"] = (
            (f[f"delta_ratio_{w}"] < -0.15)
            & (f[f"flow_intensity_pct_{w}"] >= 0.70)
            & impact_low
            & (f[f"ret_{w}"] >= -0.001)
            & (f["close_pos"] > 0.50)
        ).fillna(False)
        specs += [
            RegimeSpec(f"buying_exhaustion_absorbed_{w}m", "failed_meta_order", "short", f"reg_buying_exhaustion_{w}", w, "Large buy flow with poor upside impact / absorption."),
            RegimeSpec(f"selling_exhaustion_absorbed_{w}m", "failed_meta_order", "long", f"reg_selling_exhaustion_{w}", w, "Large sell flow with poor downside impact / absorption."),
        ]

    return specs


def add_lead_context(f: pd.DataFrame, lead_df: pd.DataFrame, lead_symbol: str, windows: list[int]) -> tuple[pd.DataFrame, list[RegimeSpec]]:
    if lead_df.empty:
        return f, []
    lead = build_features(lead_df, windows)
    prefix = lead_symbol.replace("-", "_").replace("/", "_").lower()
    keep = ["timestamp"]
    for w in windows:
        keep += [f"ret_{w}", f"delta_ratio_{w}", f"reg_persistent_buy_{w}", f"reg_persistent_sell_{w}"]
    keep = [c for c in keep if c in lead.columns]
    lead = lead[keep].copy()
    for c in lead.columns:
        if c != "timestamp":
            lead[c] = lead[c].shift(1)
    lead = lead.rename(columns={c: f"lead_{prefix}_{c}" for c in lead.columns if c != "timestamp"})
    out = pd.merge_asof(f.sort_values("timestamp"), lead.sort_values("timestamp"), on="timestamp", direction="backward")
    specs: list[RegimeSpec] = []
    for w in windows:
        buy_col = f"lead_{prefix}_reg_persistent_buy_{w}"
        sell_col = f"lead_{prefix}_reg_persistent_sell_{w}"
        ret_col = f"lead_{prefix}_ret_{w}"
        if buy_col in out.columns and ret_col in out.columns:
            out[f"reg_lead_{prefix}_buy_eth_lag_{w}"] = (
                out[buy_col].fillna(False).astype(bool)
                & (out[f"ret_{min(w, 60)}"].abs() < 0.003 if f"ret_{min(w, 60)}" in out.columns else True)
            )
            specs.append(RegimeSpec(f"{prefix}_persistent_buy_eth_lag_{w}m", "lead_lag_meta_order", "long", f"reg_lead_{prefix}_buy_eth_lag_{w}", w, f"{lead_symbol} has persistent buy regime while ETH lags."))
        if sell_col in out.columns and ret_col in out.columns:
            out[f"reg_lead_{prefix}_sell_eth_lag_{w}"] = (
                out[sell_col].fillna(False).astype(bool)
                & (out[f"ret_{min(w, 60)}"].abs() < 0.003 if f"ret_{min(w, 60)}" in out.columns else True)
            )
            specs.append(RegimeSpec(f"{prefix}_persistent_sell_eth_lag_{w}m", "lead_lag_meta_order", "short", f"reg_lead_{prefix}_sell_eth_lag_{w}", w, f"{lead_symbol} has persistent sell regime while ETH lags."))
    return out, specs


def mask_to_event_positions(mask: pd.Series, timestamps: pd.Series, *, min_gap_minutes: int) -> np.ndarray:
    arr = mask.fillna(False).astype(bool).to_numpy()
    if len(arr) == 0 or not arr.any():
        return np.array([], dtype="int64")
    # Segment starts only: avoid counting every bar of a persistent regime.
    prev = np.r_[False, arr[:-1]]
    starts = np.flatnonzero(arr & ~prev)
    if len(starts) == 0 or min_gap_minutes <= 0:
        return starts.astype("int64")
    ts = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[ns]")
    kept: list[int] = []
    last_ns: int | None = None
    gap_ns = np.int64(pd.Timedelta(minutes=min_gap_minutes).value)
    for p in starts:
        cur = ts[p].astype("int64")
        if last_ns is None or cur - last_ns >= gap_ns:
            kept.append(int(p))
            last_ns = int(cur)
    return np.asarray(kept, dtype="int64")


def build_regime_events(f: pd.DataFrame, specs: list[RegimeSpec], start_date: str, *, min_gap_minutes: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date)
    timestamps = f["timestamp"]
    prog = ProgressReporter(label="[events] institutional regime scan", total=len(specs), every=max(1, len(specs) // 20))
    for i, spec in enumerate(specs, start=1):
        if spec.mask_col not in f.columns:
            prog.update(i)
            continue
        pos = mask_to_event_positions(f[spec.mask_col], timestamps, min_gap_minutes=min_gap_minutes)
        if len(pos) == 0:
            prog.update(i)
            continue
        sig_ts = pd.to_datetime(timestamps.iloc[pos].to_numpy())
        keep = sig_ts >= start_ts
        pos = pos[keep]
        sig_ts = sig_ts[keep]
        if len(pos) == 0:
            prog.update(i)
            continue
        rows.append(
            pd.DataFrame(
                {
                    "event_name": spec.name,
                    "family": spec.family,
                    "side": spec.side,
                    "window": spec.window,
                    "description": spec.description,
                    "signal_pos": pos.astype("int64"),
                    "signal_time": sig_ts,
                }
            )
        )
        prog.update(i)
    if prog.last_done < prog.total:
        prog.close()
    else:
        prog.closed = True
    if not rows:
        return pd.DataFrame(columns=["event_name", "family", "side", "window", "description", "signal_pos", "signal_time"])
    out = pd.concat(rows, ignore_index=True).sort_values(["signal_time", "event_name"]).reset_index(drop=True)
    out["event_date"] = out["signal_time"].dt.date.astype(str)
    out["event_year"] = out["signal_time"].dt.year.astype(int)
    return out


def build_segments(f: pd.DataFrame, specs: list[RegimeSpec], start_date: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(f["timestamp"]).to_numpy(dtype="datetime64[ns]")
    close = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype="float64")
    start_ts = pd.Timestamp(start_date).to_datetime64()
    for spec in specs:
        if spec.mask_col not in f.columns:
            continue
        arr = f[spec.mask_col].fillna(False).astype(bool).to_numpy()
        if not arr.any():
            continue
        change = np.diff(np.r_[False, arr, False].astype(int))
        starts = np.flatnonzero(change == 1)
        ends = np.flatnonzero(change == -1) - 1
        for a, b in zip(starts, ends):
            if ts[a] < start_ts:
                continue
            if b <= a:
                continue
            ret = (close[b] / close[a] - 1.0) * (1.0 if spec.side == "long" else -1.0)
            duration_min = (ts[b].astype("int64") - ts[a].astype("int64")) / 1e9 / 60.0
            rows.append(
                {
                    "event_name": spec.name,
                    "family": spec.family,
                    "side": spec.side,
                    "window": spec.window,
                    "start_time": pd.Timestamp(ts[a]),
                    "end_time": pd.Timestamp(ts[b]),
                    "start_pos": int(a),
                    "end_pos": int(b),
                    "duration_min": float(duration_min),
                    "bars": int(b - a + 1),
                    "segment_directional_ret": float(ret),
                }
            )
    return pd.DataFrame(rows)


def attach_outcomes(events: pd.DataFrame, f: pd.DataFrame, horizons: Iterable[int], fee_rate: float) -> pd.DataFrame:
    out = events.copy()
    idx_time = pd.to_datetime(f["timestamp"]).to_numpy(dtype="datetime64[ns]")
    open_arr = pd.to_numeric(f["open"], errors="coerce").to_numpy(dtype="float64")
    close_arr = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype="float64")
    if out.empty:
        return out
    signal_pos = pd.to_numeric(out["signal_pos"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    entry_pos = signal_pos + 1
    valid_entry = (entry_pos >= 0) & (entry_pos < len(f))
    out["entry_pos"] = pd.Series(entry_pos).where(valid_entry, np.nan).astype("float")
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    out.loc[valid_entry, "entry_time"] = pd.to_datetime(idx_time[entry_pos[valid_entry]])
    out.loc[valid_entry, "entry_price"] = open_arr[entry_pos[valid_entry]]
    side_sign = np.where(out["side"].astype(str).to_numpy() == "long", 1.0, -1.0)
    idx_ns = idx_time.astype("int64")
    for h in horizons:
        h = int(h)
        suffix = f"h{h}"
        out[f"exit_time_{suffix}"] = pd.NaT
        out[f"exit_bar_time_{suffix}"] = pd.NaT
        out[f"exit_price_{suffix}"] = np.nan
        out[f"next_open_ret_{suffix}_gross"] = np.nan
        out[f"next_open_ret_{suffix}_net"] = np.nan
        if not valid_entry.any():
            continue
        entry_ns = idx_ns[entry_pos.clip(0, len(f) - 1)]
        target_ns = entry_ns + np.int64(pd.Timedelta(minutes=h).value)
        exit_pos = np.searchsorted(idx_ns, target_ns, side="left") - 1
        valid_exit = valid_entry & (exit_pos >= entry_pos) & (exit_pos < len(f))
        if not valid_exit.any():
            continue
        ep = entry_pos[valid_exit]
        xp = exit_pos[valid_exit]
        gross = side_sign[valid_exit] * (close_arr[xp] / open_arr[ep] - 1.0)
        out.loc[valid_exit, f"exit_bar_time_{suffix}"] = pd.to_datetime(idx_time[xp])
        out.loc[valid_exit, f"exit_time_{suffix}"] = pd.to_datetime(target_ns[valid_exit])
        out.loc[valid_exit, f"exit_price_{suffix}"] = close_arr[xp]
        out.loc[valid_exit, f"next_open_ret_{suffix}_gross"] = gross
        out.loc[valid_exit, f"next_open_ret_{suffix}_net"] = gross - float(fee_rate)
    return out


def summarize_group(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(by, dropna=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(by, key)}
        row.update(
            {
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "win_rate": float((vals > 0).mean()),
                "profit_factor": profit_factor(vals),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "top5_winner_share": top5_winner_share(vals),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mean", "profit_factor", "count"], ascending=[False, False, False])
    return out


def build_stats(events: pd.DataFrame, segments: pd.DataFrame, horizons: list[int], out_dir: Path) -> None:
    overview_rows = []
    for h in horizons:
        metric = f"next_open_ret_h{h}_net"
        vals = pd.to_numeric(events.get(metric), errors="coerce").dropna()
        if vals.empty:
            continue
        overview_rows.append(
            {
                "horizon": h,
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "win_rate": float((vals > 0).mean()),
                "profit_factor": profit_factor(vals),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            }
        )
        summarize_group(events, ["family", "event_name", "side"], metric).to_csv(out_dir / f"03_event_stats_h{h}.csv", index=False)
        summarize_group(events, ["family", "side"], metric).to_csv(out_dir / f"04_family_stats_h{h}.csv", index=False)
        summarize_group(events, ["family", "event_name", "side", "event_year"], metric).to_csv(out_dir / f"05_yearly_stats_h{h}.csv", index=False)
        dedup = events.sort_values("signal_time").drop_duplicates(["event_name", "side", "event_date"], keep="first")
        summarize_group(dedup, ["family", "event_name", "side"], metric).to_csv(out_dir / f"06_daily_dedup_event_stats_h{h}.csv", index=False)
    pd.DataFrame(overview_rows).to_csv(out_dir / "02_overview.csv", index=False)

    if not segments.empty:
        summarize_group(segments, ["family", "event_name", "side"], "segment_directional_ret").to_csv(out_dir / "07_segment_stats.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "07_segment_stats.csv", index=False)

    rank_frames = []
    for h in [120, 240, 480, 720]:
        metric = f"next_open_ret_h{h}_net"
        s = summarize_group(events, ["family", "event_name", "side"], metric)
        if not s.empty:
            s.insert(0, "horizon", h)
            s["rank_score"] = s["mean"].fillna(0) * np.log1p(s["count"].fillna(0)) * np.minimum(s["profit_factor"].replace(np.inf, 10).fillna(0), 4)
            rank_frames.append(s)
    if rank_frames:
        pd.concat(rank_frames, ignore_index=True).sort_values(["rank_score", "mean", "count"], ascending=[False, False, False]).to_csv(out_dir / "15_candidate_rank.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "15_candidate_rank.csv", index=False)


def build_audit(events: pd.DataFrame, args: argparse.Namespace, horizons: list[int]) -> pd.DataFrame:
    rows = [
        {"check": "ordinary_kline_used", "value": 0, "note": "primary uses OKXTradeBarLoader"},
        {"check": "regime_start_events_only", "value": 1, "note": "events are regime starts, not every active bar"},
        {"check": "entry_next_open", "value": 1, "note": "entry is signal_pos + 1 open"},
    ]
    if not events.empty and "entry_pos" in events.columns:
        rows.append({"check": "entry_not_next_open_flag", "value": int((pd.to_numeric(events["entry_pos"], errors="coerce") != pd.to_numeric(events["signal_pos"], errors="coerce") + 1).sum()), "note": "must be zero"})
        et = pd.to_datetime(events["entry_time"], errors="coerce")
        for h in horizons:
            xt = pd.to_datetime(events.get(f"exit_time_h{h}"), errors="coerce")
            valid = et.notna() & xt.notna()
            mismatch = int(((xt[valid] - et[valid]) != pd.Timedelta(minutes=h)).sum()) if valid.any() else 0
            rows.append({"check": f"horizon_time_mismatch_h{h}", "value": mismatch, "note": "must be zero"})
    return pd.DataFrame(rows)


def write_brief(out_dir: Path, args: argparse.Namespace) -> None:
    rank_path = out_dir / "15_candidate_rank.csv"
    rank = pd.read_csv(rank_path) if rank_path.exists() and rank_path.stat().st_size > 0 else pd.DataFrame()
    overview = pd.read_csv(out_dir / "02_overview.csv") if (out_dir / "02_overview.csv").exists() else pd.DataFrame()
    seg = pd.read_csv(out_dir / "07_segment_stats.csv") if (out_dir / "07_segment_stats.csv").exists() and (out_dir / "07_segment_stats.csv").stat().st_size > 0 else pd.DataFrame()
    lines = ["# Institutional Flow Regime Lab V1", ""]
    lines += [
        "## Run config",
        f"- symbol: `{args.symbol}`",
        f"- timeframe: `{args.primary_timeframe}` trade bars",
        f"- date range: `{args.start_date}` -> `{args.end_date}`",
        f"- regime windows: `{args.windows}` minutes",
        f"- horizons: `{args.horizons}` minutes",
        f"- min event gap: `{args.min_event_gap_minutes}` minutes",
        f"- fee rate: `{args.fee_rate}`",
        "",
        "## Overview",
    ]
    lines.append(overview.to_markdown(index=False) if not overview.empty else "No overview generated.")
    lines += ["", "## Top fixed-horizon candidates"]
    if not rank.empty:
        cols = [c for c in ["horizon", "family", "event_name", "side", "count", "mean", "median", "win_rate", "profit_factor", "top5_winner_share"] if c in rank.columns]
        lines.append(rank.head(25)[cols].to_markdown(index=False))
    else:
        lines.append("No candidates generated.")
    lines += ["", "## Regime segment stats"]
    if not seg.empty:
        cols = [c for c in ["family", "event_name", "side", "count", "mean", "median", "win_rate", "profit_factor", "top5_winner_share"] if c in seg.columns]
        lines.append(seg.head(25)[cols].to_markdown(index=False))
    else:
        lines.append("No segment stats generated.")
    lines += [
        "",
        "## Guardrails",
        "- This studies multi-minute / multi-hour regimes, not one-bar impulse events.",
        "- Events are regime starts only; segment stats describe the whole detected regime.",
        "- It is still an event study; strong candidates need path replay, stop/TP, delay, fee/slippage stress, and root-signal de-dup before strategy promotion.",
    ]
    (out_dir / "20_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Institutional Flow Regime Lab V1")
    ap.add_argument("--symbol", default="ETH-USDT-SWAP")
    ap.add_argument("--primary-timeframe", default="1m")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--warmup-start-date", default="2022-01-01")
    ap.add_argument("--windows", default="30,60,120,240,480")
    ap.add_argument("--horizons", default="15,30,60,120,240,480,720")
    ap.add_argument("--fee-rate", type=float, default=0.0011)
    ap.add_argument("--lead-symbols", default="BTC-USDT-SWAP")
    ap.add_argument("--no-lead-lag", action="store_true")
    ap.add_argument("--build-missing-lead", action="store_true")
    ap.add_argument("--min-event-gap-minutes", type=int, default=30)
    ap.add_argument("--write-full-events", action="store_true")
    ap.add_argument("--event-sample-size", type=int, default=200_000)
    ap.add_argument("--write-full-segments", action="store_true")
    ap.add_argument("--segment-sample-size", type=int, default=200_000)
    ap.add_argument("--out-dir", default="data/reports/research/institutional_flow_regime_lab_v1")
    args = ap.parse_args(argv)

    out_dir = safe_mkdir(Path(args.out_dir))
    windows = parse_list_int(args.windows)
    horizons = parse_list_int(args.horizons)

    print(f"[load] trade bars symbol={args.symbol} tf={args.primary_timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args.symbol, args.primary_timeframe, args.start_date, args.end_date, warmup_start_date=args.warmup_start_date, build_missing=True)
    if bars.empty:
        raise RuntimeError("primary trade bars are empty")
    print(f"[load] rows={len(bars):,} range={bars['timestamp'].min()} -> {bars['timestamp'].max()}", flush=True)

    print("[features] building institutional flow features", flush=True)
    feat = build_features(bars, windows)
    specs = add_regime_masks(feat, windows)

    lead_status: list[dict[str, Any]] = []
    if not args.no_lead_lag:
        for lead_symbol in parse_list_str(args.lead_symbols):
            if not lead_symbol or lead_symbol == args.symbol:
                continue
            print(f"[lead] loading {lead_symbol} build_missing={args.build_missing_lead}", flush=True)
            try:
                lead = load_trade_bars(lead_symbol, args.primary_timeframe, args.start_date, args.end_date, warmup_start_date=args.warmup_start_date, build_missing=bool(args.build_missing_lead))
            except Exception as exc:
                print(f"[lead] skip {lead_symbol}: {exc}", flush=True)
                lead_status.append({"symbol": lead_symbol, "status": "error", "error": repr(exc)})
                continue
            if lead.empty:
                print(f"[lead] skip {lead_symbol}: empty", flush=True)
                lead_status.append({"symbol": lead_symbol, "status": "empty"})
                continue
            feat, lead_specs = add_lead_context(feat, lead, lead_symbol, windows)
            specs += lead_specs
            lead_status.append({"symbol": lead_symbol, "status": "loaded", "rows": int(len(lead))})
            print(f"[lead] loaded {lead_symbol} rows={len(lead):,} specs={len(lead_specs)}", flush=True)

    pd.DataFrame([s.__dict__ for s in specs]).to_csv(out_dir / "00_regime_catalog.csv", index=False)
    print(f"[events] specs={len(specs):,}", flush=True)
    events = build_regime_events(feat, specs, args.start_date, min_gap_minutes=args.min_event_gap_minutes)
    print(f"[events] regime_start_events={len(events):,} families={events['family'].nunique() if not events.empty else 0}", flush=True)

    print("[segments] building regime segments", flush=True)
    segments = build_segments(feat, specs, args.start_date)
    print(f"[segments] count={len(segments):,}", flush=True)

    print("[outcomes] attaching fixed-horizon outcomes", flush=True)
    events_out = attach_outcomes(events, feat, horizons, args.fee_rate)

    if args.write_full_events:
        events_out.to_csv(out_dir / "01_regime_events.csv", index=False)
    else:
        events_out.head(int(args.event_sample_size)).to_csv(out_dir / "01_regime_events_sample.csv", index=False)
    if args.write_full_segments:
        segments.to_csv(out_dir / "01_regime_segments.csv", index=False)
    else:
        segments.head(int(args.segment_sample_size)).to_csv(out_dir / "01_regime_segments_sample.csv", index=False)

    print("[stats] writing reports", flush=True)
    build_stats(events_out, segments, horizons, out_dir)
    audit = build_audit(events_out, args, horizons)
    audit.to_csv(out_dir / "08_causal_audit.csv", index=False)
    meta = {
        "script": "institutional_flow_regime_lab.py",
        "version": "v1",
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "primary_rows": int(len(bars)),
        "windows": windows,
        "horizons": horizons,
        "fee_rate": float(args.fee_rate),
        "regime_specs": int(len(specs)),
        "regime_start_events": int(len(events_out)),
        "segments": int(len(segments)),
        "families": sorted(events_out["family"].dropna().unique().tolist()) if not events_out.empty else [],
        "lead_status": lead_status,
    }
    (out_dir / "10_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_brief(out_dir, args)
    print(f"[done] out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
