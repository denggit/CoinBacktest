#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT MSS Edge Discovery (mechanical, causal, first-pass research).

This is deliberately NOT a full ICT strategy backtest.  The purpose is to answer
one narrow question first:

    Does a mechanically defined MSS event on ETH have positive forward expectancy
    after realistic round-trip costs, or are manual MSS entries likely the problem?

Research design
---------------
1. Load the project's existing OKX trade-bar data through src.data_feed.
2. Build causal swing highs/lows. A pivot at bar j is usable only after the
   right-confirmation bars have closed, plus one extra bar before the structure
   can be used by a signal on the next bar.
3. Define two event families:
   - STRUCTURE_BREAK: close crosses the latest confirmed opposing swing.
   - MSS_REVERSAL: the same break, but only when the immediately prior confirmed
     structure is directionally bearish/bullish (a mechanical proxy for ICT MSS).
4. Signal is generated on the closed signal bar; entry is next-bar open.
5. Compare raw and reversal MSS across a frozen set of pivot definitions.
6. Report forward returns, MFE/MAE, year/side splits, event spacing and causal
   audits. No parameter is selected by looking at one outcome table.
7. All outputs go to data/reports/research/market_structure/ict_mss_edge_discovery.

Important semantic limitation
-----------------------------
ICT's discretionary MSS has no single universal mechanical definition. This
research therefore labels its definitions explicitly as mechanical proxies. If a
proxy has no edge, that is evidence against that proxy, not proof that every
human interpretation of ICT MSS is useless.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.event_study.causal import audit_context_available_times
from src.research_common.event_study.stats import summarize_many
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report


DEFAULT_OUT = Path("data/reports/research/market_structure/ict_mss_edge_discovery")
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-06-30"
DEFAULT_WARMUP = "2022-01-01"
DEFAULT_PIVOTS = ((2, 2), (3, 2), (3, 3), (5, 3))
DEFAULT_HORIZONS = (1, 3, 6, 12, 24, 48)
DEFAULT_COST = 0.0011


@dataclass(frozen=True)
class PivotSpec:
    left: int
    right: int

    @property
    def name(self) -> str:
        return f"L{self.left}_R{self.right}"


def parse_int_list(text: str) -> tuple[int, ...]:
    values = sorted({int(x.strip()) for x in str(text).split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise ValueError("integer list must contain positive integers")
    return tuple(values)


def parse_pivots(text: str) -> tuple[PivotSpec, ...]:
    out: list[PivotSpec] = []
    for token in str(text).split(","):
        token = token.strip().upper()
        if not token:
            continue
        if "-" not in token:
            raise ValueError("pivot specs must look like L2-R2,L3-R2")
        left_text, right_text = token.replace("L", "").replace("R", "").split("-")
        left = int(left_text)
        right = int(right_text)
        if left <= 0 or right <= 0:
            raise ValueError("pivot left/right must be positive")
        out.append(PivotSpec(left, right))
    if not out:
        raise ValueError("pivot specs must not be empty")
    return tuple(dict.fromkeys(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=("1m", "5m", "15m"), default="5m")
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--warmup-start-date", default=DEFAULT_WARMUP)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--pivot-specs", default="L2-R2,L3-R2,L3-R3,L5-R3")
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--round-trip-cost", type=float, default=DEFAULT_COST)
    p.add_argument("--mfe-mae-horizon", type=int, default=48)
    p.add_argument("--cooldown-bars", type=int, default=0,
                   help="Optional event spacing control. 0 keeps every mechanically distinct MSS.")
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--no-review-pack", action="store_true")
    return p.parse_args()


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("trade-bar loader must return a DatetimeIndex")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()]
    out = out.loc[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            raise ValueError(f"missing required bar column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()


def _confirmed_pivot_mask(values: np.ndarray, left: int, right: int, *, high: bool) -> np.ndarray:
    """Return pivot centers using only the eventual definition.

    The mask itself is only used to locate historical pivot centers. It is never
    exposed as a signal at the pivot timestamp. Availability is shifted to
    center + right + 1 in build_structure_state(), so the event axis remains causal.
    """
    series = pd.Series(values)
    if high:
        left_extreme = series.shift(1).rolling(left, min_periods=left).max()
        right_extreme = series.iloc[::-1].shift(1).rolling(right, min_periods=right).max().iloc[::-1]
        mask = (series > left_extreme) & (series >= right_extreme)
    else:
        left_extreme = series.shift(1).rolling(left, min_periods=left).min()
        right_extreme = series.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
        mask = (series < left_extreme) & (series <= right_extreme)
    return mask.fillna(False).to_numpy(dtype=bool)


def _causal_last_level(
    prices: pd.Series,
    pivot_mask: np.ndarray,
    right: int,
    *,
    side_name: str,
) -> pd.DataFrame:
    """Create a bar-axis structure table with causal pivot metadata."""
    n = len(prices)
    positions = np.flatnonzero(pivot_mask)
    value_at = np.full(n, np.nan, dtype=float)
    pos_at = np.full(n, -1, dtype=np.int64)
    time_at = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    available_positions = positions + int(right) + 1
    valid = available_positions < n
    positions = positions[valid]
    available_positions = available_positions[valid]
    values = prices.to_numpy(dtype=float)[positions]
    times = prices.index.to_numpy(dtype="datetime64[ns]")[positions]
    value_at[available_positions] = values
    pos_at[available_positions] = positions
    time_at[available_positions] = times

    value_series = pd.Series(value_at, index=prices.index).ffill()
    pos_series = pd.Series(pos_at, index=prices.index).replace(-1, np.nan).ffill()
    time_series = pd.Series(pd.to_datetime(time_at), index=prices.index).ffill()
    return pd.DataFrame(
        {
            f"last_{side_name}_price": value_series,
            f"last_{side_name}_pivot_pos": pos_series,
            f"last_{side_name}_pivot_time": time_series,
        },
        index=prices.index,
    )


def _causal_last_two(
    prices: pd.Series,
    pivot_mask: np.ndarray,
    right: int,
    *,
    side_name: str,
) -> pd.DataFrame:
    """Last two confirmed levels, available before each signal bar."""
    n = len(prices)
    positions = np.flatnonzero(pivot_mask)
    available_positions = positions + int(right) + 1
    valid = available_positions < n
    positions = positions[valid]
    available_positions = available_positions[valid]
    vals = prices.to_numpy(dtype=float)[positions]
    pos_out = np.full(n, -1, dtype=np.int64)
    prev_pos_out = np.full(n, -1, dtype=np.int64)
    val_out = np.full(n, np.nan, dtype=float)
    prev_val_out = np.full(n, np.nan, dtype=float)
    last_pos = -1
    last_val = np.nan
    prev_pos = -1
    prev_val = np.nan
    by_avail = {int(a): (int(pos), float(val)) for a, pos, val in zip(available_positions, positions, vals)}
    for i in range(n):
        item = by_avail.get(i)
        if item is not None:
            prev_pos, prev_val = last_pos, last_val
            last_pos, last_val = item
        pos_out[i] = last_pos
        prev_pos_out[i] = prev_pos
        val_out[i] = last_val
        prev_val_out[i] = prev_val
    return pd.DataFrame(
        {
            f"last_{side_name}_price": val_out,
            f"prev_{side_name}_price": prev_val_out,
            f"last_{side_name}_pivot_pos": pos_out,
            f"prev_{side_name}_pivot_pos": prev_pos_out,
        },
        index=prices.index,
    )


def build_structure_state(bars: pd.DataFrame, spec: PivotSpec) -> pd.DataFrame:
    """Build causal swing structure and mechanical MSS candidates."""
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")

    high_mask = _confirmed_pivot_mask(high.to_numpy(dtype=float), spec.left, spec.right, high=True)
    low_mask = _confirmed_pivot_mask(low.to_numpy(dtype=float), spec.left, spec.right, high=False)

    highs = _causal_last_two(high, high_mask, spec.right, side_name="swing_high")
    lows = _causal_last_two(low, low_mask, spec.right, side_name="swing_low")
    state = pd.concat([highs, lows], axis=1)

    last_high = state["last_swing_high_price"]
    prev_high = state["prev_swing_high_price"]
    last_low = state["last_swing_low_price"]
    prev_low = state["prev_swing_low_price"]

    # A break must cross the latest confirmed structure from the previous bar.
    # This prevents repeated MSS labels while price remains above/below the level.
    prior_high = last_high.shift(1)
    prior_low = last_low.shift(1)
    prev_close = close.shift(1)
    long_break = prior_high.notna() & (prev_close <= prior_high) & (close > prior_high)
    short_break = prior_low.notna() & (prev_close >= prior_low) & (close < prior_low)

    # Mechanical reversal-state proxy:
    # long MSS = latest confirmed highs are making LH OR latest confirmed lows are LL;
    # short MSS = latest confirmed highs are HH OR latest confirmed lows are HL.
    # The state is frozen before the signal bar, so this is not a post-event label.
    bearish_context = (
        (last_high.notna() & prev_high.notna() & (last_high < prev_high))
        | (last_low.notna() & prev_low.notna() & (last_low < prev_low))
    )
    bullish_context = (
        (last_high.notna() & prev_high.notna() & (last_high > prev_high))
        | (last_low.notna() & prev_low.notna() & (last_low > prev_low))
    )

    state["close"] = close
    state["prev_close"] = prev_close
    state["structure_break_long"] = long_break
    state["structure_break_short"] = short_break
    state["mss_reversal_long"] = long_break & bearish_context.shift(1).fillna(False)
    state["mss_reversal_short"] = short_break & bullish_context.shift(1).fillna(False)
    state["pivot_spec"] = spec.name
    state["pivot_left"] = spec.left
    state["pivot_right"] = spec.right
    return state


def _event_table(state: pd.DataFrame, bars: pd.DataFrame, spec: PivotSpec) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    definitions = (
        ("STRUCTURE_BREAK", "LONG", state["structure_break_long"], 1, "last_swing_high"),
        ("STRUCTURE_BREAK", "SHORT", state["structure_break_short"], -1, "last_swing_low"),
        ("MSS_REVERSAL", "LONG", state["mss_reversal_long"], 1, "last_swing_high"),
        ("MSS_REVERSAL", "SHORT", state["mss_reversal_short"], -1, "last_swing_low"),
    )
    for event_name, side_name, mask, side, level_name in definitions:
        idx = state.index[mask.astype(bool)]
        if len(idx) == 0:
            continue
        level = state.loc[idx, f"{level_name}_price"].to_numpy(dtype=float)
        close = bars.loc[idx, "close"].to_numpy(dtype=float)
        prev = bars.loc[idx, "open"].to_numpy(dtype=float)
        distance = np.where(side == 1, close / level - 1.0, level / close - 1.0)
        part = pd.DataFrame(
            {
                "signal_time": idx,
                "event_name": event_name,
                "side": side,
                "side_name": side_name,
                "pivot_spec": spec.name,
                "structure_level": level,
                "signal_close": close,
                "signal_open": prev,
                "break_distance_pct": distance,
                "structure_pivot_time": state.loc[idx, f"last_{level_name}_pivot_time"].to_numpy(),
                "structure_pivot_pos": state.loc[idx, f"last_{level_name}_pivot_pos"].to_numpy(),
            },
            index=idx,
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True).sort_values(["signal_time", "event_name", "side"], kind="mergesort")
    # The same timestamp can only represent one directional break per event family.
    events = events.drop_duplicates(subset=["pivot_spec", "event_name", "signal_time", "side"], keep="first")
    return events.reset_index(drop=True)


def apply_cooldown(events: pd.DataFrame, bars: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    if events.empty or cooldown_bars <= 0:
        return events
    positions = bars.index.get_indexer(pd.DatetimeIndex(events["signal_time"]))
    out = events.copy()
    out["signal_bar_pos"] = positions
    keep = np.ones(len(out), dtype=bool)
    last_pos: dict[tuple[str, int], int] = {}
    for i, row in enumerate(out.itertuples(index=False)):
        key = (str(row.event_name), int(row.side))
        pos = int(row.signal_bar_pos)
        previous = last_pos.get(key)
        if previous is not None and pos - previous <= cooldown_bars:
            keep[i] = False
        else:
            last_pos[key] = pos
    return out.loc[keep].drop(columns=["signal_bar_pos"]).reset_index(drop=True)


def attach_forward_outcomes(events: pd.DataFrame, bars: pd.DataFrame, horizons: Iterable[int], cost: float, mfe_horizon: int) -> pd.DataFrame:
    out = events.copy()
    positions = bars.index.get_indexer(pd.DatetimeIndex(out["signal_time"]))
    out["signal_bar_pos"] = positions
    out["entry_bar_pos"] = positions + 1
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    valid = (positions >= 0) & (positions + 1 < len(bars))
    if valid.any():
        entry_pos = positions[valid] + 1
        out.loc[valid, "entry_time"] = bars.index[entry_pos]
        out.loc[valid, "entry_price"] = bars["open"].to_numpy(dtype=float)[entry_pos]
    side = out["side"].to_numpy(dtype=float)
    entry = out["entry_price"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    for h in sorted(set(int(x) for x in horizons)):
        future_pos = positions + h
        valid_h = valid & (future_pos < len(bars))
        values = np.full(len(out), np.nan, dtype=float)
        values[valid_h] = (close[future_pos[valid_h]] / entry[valid_h] - 1.0) * side[valid_h]
        out[f"next_open_ret_h{h}_gross"] = values
        out[f"next_open_ret_h{h}_net"] = values - float(cost)
    mfe = np.full(len(out), np.nan, dtype=float)
    mae = np.full(len(out), np.nan, dtype=float)
    progress = ProgressReporter(
        label="[MSS] MFE/MAE",
        total=len(out),
        every=5000,
        enabled=False,
    )
    # Prefix extrema make this O(N) for the full bar set rather than O(events*horizon).
    # For a fixed event-specific start, use rolling windows shifted by one bar.
    # The horizon is deliberately a single predeclared value in this first pass.
    h = int(mfe_horizon)
    future_high = pd.Series(high, index=bars.index).rolling(h, min_periods=h).max().shift(-(h - 1)).to_numpy(dtype=float)
    future_low = pd.Series(low, index=bars.index).rolling(h, min_periods=h).min().shift(-(h - 1)).to_numpy(dtype=float)
    # The rolling window above is [t, t+h-1]. We need [entry=t+1, t+h].
    # Therefore use windows shifted one bar relative to the signal position.
    future_high_from_entry = np.full(len(bars), np.nan, dtype=float)
    future_low_from_entry = np.full(len(bars), np.nan, dtype=float)
    if len(bars) >= h:
        # Reverse rolling gives forward window extrema without Python loops.
        future_high_from_entry = pd.Series(high, index=bars.index).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].to_numpy(dtype=float)
        future_low_from_entry = pd.Series(low, index=bars.index).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].to_numpy(dtype=float)
    entry_pos = positions + 1
    valid_mfe = valid & (entry_pos + h <= len(bars)) & np.isfinite(entry)
    if valid_mfe.any():
        hi = future_high_from_entry[entry_pos[valid_mfe]]
        lo = future_low_from_entry[entry_pos[valid_mfe]]
        long_mask = side[valid_mfe] == 1
        mfe_values = np.where(long_mask, hi / entry[valid_mfe] - 1.0, entry[valid_mfe] / lo - 1.0)
        mae_values = np.where(long_mask, lo / entry[valid_mfe] - 1.0, entry[valid_mfe] / hi - 1.0)
        mfe[valid_mfe] = mfe_values
        mae[valid_mfe] = mae_values
    out[f"mfe_h{h}"] = mfe
    out[f"mae_h{h}"] = mae
    out["year"] = pd.to_datetime(out["signal_time"]).dt.year
    out["entry_after_signal_flag"] = pd.to_datetime(out["entry_time"]) > pd.to_datetime(out["signal_time"])
    out["entry_price_matches_next_open_flag"] = np.isclose(
        pd.to_numeric(out["entry_price"], errors="coerce"),
        np.take(bars["open"].to_numpy(dtype=float), np.clip(entry_pos, 0, len(bars) - 1)),
        rtol=0.0,
        atol=1e-12,
    ) & valid
    out["causal_fail_flag"] = ~valid | ~out["entry_after_signal_flag"].fillna(False)
    return out


def summarize_events(events: pd.DataFrame, horizons: Iterable[int], min_count: int) -> pd.DataFrame:
    cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    parts = []
    for keys in [(), ("event_name",), ("side_name",), ("pivot_spec",), ("event_name", "pivot_spec"), ("event_name", "side_name")]:
        table = summarize_many(events, cols, group_cols=keys, min_count=min_count)
        if not table.empty:
            if not keys:
                table["group"] = "ALL"
            else:
                table["group"] = "+".join(keys)
            parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_yearly(events: pd.DataFrame, horizons: Iterable[int], min_count: int) -> pd.DataFrame:
    cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    return summarize_many(events, cols, group_cols=["event_name", "year"], min_count=min_count)


def write_report(
    out_dir: Path,
    *,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    overview: pd.DataFrame,
    yearly: pd.DataFrame,
    meta: dict[str, object],
    causal_audit: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "01_events.csv", index=False)
    overview.to_csv(out_dir / "02_overview.csv", index=False)
    yearly.to_csv(out_dir / "03_yearly.csv", index=False)
    causal_audit.to_csv(out_dir / "04_causal_audit.csv", index=False)

    counts = (
        events.groupby(["pivot_spec", "event_name", "side_name"], dropna=False)
        .size().rename("event_count").reset_index()
        if not events.empty else pd.DataFrame(columns=["pivot_spec", "event_name", "side_name", "event_count"])
    )
    counts.to_csv(out_dir / "05_event_counts.csv", index=False)

    meta = dict(meta)
    meta["bars_count"] = int(len(bars))
    meta["bar_start"] = str(bars.index.min()) if not bars.empty else None
    meta["bar_end"] = str(bars.index.max()) if not bars.empty else None
    (out_dir / "10_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = [
        "# ICT MSS Edge Discovery — First Pass",
        "",
        "> Mechanical research proxy only. This is not a claim that every discretionary ICT MSS interpretation is equivalent to the definitions below.",
        "",
        "## Research question",
        "",
        "Does a causal, mechanical structure break / reversal-MSS event on ETH have positive forward expectancy after the project's 0.11% round-trip fee convention?",
        "",
        "## Causal rule",
        "",
        "- Pivot center is never used at its own timestamp.",
        "- A pivot becomes usable at `pivot_time + right + 1 bar`.",
        "- MSS is generated on the closed signal bar.",
        "- Entry is the next bar open.",
        "- No HTF context, FVG, OB, session, liquidity sweep, CVD or manual discretion is included in this first pass.",
        "",
        "## How to read the result",
        "",
        "1. If both STRUCTURE_BREAK and MSS_REVERSAL are negative after cost across pivot definitions and years, the pure MSS hypothesis is weak.",
        "2. If STRUCTURE_BREAK is weak but MSS_REVERSAL is materially better, the directional context is doing the work.",
        "3. If forward returns are weak but MFE is healthy and MAE is small, entry/exit design may be the next problem rather than event direction.",
        "4. A single strong pivot setting is not treated as a discovery; robustness across the frozen definitions matters.",
        "",
        "## Files",
        "",
        "- `01_events.csv`: event-level audit and forward outcomes.",
        "- `02_overview.csv`: grouped net-return statistics.",
        "- `03_yearly.csv`: yearly stability.",
        "- `04_causal_audit.csv`: next-open causal checks.",
        "- `05_event_counts.csv`: event sample sizes.",
        "- `10_meta.json`: frozen research configuration.",
        "",
    ]
    (out_dir / "00_README.md").write_text("\n".join(md), encoding="utf-8")


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    bars = normalize_bars(bars)
    if bars.empty:
        raise RuntimeError("No trade-bar data returned")
    print(f"[load] rows={len(bars):,} range={bars.index[0]} -> {bars.index[-1]}", flush=True)
    return bars


def main() -> None:
    args = parse_args()
    horizons = parse_int_list(args.horizons)
    pivots = parse_pivots(args.pivot_specs)
    if args.mfe_mae_horizon < 1:
        raise ValueError("mfe-mae-horizon must be positive")
    if args.round_trip_cost < 0:
        raise ValueError("round-trip cost must be non-negative")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)

    research_start = pd.Timestamp(args.start_date)
    research_end = pd.Timestamp(args.end_date)
    bars_research = bars.loc[(bars.index >= research_start) & (bars.index <= research_end)].copy()
    if bars_research.empty:
        raise RuntimeError("Research window contains no bars")

    all_events: list[pd.DataFrame] = []
    with ProgressReporter(
        label="[MSS] pivot definitions",
        total=len(pivots),
        every=1,
        enabled=not False,
    ) as progress:
        for spec in pivots:
            state = build_structure_state(bars, spec)
            events = _event_table(state, bars, spec)
            if not events.empty:
                events = events.loc[
                    (pd.to_datetime(events["signal_time"]) >= research_start)
                    & (pd.to_datetime(events["signal_time"]) <= research_end)
                ].copy()
                events = apply_cooldown(events, bars, int(args.cooldown_bars))
                if not events.empty:
                    events = attach_forward_outcomes(
                        events,
                        bars,
                        horizons=horizons,
                        cost=float(args.round_trip_cost),
                        mfe_horizon=int(args.mfe_mae_horizon),
                    )
                    all_events.append(events)
            progress.step()

    if all_events:
        events = pd.concat(all_events, ignore_index=True, sort=False)
    else:
        events = pd.DataFrame()

    if not events.empty:
        causal_audit = events[[
            "signal_time", "entry_time", "entry_after_signal_flag",
            "entry_price_matches_next_open_flag", "causal_fail_flag",
        ]].copy()
        causal_audit["signal_on_bar_index_flag"] = events["signal_bar_pos"].ge(0)
        causal_audit["context_available_time_flag"] = False
        causal_audit["used_context_available_time"] = pd.NaT
        overview = summarize_events(events, horizons, int(args.min_count))
        yearly = build_yearly(events, horizons, int(args.min_count))
    else:
        causal_audit = pd.DataFrame()
        overview = pd.DataFrame()
        yearly = pd.DataFrame()

    meta = {
        "title": "ICT MSS Edge Discovery — First Pass",
        "experiment_id": "ICT_MSS_EDGE_DISCOVERY_R01",
        "edge_id": "ICT_MSS",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "research_start": str(research_start),
        "research_end": str(research_end),
        "warmup_start": str(pd.Timestamp(args.warmup_start_date)),
        "pivot_specs": [spec.name for spec in pivots],
        "horizons": list(horizons),
        "mfe_mae_horizon": int(args.mfe_mae_horizon),
        "round_trip_cost": float(args.round_trip_cost),
        "cooldown_bars": int(args.cooldown_bars),
        "signal_definition": "closed-bar close crosses latest causally confirmed opposing swing",
        "mss_reversal_definition": "structure break plus prior confirmed directional context",
        "entry_definition": "next bar open",
        "future_function_policy": "pivot available only at pivot + right + 1 bar; no current-bar pivot confirmation",
        "strategy_status": "event-study only; not a trading recommendation",
    }
    write_report(out_dir, bars=bars_research, events=events, overview=overview, yearly=yearly, meta=meta, causal_audit=causal_audit)
    if not args.no_review_pack:
        finalize_research_report(
            out_dir,
            experiment_id="ICT_MSS_EDGE_DISCOVERY_R01",
            edge_id="ICT_MSS",
            title="ICT MSS Edge Discovery — First Pass",
        )
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] events={len(events):,}", flush=True)
    if not overview.empty:
        print(overview.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
