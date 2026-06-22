#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Range-bar + footprint event lab for ETH OKX research.

This is not a trading strategy.  It creates objective event samples and measures
what happens after each event using only future labels for research reporting.
All event features are computed with shifted rolling windows, so event detection
itself does not use future data.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader

SCRIPT_NAME = "RangeFootprintEventLabV1"


@dataclass
class Config:
    symbol: str = "ETH-USDT-SWAP"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-15"
    warmup_start_date: str = "2022-01-01"
    range_pct: float = 0.0020
    price_step: float = 1.0
    data_dir: str | None = None
    out_dir: Path = Path("data/reports/research/range_footprint_event_lab_v1")

    swing_window: int = 80
    quantile_window: int = 300
    pressure_quantile: float = 0.88
    horizon_bars: int = 80
    stop_buffer_pct: float = 0.0003
    sweep_buffer_pct: float = 0.0
    breakout_buffer_pct: float = 0.0
    fee_rate_per_side: float = 0.00055
    slippage_pct: float = 0.00015
    max_events_per_type: int = 0  # 0 = no cap
    include_footprint: bool = True


def _ts_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _progress_marks(start: str, end: str) -> list[pd.Timestamp]:
    cur = pd.Timestamp(start) + pd.DateOffset(months=1)
    end_ts = pd.Timestamp(end)
    marks: list[pd.Timestamp] = []
    while cur <= end_ts:
        marks.append(cur)
        cur += pd.DateOffset(months=1)
    if not marks or marks[-1] < end_ts:
        marks.append(end_ts)
    return marks


def _rolling_quantile_shifted(s: pd.Series, window: int, q: float) -> pd.Series:
    return s.rolling(int(window), min_periods=max(20, int(window // 4))).quantile(float(q)).shift(1)


def _safe_div(num: pd.Series, den: pd.Series, default: float = np.nan) -> pd.Series:
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], default)


class FootprintFeatureStore:
    """SQLite footprint access for one-row-per-bar max buy/sell bucket features."""

    def __init__(self, cfg: Config):
        self.loader = OKXRangeFootprintLoader(
            symbol=cfg.symbol,
            range_pct=cfg.range_pct,
            price_step=cfg.price_step,
            data_dir=cfg.data_dir,
        )
        self.db_path = Path(self.loader.db_path)
        self.table_name = self.loader.table_name
        self._ensure_indexes()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _ensure_indexes(self) -> None:
        with self._connect() as conn:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_bar_id ON {self.table_name}(bar_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_end_ts ON {self.table_name}(end_ts)")

    def load_max_bucket_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        sell_sql = f"""
            SELECT bar_id, price_bucket AS max_sell_bucket,
                   sell_notional AS max_bucket_sell_notional,
                   delta_notional AS max_sell_bucket_delta_notional,
                   large_sell_notional AS max_bucket_large_sell_notional
            FROM (
                SELECT bar_id, price_bucket, sell_notional, delta_notional, large_sell_notional,
                       ROW_NUMBER() OVER (
                           PARTITION BY bar_id
                           ORDER BY sell_notional DESC, price_bucket ASC
                       ) AS rn
                FROM {self.table_name}
                WHERE end_ts >= ? AND end_ts <= ? AND sell_notional > 0
            ) ranked
            WHERE rn = 1
        """
        buy_sql = f"""
            SELECT bar_id, price_bucket AS max_buy_bucket,
                   buy_notional AS max_bucket_buy_notional,
                   delta_notional AS max_buy_bucket_delta_notional,
                   large_buy_notional AS max_bucket_large_buy_notional
            FROM (
                SELECT bar_id, price_bucket, buy_notional, delta_notional, large_buy_notional,
                       ROW_NUMBER() OVER (
                           PARTITION BY bar_id
                           ORDER BY buy_notional DESC, price_bucket DESC
                       ) AS rn
                FROM {self.table_name}
                WHERE end_ts >= ? AND end_ts <= ? AND buy_notional > 0
            ) ranked
            WHERE rn = 1
        """
        params = (_ts_text(start_date), _ts_text(end_date))
        with self._connect() as conn:
            sell = pd.read_sql_query(sell_sql, conn, params=params)
            buy = pd.read_sql_query(buy_sql, conn, params=params)
        out = sell.merge(buy, on="bar_id", how="outer")
        return out


def load_bars(cfg: Config) -> pd.DataFrame:
    loader = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir)
    bars = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
    if bars.empty:
        raise RuntimeError("No local range bars found. Build/preload range bars first.")
    bars = bars.reset_index(drop=True).copy()
    bars["end_ts"] = pd.to_datetime(bars["end_ts"])
    bars["start_ts"] = pd.to_datetime(bars["start_ts"])
    bars = bars.sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    print(
        f"Loaded range bars: {len(bars):,} | {bars['end_ts'].iloc[0]} -> {bars['end_ts'].iloc[-1]}",
        flush=True,
    )
    if cfg.include_footprint:
        try:
            fp = FootprintFeatureStore(cfg).load_max_bucket_features(cfg.warmup_start_date, cfg.end_date)
            bars = bars.merge(fp, on="bar_id", how="left")
            print(f"Loaded footprint max-bucket features: {len(fp):,}", flush=True)
        except Exception as exc:
            print(f"[WARN] footprint max-bucket features unavailable: {exc}", flush=True)
    return bars


def add_features(b: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = b.copy()
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "notional",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "taker_buy_ratio",
        "duration_seconds",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    price_span = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_pct_of_range"] = ((out["close"] - out["open"]).abs() / price_span).fillna(0.0)
    out["close_pos"] = ((out["close"] - out["low"]) / price_span).clip(0, 1).fillna(0.5)
    out["bar_return_pct"] = _safe_div(out["close"] - out["open"], out["open"], 0.0)
    out["delta_ratio"] = _safe_div(out["delta_notional"], out["notional"].abs(), 0.0)
    out["buy_sell_imbalance"] = _safe_div(out["buy_notional"] - out["sell_notional"], out["buy_notional"] + out["sell_notional"], 0.0)
    out["cvd_notional"] = out["delta_notional"].cumsum()
    out["rolling_delta_sum"] = out["delta_notional"].rolling(cfg.swing_window, min_periods=10).sum().shift(1)

    out["prior_swing_high"] = out["high"].shift(1).rolling(cfg.swing_window, min_periods=max(10, cfg.swing_window // 4)).max()
    out["prior_swing_low"] = out["low"].shift(1).rolling(cfg.swing_window, min_periods=max(10, cfg.swing_window // 4)).min()
    out["prior_cvd_high"] = out["cvd_notional"].shift(1).rolling(cfg.swing_window, min_periods=max(10, cfg.swing_window // 4)).max()
    out["prior_cvd_low"] = out["cvd_notional"].shift(1).rolling(cfg.swing_window, min_periods=max(10, cfg.swing_window // 4)).min()

    out["buy_notional_q"] = _rolling_quantile_shifted(out["buy_notional"], cfg.quantile_window, cfg.pressure_quantile)
    out["sell_notional_q"] = _rolling_quantile_shifted(out["sell_notional"], cfg.quantile_window, cfg.pressure_quantile)
    out["abs_delta_q"] = _rolling_quantile_shifted(out["delta_notional"].abs(), cfg.quantile_window, cfg.pressure_quantile)
    out["volume_q"] = _rolling_quantile_shifted(out["volume"], cfg.quantile_window, cfg.pressure_quantile)

    # Footprint bucket positions inside the bar, when available.
    for col in ["max_sell_bucket", "max_buy_bucket", "max_bucket_sell_notional", "max_bucket_buy_notional"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "max_sell_bucket" in out.columns:
        out["max_sell_bucket_pos"] = ((out["max_sell_bucket"] - out["low"]) / price_span).replace([np.inf, -np.inf], np.nan)
    else:
        out["max_sell_bucket_pos"] = np.nan
    if "max_buy_bucket" in out.columns:
        out["max_buy_bucket_pos"] = ((out["max_buy_bucket"] - out["low"]) / price_span).replace([np.inf, -np.inf], np.nan)
    else:
        out["max_buy_bucket_pos"] = np.nan
    return out


def _first_touch_outcomes(
    *,
    side: str,
    future_high: np.ndarray,
    future_low: np.ndarray,
    entry: float,
    stop: float,
    risk: float,
    targets_r: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for r in targets_r:
        target = entry + r * risk if side == "long" else entry - r * risk
        hit = False
        hit_bar = np.nan
        for j, (hi, lo) in enumerate(zip(future_high, future_low), start=1):
            # Conservative same-bar ordering: stop first if both are touched.
            if side == "long":
                if lo <= stop:
                    hit = False
                    hit_bar = j
                    break
                if hi >= target:
                    hit = True
                    hit_bar = j
                    break
            else:
                if hi >= stop:
                    hit = False
                    hit_bar = j
                    break
                if lo <= target:
                    hit = True
                    hit_bar = j
                    break
        results[f"hit_{int(r)}r_before_stop"] = bool(hit)
        results[f"touch_{int(r)}r_or_stop_bars"] = float(hit_bar) if np.isfinite(hit_bar) else np.nan
    return results


def _evaluate_event(b: pd.DataFrame, i: int, side: str, event_type: str, cfg: Config) -> dict[str, Any] | None:
    n = len(b)
    entry_i = i + 1
    end_i = min(n, entry_i + cfg.horizon_bars)
    if entry_i >= n or end_i <= entry_i:
        return None

    open_a = b["open"].to_numpy(float)
    high_a = b["high"].to_numpy(float)
    low_a = b["low"].to_numpy(float)
    close_a = b["close"].to_numpy(float)

    if side == "long":
        entry = open_a[entry_i] * (1.0 + cfg.slippage_pct)
        stop = low_a[i] * (1.0 - cfg.stop_buffer_pct)
        risk = entry - stop
        if risk <= 0:
            return None
        future_high = high_a[entry_i:end_i]
        future_low = low_a[entry_i:end_i]
        mfe_r = (float(np.max(future_high)) - entry) / risk
        mae_r = (entry - float(np.min(future_low))) / risk
        horizon_r = (close_a[end_i - 1] - entry) / risk
    else:
        entry = open_a[entry_i] * (1.0 - cfg.slippage_pct)
        stop = high_a[i] * (1.0 + cfg.stop_buffer_pct)
        risk = stop - entry
        if risk <= 0:
            return None
        future_high = high_a[entry_i:end_i]
        future_low = low_a[entry_i:end_i]
        mfe_r = (entry - float(np.min(future_low))) / risk
        mae_r = (float(np.max(future_high)) - entry) / risk
        horizon_r = (entry - close_a[end_i - 1]) / risk

    touches = _first_touch_outcomes(
        side=side,
        future_high=future_high,
        future_low=future_low,
        entry=entry,
        stop=stop,
        risk=risk,
    )
    approx_roundtrip_cost_r = entry * (2.0 * cfg.fee_rate_per_side + 2.0 * cfg.slippage_pct) / risk

    row = b.iloc[i]
    out = {
        "event_time": row["end_ts"],
        "entry_time": b.iloc[entry_i]["start_ts"],
        "event_i": int(i),
        "entry_i": int(entry_i),
        "bar_id": int(row["bar_id"]),
        "event_type": event_type,
        "side": side,
        "entry": float(entry),
        "stop": float(stop),
        "risk": float(risk),
        "horizon_bars": int(end_i - entry_i),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "horizon_r": float(horizon_r),
        "approx_roundtrip_cost_r": float(approx_roundtrip_cost_r),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "prior_swing_high": float(row.get("prior_swing_high", np.nan)),
        "prior_swing_low": float(row.get("prior_swing_low", np.nan)),
        "delta_notional": float(row.get("delta_notional", np.nan)),
        "delta_ratio": float(row.get("delta_ratio", np.nan)),
        "buy_notional": float(row.get("buy_notional", np.nan)),
        "sell_notional": float(row.get("sell_notional", np.nan)),
        "taker_buy_ratio": float(row.get("taker_buy_ratio", np.nan)),
        "volume": float(row.get("volume", np.nan)),
        "close_pos": float(row.get("close_pos", np.nan)),
        "max_sell_bucket_pos": float(row.get("max_sell_bucket_pos", np.nan)),
        "max_buy_bucket_pos": float(row.get("max_buy_bucket_pos", np.nan)),
    }
    out.update(touches)
    return out


def generate_events(b: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(cfg.start_date)
    end_ts = pd.Timestamp(cfg.end_date)
    marks = _progress_marks(cfg.start_date, cfg.end_date)
    mark_pos = 0
    monthly_counts: dict[str, int] = {}

    required_cols = ["prior_swing_high", "prior_swing_low", "buy_notional_q", "sell_notional_q", "abs_delta_q"]
    valid = b[required_cols].notna().all(axis=1)

    high_pressure_buy = (b["buy_notional"] >= b["buy_notional_q"]) & (b["delta_notional"] >= b["abs_delta_q"])
    high_pressure_sell = (b["sell_notional"] >= b["sell_notional_q"]) & (b["delta_notional"] <= -b["abs_delta_q"])

    sweep_low = (b["low"] < b["prior_swing_low"] * (1.0 - cfg.sweep_buffer_pct)) & (b["close"] > b["prior_swing_low"])
    sweep_high = (b["high"] > b["prior_swing_high"] * (1.0 + cfg.sweep_buffer_pct)) & (b["close"] < b["prior_swing_high"])

    breakout_up = (b["close"] > b["prior_swing_high"] * (1.0 + cfg.breakout_buffer_pct)) & (b["close_pos"] >= 0.70)
    breakout_down = (b["close"] < b["prior_swing_low"] * (1.0 - cfg.breakout_buffer_pct)) & (b["close_pos"] <= 0.30)

    cvd_higher_low = b["cvd_notional"] > b["prior_cvd_low"]
    cvd_lower_high = b["cvd_notional"] < b["prior_cvd_high"]

    event_masks: list[tuple[str, str, pd.Series]] = [
        ("sweep_low_reclaim_sell_pressure", "long", valid & sweep_low & high_pressure_sell),
        ("sweep_high_reclaim_buy_pressure", "short", valid & sweep_high & high_pressure_buy),
        ("initiative_breakout_up_buy_pressure", "long", valid & breakout_up & high_pressure_buy),
        ("initiative_breakout_down_sell_pressure", "short", valid & breakout_down & high_pressure_sell),
        ("delta_divergence_low_reclaim", "long", valid & sweep_low & (b["delta_notional"] < 0) & cvd_higher_low),
        ("delta_divergence_high_reclaim", "short", valid & sweep_high & (b["delta_notional"] > 0) & cvd_lower_high),
    ]

    for event_type, side, mask in event_masks:
        idxs = np.flatnonzero(mask.to_numpy(dtype=bool))
        if cfg.max_events_per_type and len(idxs) > cfg.max_events_per_type:
            idxs = idxs[-cfg.max_events_per_type :]
        print(f"[{SCRIPT_NAME}] event_type={event_type} side={side} candidates={len(idxs):,}", flush=True)
        for i in idxs:
            ts = pd.Timestamp(b.iloc[i]["end_ts"])
            if ts < start_ts or ts > end_ts:
                continue
            while mark_pos < len(marks) and ts >= marks[mark_pos]:
                print(
                    f"[{SCRIPT_NAME}] completed_to={marks[mark_pos].strftime('%Y-%m-%d')} events={len(rows):,}",
                    flush=True,
                )
                mark_pos += 1
            ev = _evaluate_event(b, int(i), side, event_type, cfg)
            if ev is not None:
                rows.append(ev)
                monthly_counts[event_type] = monthly_counts.get(event_type, 0) + 1

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["event_time"] = pd.to_datetime(events["event_time"])
    events["entry_time"] = pd.to_datetime(events["entry_time"])
    return events.sort_values(["event_time", "event_type"]).reset_index(drop=True)


def _summarize_group(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    hit1 = float(g["hit_1r_before_stop"].mean()) if n else np.nan
    hit2 = float(g["hit_2r_before_stop"].mean()) if n else np.nan
    hit3 = float(g["hit_3r_before_stop"].mean()) if n else np.nan
    gross_exp_1r = hit1 * 1.0 - (1.0 - hit1) * 1.0
    gross_exp_2r = hit2 * 2.0 - (1.0 - hit2) * 1.0
    gross_exp_3r = hit3 * 3.0 - (1.0 - hit3) * 1.0
    avg_cost = float(g["approx_roundtrip_cost_r"].mean())
    return pd.Series(
        {
            "events": n,
            "avg_mfe_r": float(g["mfe_r"].mean()),
            "median_mfe_r": float(g["mfe_r"].median()),
            "avg_mae_r": float(g["mae_r"].mean()),
            "median_mae_r": float(g["mae_r"].median()),
            "avg_horizon_r": float(g["horizon_r"].mean()),
            "hit_1r_before_stop_pct": hit1 * 100.0,
            "hit_2r_before_stop_pct": hit2 * 100.0,
            "hit_3r_before_stop_pct": hit3 * 100.0,
            "gross_exp_1r_stop_r": gross_exp_1r,
            "gross_exp_2r_stop_r": gross_exp_2r,
            "gross_exp_3r_stop_r": gross_exp_3r,
            "avg_cost_r": avg_cost,
            "net_exp_1r_stop_r": gross_exp_1r - avg_cost,
            "net_exp_2r_stop_r": gross_exp_2r - avg_cost,
            "net_exp_3r_stop_r": gross_exp_3r - avg_cost,
        }
    )


def build_summaries(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = events.groupby(["event_type", "side"], dropna=False).apply(_summarize_group, include_groups=False).reset_index()
    events = events.copy()
    events["year"] = events["event_time"].dt.year
    yearly = events.groupby(["year", "event_type", "side"], dropna=False).apply(_summarize_group, include_groups=False).reset_index()
    return summary.sort_values("net_exp_2r_stop_r", ascending=False), yearly.sort_values(["event_type", "year"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=SCRIPT_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="data/reports/research/range_footprint_event_lab_v1")
    p.add_argument("--swing-window", type=int, default=80)
    p.add_argument("--quantile-window", type=int, default=300)
    p.add_argument("--pressure-quantile", type=float, default=0.88)
    p.add_argument("--horizon-bars", type=int, default=80)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0003)
    p.add_argument("--sweep-buffer-pct", type=float, default=0.0)
    p.add_argument("--breakout-buffer-pct", type=float, default=0.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)
    p.add_argument("--max-events-per-type", type=int, default=0)
    p.add_argument("--no-footprint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        range_pct=args.range_pct,
        price_step=args.price_step,
        data_dir=args.data_dir,
        out_dir=Path(args.out_dir),
        swing_window=args.swing_window,
        quantile_window=args.quantile_window,
        pressure_quantile=args.pressure_quantile,
        horizon_bars=args.horizon_bars,
        stop_buffer_pct=args.stop_buffer_pct,
        sweep_buffer_pct=args.sweep_buffer_pct,
        breakout_buffer_pct=args.breakout_buffer_pct,
        fee_rate_per_side=args.fee_rate_per_side,
        slippage_pct=args.slippage_pct,
        max_events_per_type=args.max_events_per_type,
        include_footprint=not args.no_footprint,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{SCRIPT_NAME}] cfg={json.dumps({**cfg.__dict__, 'out_dir': str(cfg.out_dir)}, default=str, ensure_ascii=False)}", flush=True)

    bars = load_bars(cfg)
    bars = add_features(bars, cfg)
    events = generate_events(bars, cfg)
    summary, yearly = build_summaries(events)

    events_path = cfg.out_dir / "range_footprint_event_lab_v1_events.csv"
    summary_path = cfg.out_dir / "range_footprint_event_lab_v1_summary.csv"
    yearly_path = cfg.out_dir / "range_footprint_event_lab_v1_yearly.csv"
    config_path = cfg.out_dir / "range_footprint_event_lab_v1_config.json"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    config_path.write_text(json.dumps({**cfg.__dict__, "out_dir": str(cfg.out_dir)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== Event Summary sorted by net_exp_2r_stop_r ===")
    if summary.empty:
        print("No events generated.")
    else:
        cols = [
            "event_type",
            "side",
            "events",
            "avg_mfe_r",
            "avg_mae_r",
            "hit_1r_before_stop_pct",
            "hit_2r_before_stop_pct",
            "net_exp_1r_stop_r",
            "net_exp_2r_stop_r",
            "avg_cost_r",
        ]
        print(summary[cols].to_string(index=False))
    print(f"\nOutputs: {cfg.out_dir}")


if __name__ == "__main__":
    main()
