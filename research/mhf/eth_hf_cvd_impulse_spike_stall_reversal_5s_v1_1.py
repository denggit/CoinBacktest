#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 5s CVD impulse spike-stall reversal V1.1 research.

Research-only hypothesis factory around the user's idea:

    On 5s trade bars, accumulated CVD suddenly changes with an almost vertical
    slope. The impulse bar should also be an extreme notional / delta shock.
    If the next 5s bar's accumulated CVD slope stalls or reverses, fade the
    impulse and enter in the opposite direction at the next 5s open.

V1 showed the broad condition was too noisy. V1.1 expands the idea with
pre-declared, non-retrospective hypothesis families:

    1. pure_spike_stall: extreme CVD slope + impulse-bar notional spike + stall;
    2. delta_spike_stall: same but requires delta-notional spike;
    3. price_no_follow_absorption: extreme CVD/notional impulse but price fails
       to travel enough, a proxy for absorption;
    4. stall_reversal_confirm: stall bar's CVD delta flips against the impulse;
    5. stop_run_reclaim: impulse pierces a short rolling high/low then stall bar
       reclaims inside the range;
    6. trend_counter_stoprun: impulse is against the recent micro trend, then
       stalls, a stop-run/rejoin setup.

It also tests multiple pre-declared holding / payoff structures:

    - time_only;
    - tp_timeout;
    - tp_sl_timeout with conservative same-bar TP/SL ordering (SL first);
    - tp_cvd_resume_timeout: exit next open if CVD resumes the impulse direction.

Important: CVD here is accumulated CVD (`cvd_notional`), not a single-bar CVD
feature. Event detection uses slopes/changes of the accumulated CVD series.

This script does not register a tradable edge, modify portfolio code, or import
business logic from other research scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "eth_hf_cvd_impulse_spike_stall_reversal_5s_v1_1.py"
SCRIPT_VERSION = "1.1.2"
EXPERIMENT_ID = "ETH_HF_CVD_IMPULSE_SPIKE_STALL_REVERSAL_5S_V1_1"
EDGE_ID = "ETH_EDGE_HF_CVD_IMPULSE_SPIKE_STALL_REVERSAL_5S_RESEARCH_V1_1"
DEFAULT_OUT_DIR = "data/reports/research/hf_cvd_impulse_spike_stall_reversal_5s_v1_1"
CAUSAL_POLICY = "closed 5s trade-bar signal; next 5s open entry; fixed TP/SL/time/CVD-resume exits"
MATCHED_BASELINE_COLUMNS = (
    "year",
    "month",
    "session",
    "regime",
    "volatility_bucket",
    "trend_bucket",
    "direction",
)


@dataclass(frozen=True)
class EventSpec:
    hypothesis: str
    impulse_bars: int
    cvd_slope_mult: float
    flat_ratio: float
    impulse_notional_mult: float
    impulse_delta_mult: float
    price_mode: str
    environment: str
    min_abs_price_move_pct: float
    flat_baseline_mult: float

    @property
    def variant(self) -> str:
        sm = _fmt(self.cvd_slope_mult)
        fr = _fmt(self.flat_ratio)
        nm = _fmt(self.impulse_notional_mult)
        dm = _fmt(self.impulse_delta_mult)
        pm = int(round(self.min_abs_price_move_pct * 10000))
        return (
            f"{self.hypothesis}__ib{self.impulse_bars}_sm{sm}_fr{fr}_"
            f"nm{nm}_dm{dm}_{self.price_mode}_{self.environment}_ret{pm}bp"
        )

    def event_name(self, direction: str) -> str:
        return f"cvd_spike_stall_reversal_5s_v1_1__{self.variant}__{direction}"


@dataclass(frozen=True)
class ExitSpec:
    exit_model: str
    max_hold_bars: int
    tp_pct: float
    sl_pct: float
    cvd_resume_mult: float

    @property
    def variant(self) -> str:
        tp = int(round(self.tp_pct * 10000))
        sl = int(round(self.sl_pct * 10000))
        rm = _fmt(self.cvd_resume_mult)
        return f"{self.exit_model}_tp{tp}bp_sl{sl}bp_mh{self.max_hold_bars}_rm{rm}"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH 5s accumulated-CVD impulse spike-stall reversal V1.1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="5s")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--hypotheses", default="pure_spike_stall,delta_spike_stall,price_no_follow_absorption,stall_reversal_confirm,stop_run_reclaim,trend_counter_stoprun")
    p.add_argument("--impulse-bars-list", default="1,2")
    p.add_argument("--cvd-slope-multipliers", default="12.0,20.0")
    p.add_argument("--flat-ratios", default="0.10,0.20")
    p.add_argument("--flat-baseline-mult", type=float, default=1.0)
    p.add_argument("--impulse-notional-mults", default="8.0,10.0,15.0")
    p.add_argument("--impulse-delta-mults", default="5.0,10.0")
    p.add_argument("--price-modes", default="align,no_follow,any")
    p.add_argument("--environments", default="all,low_vol_prior,high_vol_now,trend_counter,trend_with,range_reclaim")
    p.add_argument("--min-abs-price-move-pct", type=float, default=0.0003)
    p.add_argument("--cvd-norm-window", type=int, default=180)
    p.add_argument("--notional-window", type=int, default=180)
    p.add_argument("--range-window", type=int, default=60)
    p.add_argument("--trend-window", type=int, default=180, help="5s bars; 180=15m")
    p.add_argument("--exit-models", default="time_only,tp_timeout,tp_sl_timeout,tp_cvd_resume_timeout")
    p.add_argument("--tp-pct-list", default="0.002,0.003,0.004")
    p.add_argument("--sl-pct-list", default="0.002,0.003,0.004")
    p.add_argument("--max-hold-bars-list", default="12,36,60,120,180,360")
    p.add_argument("--cvd-resume-mults", default="1.0,2.0")
    p.add_argument("--mfe-mae-horizon", type=int, default=360)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3")
    p.add_argument("--min-count", type=int, default=500)
    p.add_argument("--min-events-per-year", type=float, default=120.0)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--baseline-samples", type=int, default=50)
    p.add_argument("--baseline-max-events-per-group", type=int, default=1000)
    p.add_argument("--baseline-prefilter-mean-net", type=float, default=-0.0002)
    p.add_argument("--baseline-prefilter-pf", type=float, default=0.95)
    p.add_argument("--baseline-seed", type=int, default=42)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--trade-sample-size", type=int, default=20000)
    p.add_argument("--write-full-trades", action="store_true")
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--disable-two-stage", action="store_true", help="Run the full event x exit grid directly. Not recommended for full 5s history.")
    p.add_argument("--stage1-max-hold-bars-list", default="60,180", help="Predeclared probe holds for two-stage triage; 60=5m, 180=15m on 5s bars.")
    p.add_argument("--stage1-tp-pct-list", default="0.003", help="Probe TP values for two-stage triage.")
    p.add_argument("--stage1-sl-pct-list", default="0.003", help="Probe SL values for two-stage triage.")
    p.add_argument("--stage1-event-mean-net-prefilter", type=float, default=-0.0015, help="Broad event-spec triage threshold. Used only to avoid running hopeless full grids, not to declare edge.")
    p.add_argument("--stage1-event-pf-prefilter", type=float, default=0.65, help="Broad event-spec triage PF threshold. Used only to avoid running hopeless full grids, not to declare edge.")
    p.add_argument("--stage1-top-event-specs-per-hypothesis-direction", type=int, default=6, help="Always keep top N event specs per hypothesis/direction if sample count is enough, to avoid over-pruning.")
    p.add_argument("--stage1-max-event-specs", type=int, default=48, help="Safety cap for event names sent to full exit replay.")
    p.add_argument("--max-stage2-replay-rows", type=int, default=5_000_000, help="Safety guard. If selected stage2 replay rows exceed this, stop and write stage1 triage only instead of risking memory explosion.")
    return p.parse_args(argv)


def _fmt(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def _parse_csv_strs(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys([p.strip() for p in str(raw).split(",") if p.strip()]))


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys([int(p.strip()) for p in str(raw).split(",") if p.strip()]))


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(dict.fromkeys([float(p.strip()) for p in str(raw).split(",") if p.strip()]))


def _annualized_years(start_date: str, end_date: str) -> float:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return max((end - start).total_seconds() / (365.25 * 86400.0), 1e-9)


def _profit_factor(values: np.ndarray | pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    gains = v[v > 0].sum()
    losses = -v[v < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _top_winner_share(values: np.ndarray | pd.Series, top_n: int = 5) -> float:
    v = np.asarray(values, dtype=float)
    winners = np.sort(v[np.isfinite(v) & (v > 0)])[::-1]
    if winners.size == 0:
        return 0.0
    return float(winners[:top_n].sum() / max(winners.sum(), 1e-12))


def _max_days_without_event(times: pd.Series | pd.DatetimeIndex) -> float:
    if len(times) <= 1:
        return float("nan")
    ts = pd.to_datetime(pd.Series(times).dropna()).sort_values()
    if len(ts) <= 1:
        return float("nan")
    gaps = ts.diff().dropna().dt.total_seconds() / 86400.0
    return float(gaps.max()) if not gaps.empty else float("nan")


def _assign_session(idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour.astype(int)
    return np.select([h < 8, h < 16], ["asia_00_08", "asia_europe_08_16"], default="us_16_24")


def _month_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while cur <= end_ts:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(nxt - pd.Timedelta(seconds=1), end_ts)
        chunks.append((cur, chunk_end))
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------


def _make_loader(args: argparse.Namespace) -> OKXTradeBarLoader:
    kwargs: dict[str, object] = {"symbol": args.symbol, "timeframe": args.primary_timeframe, "db_name": args.db_name}
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    return OKXTradeBarLoader(**kwargs)


def load_trade_bars_range(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    loader = _make_loader(args)
    df = loader.fetch_data_by_date_range(
        start,
        end,
        chunksize=args.chunksize,
        force_rebuild=bool(args.force_rebuild),
        cvd_mode="range",
        build_missing=not bool(args.no_build_missing),
    )
    if df.empty:
        return df
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    keep = [
        "open", "high", "low", "close", "notional", "delta_notional", "cvd_notional",
        "buy_notional", "sell_notional", "trades_count", "large_buy_notional", "large_sell_notional",
        "large_delta_notional", "large_trades_count", "max_trade_notional",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def build_features(
    df: pd.DataFrame,
    *,
    cvd_norm_window: int,
    notional_window: int,
    range_window: int,
    trend_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    required = ["open", "high", "low", "close", "cvd_notional"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required 5s trade-bar columns: {missing}")

    base = df.copy()
    idx = base.index
    close = base["close"].astype(float)
    high = base["high"].astype(float)
    low = base["low"].astype(float)
    cvd = base["cvd_notional"].astype(float)
    cvd_delta_1 = cvd.diff()
    rng_pct = (high - low) / close.replace(0, np.nan)
    ret_1 = close.pct_change()
    notional = base.get("notional", pd.Series(0.0, index=idx)).astype(float)
    delta_notional = base.get("delta_notional", cvd_delta_1).astype(float)

    notional_base = notional.shift(2).rolling(notional_window, min_periods=max(20, notional_window // 5)).median()
    delta_abs_base = delta_notional.abs().shift(2).rolling(notional_window, min_periods=max(20, notional_window // 5)).median()
    notional_z_base = notional.shift(1).rolling(notional_window, min_periods=max(20, notional_window // 5)).mean()
    notional_z_std = notional.shift(1).rolling(notional_window, min_periods=max(20, notional_window // 5)).std(ddof=0)
    notional_z = (notional - notional_z_base) / notional_z_std.replace(0, np.nan)

    abs_cvd_delta = cvd_delta_1.abs()
    cvd_delta_base = abs_cvd_delta.shift(1).rolling(cvd_norm_window, min_periods=max(20, cvd_norm_window // 5)).median()
    rng_base = rng_pct.shift(1).rolling(720, min_periods=120).median()
    vol_ratio = rng_pct / rng_base.replace(0, np.nan)
    prior_vol_median = rng_pct.shift(1).rolling(180, min_periods=60).median()
    prior_vol_ratio = prior_vol_median / rng_base.replace(0, np.nan)

    trend_ret = close.pct_change(trend_window)
    trend_bucket = np.select([trend_ret > 0.003, trend_ret < -0.003], ["up", "down"], default="flat")
    regime = np.select([close.pct_change(720).values > 0.01, close.pct_change(720).values < -0.01], ["trend_up", "trend_down"], default="normal")
    rolling_high = high.shift(2).rolling(range_window, min_periods=max(10, range_window // 3)).max()
    rolling_low = low.shift(2).rolling(range_window, min_periods=max(10, range_window // 3)).min()

    features = pd.DataFrame(
        {
            "cvd": cvd,
            "cvd_delta_1": cvd_delta_1,
            "abs_cvd_delta_1": abs_cvd_delta,
            "cvd_delta_base": cvd_delta_base,
            "notional_base": notional_base,
            "delta_abs_base": delta_abs_base,
            "notional_z": notional_z,
            "ret_1": ret_1,
            "range_pct": rng_pct,
            "vol_ratio": vol_ratio,
            "prior_vol_ratio": prior_vol_ratio,
            "trend_ret": trend_ret,
            "trend_bucket": trend_bucket,
            "rolling_high": rolling_high,
            "rolling_low": rolling_low,
            "year": idx.year.astype(int),
            "month": idx.month.astype(int),
            "session": _assign_session(idx),
            "weekday": idx.dayofweek.astype(int),
            "regime": regime,
        },
        index=idx,
    )
    features["volatility_bucket"] = pd.qcut(features["vol_ratio"].rank(method="first"), q=5, labels=False, duplicates="drop").astype("float")
    features["volatility_bucket"] = features["volatility_bucket"].fillna(-1).astype(int)

    out = pd.concat([base, features], axis=1)
    return out, list(features.columns)


# ---------------------------------------------------------------------------
# Specs / events / exits
# ---------------------------------------------------------------------------


def build_specs(args: argparse.Namespace) -> list[EventSpec]:
    hypotheses = set(_parse_csv_strs(args.hypotheses))
    all_price_modes = _parse_csv_strs(args.price_modes)
    all_envs = _parse_csv_strs(args.environments)
    specs: list[EventSpec] = []

    def add_grid(hypothesis: str, price_modes: Sequence[str], envs: Sequence[str], nm_vals: Sequence[float], dm_vals: Sequence[float], flat_vals: Sequence[float]) -> None:
        if hypothesis not in hypotheses:
            return
        for ib in _parse_csv_ints(args.impulse_bars_list):
            for sm in _parse_csv_floats(args.cvd_slope_multipliers):
                for fr in flat_vals:
                    for nm in nm_vals:
                        for dm in dm_vals:
                            for pmode in price_modes:
                                if pmode not in all_price_modes and "any" not in all_price_modes:
                                    continue
                                for env in envs:
                                    if env not in all_envs and "all" not in all_envs:
                                        continue
                                    specs.append(
                                        EventSpec(
                                            hypothesis=hypothesis,
                                            impulse_bars=ib,
                                            cvd_slope_mult=sm,
                                            flat_ratio=float(fr),
                                            impulse_notional_mult=float(nm),
                                            impulse_delta_mult=float(dm),
                                            price_mode=pmode,
                                            environment=env,
                                            min_abs_price_move_pct=float(args.min_abs_price_move_pct),
                                            flat_baseline_mult=float(args.flat_baseline_mult),
                                        )
                                    )

    nm_all = _parse_csv_floats(args.impulse_notional_mults)
    dm_all = _parse_csv_floats(args.impulse_delta_mults)
    fr_all = _parse_csv_floats(args.flat_ratios)
    # Pre-declared hypothesis families. Keep the grid intentionally limited.
    add_grid("pure_spike_stall", ["align"], ["all", "low_vol_prior", "high_vol_now"], nm_all, [0.0], fr_all)
    add_grid("delta_spike_stall", ["align"], ["all", "trend_counter"], nm_all[:2] or nm_all, dm_all, fr_all)
    add_grid("price_no_follow_absorption", ["no_follow"], ["all", "high_vol_now"], nm_all[1:] or nm_all, dm_all[:1] or dm_all, fr_all)
    add_grid("stall_reversal_confirm", ["align", "any"], ["all", "high_vol_now"], nm_all[1:] or nm_all, dm_all[:1] or dm_all, [max(fr_all) if fr_all else 0.2])
    add_grid("stop_run_reclaim", ["align"], ["range_reclaim"], nm_all[1:] or nm_all, dm_all[:1] or dm_all, fr_all)
    add_grid("trend_counter_stoprun", ["align"], ["trend_counter"], nm_all[1:] or nm_all, dm_all[:1] or dm_all, fr_all)
    # Deduplicate while preserving order.
    unique: dict[tuple[object, ...], EventSpec] = {}
    for s in specs:
        unique[(s.hypothesis, s.impulse_bars, s.cvd_slope_mult, s.flat_ratio, s.impulse_notional_mult, s.impulse_delta_mult, s.price_mode, s.environment)] = s
    return list(unique.values())


def build_exit_specs(args: argparse.Namespace) -> list[ExitSpec]:
    models = set(_parse_csv_strs(args.exit_models))
    holds = _parse_csv_ints(args.max_hold_bars_list)
    tps = _parse_csv_floats(args.tp_pct_list)
    sls = _parse_csv_floats(args.sl_pct_list)
    resumes = _parse_csv_floats(args.cvd_resume_mults)
    specs: list[ExitSpec] = []
    for mh in holds:
        if "time_only" in models:
            specs.append(ExitSpec("time_only", mh, 0.0, 0.0, 0.0))
        if "tp_timeout" in models:
            for tp in tps:
                specs.append(ExitSpec("tp_timeout", mh, tp, 0.0, 0.0))
        if "tp_sl_timeout" in models:
            for tp in tps:
                for sl in sls:
                    specs.append(ExitSpec("tp_sl_timeout", mh, tp, sl, 0.0))
        if "tp_cvd_resume_timeout" in models:
            for tp in tps:
                for rm in resumes:
                    specs.append(ExitSpec("tp_cvd_resume_timeout", mh, tp, 0.0, rm))
    # Deduplicate.
    unique: dict[tuple[object, ...], ExitSpec] = {}
    for e in specs:
        unique[(e.exit_model, e.max_hold_bars, e.tp_pct, e.sl_pct, e.cvd_resume_mult)] = e
    return list(unique.values())




def build_stage1_exit_specs(args: argparse.Namespace) -> list[ExitSpec]:
    """Small, predeclared probe exit set for two-stage research triage.

    This is a compute guard, not an optimization target. Stage 1 only decides
    which event structures are not obviously hopeless before running the full
    exit matrix. Final decisions still come from stage 2 reports.
    """
    holds = _parse_csv_ints(args.stage1_max_hold_bars_list)
    tps = _parse_csv_floats(args.stage1_tp_pct_list)
    sls = _parse_csv_floats(args.stage1_sl_pct_list)
    specs: list[ExitSpec] = []
    for mh in holds:
        specs.append(ExitSpec("time_only", mh, 0.0, 0.0, 0.0))
        for tp in tps:
            specs.append(ExitSpec("tp_timeout", mh, tp, 0.0, 0.0))
            for sl in sls:
                specs.append(ExitSpec("tp_sl_timeout", mh, tp, sl, 0.0))
    unique: dict[tuple[object, ...], ExitSpec] = {}
    for e in specs:
        unique[(e.exit_model, e.max_hold_bars, e.tp_pct, e.sl_pct, e.cvd_resume_mult)] = e
    return list(unique.values())


def select_stage2_event_names(probe_summary: pd.DataFrame, args: argparse.Namespace) -> tuple[set[str], pd.DataFrame]:
    """Select event names for full exit replay using broad predeclared triage.

    The thresholds are deliberately loose and audited in `18_stage1_selection.csv`.
    They prevent the research script from exploding into a parameter soup while
    keeping the best few structures per hypothesis/direction for validation.
    """
    if probe_summary.empty:
        return set(), pd.DataFrame()
    keys = ["event_name", "family", "hypothesis", "variant", "direction"]
    rows: list[dict[str, object]] = []
    for key, g in probe_summary.groupby(keys, sort=False):
        count_max = int(g["count"].max())
        best_mean_idx = g["mean_net"].astype(float).idxmax()
        best_pf_idx = g["profit_factor"].astype(float).idxmax()
        br = g.loc[best_mean_idx]
        pr = g.loc[best_pf_idx]
        rows.append(
            {
                "event_name": key[0],
                "family": key[1],
                "hypothesis": key[2],
                "variant": key[3],
                "direction": key[4],
                "probe_count_max": count_max,
                "probe_best_mean_net": float(br["mean_net"]),
                "probe_best_mean_exit_variant": str(br["exit_variant"]),
                "probe_best_pf": float(pr["profit_factor"]),
                "probe_best_pf_exit_variant": str(pr["exit_variant"]),
                "probe_best_win_rate": float(g["win_rate"].max()),
            }
        )
    sel = pd.DataFrame(rows)
    if sel.empty:
        return set(), sel
    enough = sel["probe_count_max"].astype(int) >= int(args.min_count)
    broad = enough & (
        (sel["probe_best_mean_net"].astype(float) >= float(args.stage1_event_mean_net_prefilter))
        | (sel["probe_best_pf"].astype(float) >= float(args.stage1_event_pf_prefilter))
    )
    sel["stage1_broad_pass"] = broad
    sel["stage1_top_keep"] = False
    top_n = max(0, int(args.stage1_top_event_specs_per_hypothesis_direction))
    if top_n > 0:
        for _, idxs in sel.loc[enough].groupby(["hypothesis", "direction"], sort=False).groups.items():
            idx_list = list(idxs)
            top = sel.loc[idx_list].sort_values(["probe_best_mean_net", "probe_best_pf"], ascending=False).head(top_n).index
            sel.loc[top, "stage1_top_keep"] = True
    sel["stage1_selected"] = sel["stage1_broad_pass"] | sel["stage1_top_keep"]
    selected = sel.loc[sel["stage1_selected"]].sort_values(
        ["probe_best_mean_net", "probe_best_pf", "probe_count_max"], ascending=[False, False, False]
    )
    max_specs = max(1, int(args.stage1_max_event_specs))
    if len(selected) > max_specs:
        keep_idx = selected.head(max_specs).index
        sel.loc[sel["stage1_selected"] & ~sel.index.isin(keep_idx), "stage1_selected"] = False
        sel["stage1_cap_dropped"] = sel.index.isin(selected.index) & ~sel.index.isin(keep_idx)
    else:
        sel["stage1_cap_dropped"] = False
    sel["stage1_reason"] = np.where(
        sel["stage1_selected"],
        np.where(sel["stage1_broad_pass"], "broad_prefilter_or_top_keep", "top_keep_only"),
        "stage1_rejected_for_full_exit_replay",
    )
    names = set(sel.loc[sel["stage1_selected"], "event_name"].astype(str))
    return names, sel.sort_values(["stage1_selected", "probe_best_mean_net", "probe_best_pf"], ascending=[False, False, False]).reset_index(drop=True)

def _apply_cooldown(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    if cooldown_bars <= 0 or mask.sum() <= 1:
        return mask
    idxs = np.flatnonzero(mask)
    keep: list[int] = []
    last = -10**18
    for i in idxs:
        if i - last > cooldown_bars:
            keep.append(int(i))
            last = int(i)
    out = np.zeros_like(mask, dtype=bool)
    out[np.asarray(keep, dtype=int)] = True
    return out


def _environment_mask(feat: pd.DataFrame, spec: EventSpec, impulse_prev: pd.Series, price_impulse_prev: pd.Series) -> pd.Series:
    env = spec.environment
    idx = feat.index
    if env == "all":
        return pd.Series(True, index=idx)
    if env == "low_vol_prior":
        return feat["prior_vol_ratio"].fillna(999.0) <= 0.80
    if env == "high_vol_now":
        return feat["vol_ratio"].fillna(0.0) >= 1.50
    if env == "trend_counter":
        trend = feat["trend_ret"].fillna(0.0)
        return ((trend > 0.003) & (impulse_prev < 0)) | ((trend < -0.003) & (impulse_prev > 0))
    if env == "trend_with":
        trend = feat["trend_ret"].fillna(0.0)
        return ((trend > 0.003) & (impulse_prev > 0)) | ((trend < -0.003) & (impulse_prev < 0))
    if env == "range_reclaim":
        # The impulse bar pierces a prior short range, and the stall bar closes back inside it.
        return (
            ((impulse_prev < 0) & (feat["low"].shift(1) < feat["rolling_low"]) & (feat["close"] > feat["rolling_low"]))
            | ((impulse_prev > 0) & (feat["high"].shift(1) > feat["rolling_high"]) & (feat["close"] < feat["rolling_high"]))
        )
    return pd.Series(True, index=idx)


def build_events_for_chunk(
    feat: pd.DataFrame,
    specs: Sequence[EventSpec],
    *,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    cooldown_bars: int,
) -> pd.DataFrame:
    if feat.empty:
        return pd.DataFrame()
    idx = feat.index
    in_chunk = (idx >= chunk_start) & (idx <= chunk_end)
    rows: list[pd.DataFrame] = []

    cvd = feat["cvd"].astype(float)
    cvd_delta_now = feat["cvd_delta_1"].astype(float)
    close = feat["close"].astype(float)
    notional = feat.get("notional", pd.Series(0.0, index=idx)).astype(float)
    delta_notional = feat.get("delta_notional", cvd_delta_now).astype(float)
    impulse_notional_prev = notional.shift(1)
    impulse_delta_abs_prev = delta_notional.abs().shift(1)
    impulse_notional_ratio = impulse_notional_prev / feat["notional_base"].replace(0, np.nan)
    impulse_delta_ratio = impulse_delta_abs_prev / feat["delta_abs_base"].replace(0, np.nan)

    for spec in specs:
        impulse_raw = cvd.diff(spec.impulse_bars)
        impulse_prev = impulse_raw.shift(1)
        slope_base = feat["cvd_delta_base"].astype(float) * math.sqrt(max(spec.impulse_bars, 1))
        price_impulse_prev = close.pct_change(spec.impulse_bars).shift(1)
        impulse_sign = np.sign(impulse_prev)

        extreme = impulse_prev.abs() >= (slope_base * spec.cvd_slope_mult)
        spike_notional = impulse_notional_ratio >= spec.impulse_notional_mult
        spike_delta = True if spec.impulse_delta_mult <= 0 else (impulse_delta_ratio >= spec.impulse_delta_mult)
        flat_vs_impulse = cvd_delta_now.abs() <= (impulse_prev.abs() * spec.flat_ratio)
        flat_vs_base = cvd_delta_now.abs() <= (feat["cvd_delta_base"].astype(float) * spec.flat_baseline_mult)
        stall = flat_vs_impulse & flat_vs_base
        if spec.hypothesis == "stall_reversal_confirm":
            stall = stall & ((np.sign(cvd_delta_now) * impulse_sign) < 0)

        if spec.price_mode == "align":
            price_ok = ((impulse_sign * price_impulse_prev) > 0) & (price_impulse_prev.abs() >= spec.min_abs_price_move_pct)
        elif spec.price_mode == "no_follow":
            price_ok = price_impulse_prev.abs() <= spec.min_abs_price_move_pct
        else:
            price_ok = pd.Series(True, index=idx)

        env_ok = _environment_mask(feat, spec, impulse_prev, price_impulse_prev)
        base_mask = extreme & spike_notional & spike_delta & stall & price_ok & env_ok & pd.Series(in_chunk, index=idx)
        long_mask = (base_mask & (impulse_prev < 0)).fillna(False).to_numpy(dtype=bool)
        short_mask = (base_mask & (impulse_prev > 0)).fillna(False).to_numpy(dtype=bool)
        long_mask = _apply_cooldown(long_mask, cooldown_bars)
        short_mask = _apply_cooldown(short_mask, cooldown_bars)

        for direction, mask in (("long", long_mask), ("short", short_mask)):
            pos = np.flatnonzero(mask)
            if pos.size == 0:
                continue
            ev = pd.DataFrame(
                {
                    "event_name": spec.event_name(direction),
                    "family": "cvd_impulse_spike_stall_reversal_5s_v1_1",
                    "hypothesis": spec.hypothesis,
                    "variant": spec.variant,
                    "direction": direction,
                    "signal_time": idx[pos],
                    "signal_pos": pos.astype(np.int64),
                    "impulse_bars": spec.impulse_bars,
                    "cvd_slope_mult": spec.cvd_slope_mult,
                    "flat_ratio": spec.flat_ratio,
                    "impulse_notional_mult": spec.impulse_notional_mult,
                    "impulse_delta_mult": spec.impulse_delta_mult,
                    "price_mode": spec.price_mode,
                    "environment": spec.environment,
                    "impulse_cvd_change": impulse_prev.iloc[pos].to_numpy(dtype=float),
                    "stall_cvd_delta": cvd_delta_now.iloc[pos].to_numpy(dtype=float),
                    "price_impulse_ret": price_impulse_prev.iloc[pos].to_numpy(dtype=float),
                    "impulse_notional_ratio": impulse_notional_ratio.iloc[pos].to_numpy(dtype=float),
                    "impulse_delta_ratio": impulse_delta_ratio.iloc[pos].to_numpy(dtype=float),
                    "year": feat["year"].iloc[pos].to_numpy(dtype=int),
                    "month": feat["month"].iloc[pos].to_numpy(dtype=int),
                    "session": feat["session"].iloc[pos].astype(str).to_numpy(),
                    "regime": feat["regime"].iloc[pos].astype(str).to_numpy(),
                    "volatility_bucket": feat["volatility_bucket"].iloc[pos].to_numpy(dtype=int),
                    "trend_bucket": feat["trend_bucket"].iloc[pos].astype(str).to_numpy(),
                }
            )
            rows.append(ev)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Replay / aggregation
# ---------------------------------------------------------------------------


def replay_positions(
    feat: pd.DataFrame,
    events: pd.DataFrame,
    *,
    exit_spec: ExitSpec,
    delay_bars: int,
    round_trip_cost_pct: float,
    cost_multiplier: float = 1.0,
    include_detail: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    n = len(feat)
    times = feat.index.to_numpy()
    open_px = feat["open"].to_numpy(dtype=float)
    high_px = feat["high"].to_numpy(dtype=float)
    low_px = feat["low"].to_numpy(dtype=float)
    cvd_delta = feat["cvd_delta_1"].to_numpy(dtype=float)
    cvd_base = feat["cvd_delta_base"].to_numpy(dtype=float)

    signal_pos0 = events["signal_pos"].to_numpy(dtype=np.int64)
    entry_pos = signal_pos0 + 1 + int(delay_bars)
    extra = 2 if exit_spec.exit_model == "tp_cvd_resume_timeout" else 1
    valid = (entry_pos >= 0) & ((entry_pos + int(exit_spec.max_hold_bars) + extra) < n)
    if not np.any(valid):
        return pd.DataFrame()
    ev = events.loc[valid].reset_index(drop=True)
    signal_pos = signal_pos0[valid]
    entry_pos = entry_pos[valid]
    entry_price = open_px[entry_pos]
    dirs = ev["direction"].astype(str).to_numpy()
    is_long = dirs == "long"
    impulse_change = ev["impulse_cvd_change"].to_numpy(dtype=float)
    impulse_sign = np.sign(impulse_change)

    m = len(ev)
    exit_pos = np.full(m, -1, dtype=np.int64)
    exit_price = np.full(m, np.nan, dtype=float)
    exit_reason = np.full(m, "time", dtype=object)
    same_bar_tp_sl = np.zeros(m, dtype=np.int8)
    hi_max = np.full(m, -np.inf, dtype=float)
    lo_min = np.full(m, np.inf, dtype=float)
    active = np.ones(m, dtype=bool)

    has_tp = exit_spec.exit_model in {"tp_timeout", "tp_sl_timeout", "tp_cvd_resume_timeout"} and exit_spec.tp_pct > 0
    has_sl = exit_spec.exit_model == "tp_sl_timeout" and exit_spec.sl_pct > 0
    tp_price = np.where(is_long, entry_price * (1.0 + exit_spec.tp_pct), entry_price * (1.0 - exit_spec.tp_pct))
    sl_price = np.where(is_long, entry_price * (1.0 - exit_spec.sl_pct), entry_price * (1.0 + exit_spec.sl_pct))

    for off in range(int(exit_spec.max_hold_bars)):
        p = entry_pos + off
        hi = high_px[p]
        lo = low_px[p]
        hi_max = np.maximum(hi_max, hi)
        lo_min = np.minimum(lo_min, lo)
        if not np.any(active):
            break
        tp_hit = np.zeros(m, dtype=bool)
        sl_hit = np.zeros(m, dtype=bool)
        if has_tp:
            tp_hit = np.where(is_long, hi >= tp_price, lo <= tp_price) & active
        if has_sl:
            sl_hit = np.where(is_long, lo <= sl_price, hi >= sl_price) & active
        both = tp_hit & sl_hit
        if np.any(both):
            same_bar_tp_sl[both] = 1
        # Conservative ordering: SL first if both TP and SL appear in the same 5s bar.
        sl_first = sl_hit
        tp_only = tp_hit & ~sl_hit
        if np.any(sl_first):
            exit_pos[sl_first] = p[sl_first]
            exit_price[sl_first] = sl_price[sl_first]
            exit_reason[sl_first] = "sl" if has_sl else "time"
            active[sl_first] = False
        if np.any(tp_only):
            exit_pos[tp_only] = p[tp_only]
            exit_price[tp_only] = tp_price[tp_only]
            exit_reason[tp_only] = "tp"
            active[tp_only] = False
        if exit_spec.exit_model == "tp_cvd_resume_timeout" and np.any(active):
            resume = (
                (np.sign(cvd_delta[p]) * impulse_sign) > 0
            ) & (np.abs(cvd_delta[p]) >= np.nan_to_num(cvd_base[p], nan=np.inf) * float(exit_spec.cvd_resume_mult)) & active
            if np.any(resume):
                next_pos = p[resume] + 1
                exit_pos[resume] = next_pos
                exit_price[resume] = open_px[next_pos]
                exit_reason[resume] = "cvd_resume"
                active[resume] = False

    timeout_pos = entry_pos + int(exit_spec.max_hold_bars)
    timeout = exit_pos < 0
    exit_pos[timeout] = timeout_pos[timeout]
    exit_price[timeout] = open_px[timeout_pos[timeout]]
    exit_reason[timeout] = "time"

    gross = np.where(is_long, exit_price / entry_price - 1.0, entry_price / exit_price - 1.0)
    net = gross - float(round_trip_cost_pct) * float(cost_multiplier)
    mfe = np.where(is_long, hi_max / entry_price - 1.0, entry_price / lo_min - 1.0)
    mae = np.where(is_long, lo_min / entry_price - 1.0, entry_price / hi_max - 1.0)

    data: dict[str, object] = {
        "event_name": ev["event_name"].to_numpy(),
        "family": ev["family"].to_numpy(),
        "hypothesis": ev["hypothesis"].to_numpy(),
        "variant": ev["variant"].to_numpy(),
        "direction": dirs,
        "exit_model": exit_spec.exit_model,
        "exit_variant": exit_spec.variant,
        "max_hold_bars": int(exit_spec.max_hold_bars),
        "tp_pct": float(exit_spec.tp_pct),
        "sl_pct": float(exit_spec.sl_pct),
        "cvd_resume_mult": float(exit_spec.cvd_resume_mult),
        "delay_bars": int(delay_bars),
        "signal_time": pd.to_datetime(ev["signal_time"]).to_numpy(),
        "entry_time": pd.to_datetime(times[entry_pos]).to_numpy(),
        "exit_time": pd.to_datetime(times[exit_pos]).to_numpy(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "tp_hit": exit_reason == "tp",
        "sl_hit": exit_reason == "sl",
        "same_bar_tp_sl_flag": same_bar_tp_sl,
        "gross_return": gross,
        "net_return": net,
        "mfe": mfe,
        "mae": mae,
        "year": ev["year"].to_numpy(dtype=int),
        "month": ev["month"].to_numpy(dtype=int),
        "session": ev["session"].astype(str).to_numpy(),
        "regime": ev["regime"].astype(str).to_numpy(),
        "volatility_bucket": ev["volatility_bucket"].to_numpy(dtype=int),
        "trend_bucket": ev["trend_bucket"].astype(str).to_numpy(),
        "expected_entry_time": pd.to_datetime(times[signal_pos + 1 + int(delay_bars)]).to_numpy(),
        "entry_not_next_open_flag": 0,
        "forward_window_valid_flag": True,
        "lookahead_flag": 0,
    }
    if include_detail:
        data.update(
            {
                "impulse_bars": ev["impulse_bars"].to_numpy(dtype=int),
                "cvd_slope_mult": ev["cvd_slope_mult"].to_numpy(dtype=float),
                "flat_ratio": ev["flat_ratio"].to_numpy(dtype=float),
                "impulse_notional_mult": ev["impulse_notional_mult"].to_numpy(dtype=float),
                "impulse_delta_mult": ev["impulse_delta_mult"].to_numpy(dtype=float),
                "price_mode": ev["price_mode"].astype(str).to_numpy(),
                "environment": ev["environment"].astype(str).to_numpy(),
                "impulse_cvd_change": ev["impulse_cvd_change"].to_numpy(dtype=float),
                "stall_cvd_delta": ev["stall_cvd_delta"].to_numpy(dtype=float),
                "price_impulse_ret": ev["price_impulse_ret"].to_numpy(dtype=float),
                "impulse_notional_ratio": ev["impulse_notional_ratio"].to_numpy(dtype=float),
                "impulse_delta_ratio": ev["impulse_delta_ratio"].to_numpy(dtype=float),
            }
        )
    return pd.DataFrame(data)


def _group_keys() -> list[str]:
    return ["event_name", "family", "hypothesis", "variant", "direction", "exit_model", "exit_variant", "max_hold_bars", "tp_pct", "sl_pct"]


def summarize_replay(trades: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    years = _annualized_years(start_date, end_date)
    rows: list[dict[str, object]] = []
    for key, g in trades.groupby(_group_keys(), sort=False):
        net = g["net_return"].to_numpy(dtype=float)
        gross = g["gross_return"].to_numpy(dtype=float)
        yearly = g.groupby("year")["net_return"].mean()
        rows.append(
            {
                "event_name": key[0], "family": key[1], "hypothesis": key[2], "variant": key[3], "direction": key[4],
                "exit_model": key[5], "exit_variant": key[6], "max_hold_bars": int(key[7]), "tp_pct": float(key[8]), "sl_pct": float(key[9]),
                "count": int(len(g)),
                "events_per_year": float(len(g) / years),
                "events_per_month": float(len(g) / max(years * 12.0, 1e-9)),
                "mean_gross": float(np.nanmean(gross)),
                "mean_net": float(np.nanmean(net)),
                "median_net": float(np.nanmedian(net)),
                "win_rate": float(np.nanmean(net > 0)),
                "profit_factor": _profit_factor(net),
                "tp_hit_rate": float(g["tp_hit"].mean()) if "tp_hit" in g else float("nan"),
                "sl_hit_rate": float(g["sl_hit"].mean()) if "sl_hit" in g else float("nan"),
                "time_exit_rate": float((g["exit_reason"] == "time").mean()) if "exit_reason" in g else float("nan"),
                "cvd_resume_exit_rate": float((g["exit_reason"] == "cvd_resume").mean()) if "exit_reason" in g else float("nan"),
                "same_bar_tp_sl_rate": float(g["same_bar_tp_sl_flag"].mean()) if "same_bar_tp_sl_flag" in g else 0.0,
                "mfe_mean": float(g["mfe"].mean()),
                "mae_mean": float(g["mae"].mean()),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.shape[0]),
                "max_days_without_event": _max_days_without_event(g["signal_time"]),
                "top5_winner_share": _top_winner_share(net),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_net", "profit_factor"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streaming aggregation helpers
# ---------------------------------------------------------------------------

_AGG_NUMERIC_COLS = [
    "count", "sum_gross", "sum_net", "win_count", "gain_sum", "loss_sum",
    "tp_hit_count", "sl_hit_count", "time_exit_count", "cvd_resume_exit_count",
    "same_bar_tp_sl_count", "sum_mfe", "sum_mae", "top_gain_1", "top_gain_2",
    "top_gain_3", "top_gain_4", "top_gain_5", "max_gap_seconds",
]


def _top5_values(v: np.ndarray | pd.Series) -> list[float]:
    arr = np.asarray(v, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return [0.0] * 5
    if arr.size > 5:
        arr = np.partition(arr, -5)[-5:]
    arr = np.sort(arr)[::-1]
    out = arr.tolist()[:5]
    return out + [0.0] * (5 - len(out))


def _max_gap_seconds(times: pd.Series) -> float:
    if len(times) <= 1:
        return 0.0
    t = pd.to_datetime(times).sort_values().to_numpy(dtype="datetime64[ns]")
    if t.size <= 1:
        return 0.0
    d = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return float(np.nanmax(d)) if d.size else 0.0


def aggregate_trade_stats(trades: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate a replay chunk without retaining all rows in memory.

    This is exact for mean/win/PF/rates/yearly means and conservative for
    max-days-without-event via first/last/max-gap combination across chunks.
    Median is intentionally not computed from huge 5s replay rows; downstream
    reports expose `median_net` as NaN for streaming summaries instead of
    silently using a biased sample median.
    """
    if trades.empty:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in trades.columns]
    if not cols:
        return pd.DataFrame()
    tmp = trades.copy()
    net = tmp["net_return"].astype(float)
    gross = tmp["gross_return"].astype(float)
    tmp["_count"] = 1
    tmp["_sum_gross"] = gross
    tmp["_sum_net"] = net
    tmp["_win_count"] = (net > 0).astype(np.int64)
    tmp["_gain_sum"] = net.clip(lower=0.0)
    tmp["_loss_sum"] = (-net.clip(upper=0.0))
    tmp["_tp_hit_count"] = tmp.get("tp_hit", pd.Series(False, index=tmp.index)).astype(bool).astype(np.int64)
    tmp["_sl_hit_count"] = tmp.get("sl_hit", pd.Series(False, index=tmp.index)).astype(bool).astype(np.int64)
    reason = tmp.get("exit_reason", pd.Series("", index=tmp.index)).astype(str)
    tmp["_time_exit_count"] = (reason == "time").astype(np.int64)
    tmp["_cvd_resume_exit_count"] = (reason == "cvd_resume").astype(np.int64)
    tmp["_same_bar_tp_sl_count"] = tmp.get("same_bar_tp_sl_flag", pd.Series(0, index=tmp.index)).astype(float)
    tmp["_sum_mfe"] = tmp.get("mfe", pd.Series(np.nan, index=tmp.index)).astype(float).fillna(0.0)
    tmp["_sum_mae"] = tmp.get("mae", pd.Series(np.nan, index=tmp.index)).astype(float).fillna(0.0)

    agg = tmp.groupby(cols, sort=False, observed=True).agg(
        count=("_count", "sum"),
        sum_gross=("_sum_gross", "sum"),
        sum_net=("_sum_net", "sum"),
        win_count=("_win_count", "sum"),
        gain_sum=("_gain_sum", "sum"),
        loss_sum=("_loss_sum", "sum"),
        tp_hit_count=("_tp_hit_count", "sum"),
        sl_hit_count=("_sl_hit_count", "sum"),
        time_exit_count=("_time_exit_count", "sum"),
        cvd_resume_exit_count=("_cvd_resume_exit_count", "sum"),
        same_bar_tp_sl_count=("_same_bar_tp_sl_count", "sum"),
        sum_mfe=("_sum_mfe", "sum"),
        sum_mae=("_sum_mae", "sum"),
        first_signal_time=("signal_time", "min"),
        last_signal_time=("signal_time", "max"),
    ).reset_index()
    top_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for key, g in tmp.groupby(cols, sort=False, observed=True):
        kk = key if isinstance(key, tuple) else (key,)
        row = {c: v for c, v in zip(cols, kk)}
        tops = _top5_values(g["net_return"])
        for i, val in enumerate(tops, start=1):
            row[f"top_gain_{i}"] = float(val)
        top_rows.append(row)
        grow = {c: v for c, v in zip(cols, kk)}
        grow["max_gap_seconds"] = _max_gap_seconds(g["signal_time"])
        gap_rows.append(grow)
    if top_rows:
        agg = agg.merge(pd.DataFrame(top_rows), on=cols, how="left")
    if gap_rows:
        agg = agg.merge(pd.DataFrame(gap_rows), on=cols, how="left")
    for c in [f"top_gain_{i}" for i in range(1, 6)] + ["max_gap_seconds"]:
        if c not in agg.columns:
            agg[c] = 0.0
    return agg


def combine_trade_stats(parts: Sequence[pd.DataFrame], group_cols: Sequence[str], *, start_date: str, end_date: str) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    cols = [c for c in group_cols if c in parts[0].columns]
    raw = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in _AGG_NUMERIC_COLS if c in raw.columns and c not in {"max_gap_seconds", "top_gain_1", "top_gain_2", "top_gain_3", "top_gain_4", "top_gain_5"}]
    agg = raw.groupby(cols, sort=False, observed=True)[sum_cols].sum().reset_index()
    first_last = raw.groupby(cols, sort=False, observed=True).agg(
        first_signal_time=("first_signal_time", "min"),
        last_signal_time=("last_signal_time", "max"),
    ).reset_index()
    agg = agg.merge(first_last, on=cols, how="left")

    # combine max gap exactly from chunk first/last/max gap in chronological order
    gap_rows: list[dict[str, object]] = []
    for key, g in raw.groupby(cols, sort=False, observed=True):
        kk = key if isinstance(key, tuple) else (key,)
        gg = g.sort_values("first_signal_time")
        max_gap = float(gg.get("max_gap_seconds", pd.Series(0.0, index=gg.index)).max())
        prev_last = None
        for _, r in gg.iterrows():
            if prev_last is not None:
                gap = (pd.Timestamp(r["first_signal_time"]) - pd.Timestamp(prev_last)).total_seconds()
                max_gap = max(max_gap, float(gap))
            prev_last = r["last_signal_time"]
        row = {c: v for c, v in zip(cols, kk)}
        row["max_days_without_event"] = float(max_gap / 86400.0)
        gap_rows.append(row)
    if gap_rows:
        agg = agg.merge(pd.DataFrame(gap_rows), on=cols, how="left")
    else:
        agg["max_days_without_event"] = np.nan

    # combine top-5 winners from chunk top-5 candidates
    top_rows: list[dict[str, object]] = []
    top_cols = [f"top_gain_{i}" for i in range(1, 6)]
    for key, g in raw.groupby(cols, sort=False, observed=True):
        kk = key if isinstance(key, tuple) else (key,)
        vals = g[[c for c in top_cols if c in g.columns]].to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size > 5:
            vals = np.partition(vals, -5)[-5:]
        vals = np.sort(vals)[::-1]
        top_sum = float(vals[:5].sum()) if vals.size else 0.0
        row = {c: v for c, v in zip(cols, kk)}
        row["top5_winner_share"] = top_sum
        top_rows.append(row)
    if top_rows:
        agg = agg.merge(pd.DataFrame(top_rows), on=cols, how="left")
    agg["top5_winner_share"] = agg["top5_winner_share"] / agg["gain_sum"].replace(0, np.nan)
    agg["top5_winner_share"] = agg["top5_winner_share"].fillna(0.0)

    years = _annualized_years(start_date, end_date)
    agg["events_per_year"] = agg["count"] / years
    agg["events_per_month"] = agg["count"] / max(years * 12.0, 1e-9)
    agg["mean_gross"] = agg["sum_gross"] / agg["count"].replace(0, np.nan)
    agg["mean_net"] = agg["sum_net"] / agg["count"].replace(0, np.nan)
    agg["median_net"] = np.nan
    agg["win_rate"] = agg["win_count"] / agg["count"].replace(0, np.nan)
    agg["profit_factor"] = agg["gain_sum"] / agg["loss_sum"].replace(0, np.nan)
    pf_fill = pd.Series(np.where(agg["gain_sum"] > 0, np.inf, 0.0), index=agg.index)
    agg["profit_factor"] = agg["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(pf_fill)
    agg["tp_hit_rate"] = agg["tp_hit_count"] / agg["count"].replace(0, np.nan)
    agg["sl_hit_rate"] = agg["sl_hit_count"] / agg["count"].replace(0, np.nan)
    agg["time_exit_rate"] = agg["time_exit_count"] / agg["count"].replace(0, np.nan)
    agg["cvd_resume_exit_rate"] = agg["cvd_resume_exit_count"] / agg["count"].replace(0, np.nan)
    agg["same_bar_tp_sl_rate"] = agg["same_bar_tp_sl_count"] / agg["count"].replace(0, np.nan)
    agg["mfe_mean"] = agg["sum_mfe"] / agg["count"].replace(0, np.nan)
    agg["mae_mean"] = agg["sum_mae"] / agg["count"].replace(0, np.nan)
    return agg


def aggregate_year_stats(trades: pd.DataFrame) -> pd.DataFrame:
    return aggregate_trade_stats(trades, _group_keys() + ["year"])


def combine_year_stats(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    cols = _group_keys() + ["year"]
    sum_cols = ["count", "sum_net", "win_count"]
    out = raw.groupby(cols, sort=False, observed=True)[sum_cols].sum().reset_index()
    out["mean_net"] = out["sum_net"] / out["count"].replace(0, np.nan)
    out["win_rate"] = out["win_count"] / out["count"].replace(0, np.nan)
    return out


def attach_positive_years(summary: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    if yearly.empty:
        out["positive_years"] = 0
        out["year_count"] = 0
        return out
    y = yearly.groupby(_group_keys(), sort=False, observed=True).agg(
        positive_years=("mean_net", lambda s: int((s > 0).sum())),
        year_count=("year", "nunique"),
    ).reset_index()
    out = out.merge(y, on=_group_keys(), how="left")
    out["positive_years"] = out["positive_years"].fillna(0).astype(int)
    out["year_count"] = out["year_count"].fillna(0).astype(int)
    return out


def aggregate_cost_stats(trades: pd.DataFrame, cost_multipliers: Sequence[float], round_trip_cost_pct: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    gross = trades["gross_return"].astype(float)
    for cm in cost_multipliers:
        t = trades.copy()
        t["net_return"] = gross - float(round_trip_cost_pct) * float(cm)
        st = aggregate_trade_stats(t, _group_keys())
        if not st.empty:
            st["cost_multiplier"] = float(cm)
            parts.append(st)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def finalize_cost_stats(parts: Sequence[pd.DataFrame], *, start_date: str, end_date: str) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    out_parts = []
    for cm, g in raw.groupby("cost_multiplier", sort=False):
        c = combine_trade_stats([g.drop(columns=["cost_multiplier"])], _group_keys(), start_date=start_date, end_date=end_date)
        if not c.empty:
            c["cost_multiplier"] = float(cm)
            out_parts.append(c)
    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    keep = _group_keys() + ["cost_multiplier", "count", "mean_net", "profit_factor", "win_rate"]
    return out[[c for c in keep if c in out.columns]] if not out.empty else out


def finalize_delay_stats(parts: Sequence[pd.DataFrame], *, start_date: str, end_date: str) -> pd.DataFrame:
    out = combine_trade_stats(parts, _group_keys() + ["delay_bars"], start_date=start_date, end_date=end_date)
    if out.empty:
        return out
    keep = _group_keys() + ["delay_bars", "count", "mean_net", "profit_factor", "win_rate"]
    return out[[c for c in keep if c in out.columns]]


def finalize_breakdown_stats(parts: Sequence[pd.DataFrame], group_cols: Sequence[str], *, start_date: str, end_date: str) -> pd.DataFrame:
    out = combine_trade_stats(parts, group_cols, start_date=start_date, end_date=end_date)
    if out.empty:
        return out
    keep = list(group_cols) + [
        "count", "mean_net", "win_rate", "profit_factor", "tp_hit_rate", "sl_hit_rate", "same_bar_tp_sl_rate"
    ]
    return out[[c for c in keep if c in out.columns]]


def combine_event_summary(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    keys = ["hypothesis", "event_name", "family", "variant", "direction"]
    out = raw.groupby(keys, sort=False, observed=True).agg(
        count=("count", "sum"),
        first_signal=("first_signal", "min"),
        last_signal=("last_signal", "max"),
        impulse_notional_ratio_sum=("impulse_notional_ratio_sum", "sum"),
        impulse_delta_ratio_sum=("impulse_delta_ratio_sum", "sum"),
    ).reset_index()
    out["impulse_notional_ratio_mean"] = out["impulse_notional_ratio_sum"] / out["count"].replace(0, np.nan)
    out["impulse_delta_ratio_mean"] = out["impulse_delta_ratio_sum"] / out["count"].replace(0, np.nan)
    return out.drop(columns=["impulse_notional_ratio_sum", "impulse_delta_ratio_sum"])


def summarize_events_chunk(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    keys = ["hypothesis", "event_name", "family", "variant", "direction"]
    tmp = events.copy()
    tmp["impulse_notional_ratio_sum"] = tmp["impulse_notional_ratio"].astype(float)
    tmp["impulse_delta_ratio_sum"] = tmp["impulse_delta_ratio"].astype(float)
    return tmp.groupby(keys, sort=False, observed=True).agg(
        count=("signal_time", "size"),
        first_signal=("signal_time", "min"),
        last_signal=("signal_time", "max"),
        impulse_notional_ratio_sum=("impulse_notional_ratio_sum", "sum"),
        impulse_delta_ratio_sum=("impulse_delta_ratio_sum", "sum"),
    ).reset_index()


def append_sample(parts: list[pd.DataFrame], df: pd.DataFrame, limit: int) -> None:
    if df.empty or limit <= 0:
        return
    current = sum(len(p) for p in parts)
    if current >= limit:
        return
    parts.append(df.head(limit - current).copy())


def _merge_keys_no_delay() -> list[str]:
    return ["event_name", "direction", "exit_model", "exit_variant", "max_hold_bars", "tp_pct", "sl_pct"]


def build_cost_stress(base_trades: pd.DataFrame, cost_multipliers: Sequence[float], round_trip_cost_pct: float) -> pd.DataFrame:
    if base_trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, g in base_trades.groupby(_group_keys(), sort=False):
        gross = g["gross_return"].to_numpy(dtype=float)
        for cm in cost_multipliers:
            net = gross - round_trip_cost_pct * float(cm)
            rows.append(
                {
                    "event_name": key[0], "family": key[1], "hypothesis": key[2], "variant": key[3], "direction": key[4],
                    "exit_model": key[5], "exit_variant": key[6], "max_hold_bars": int(key[7]), "tp_pct": float(key[8]), "sl_pct": float(key[9]),
                    "cost_multiplier": float(cm),
                    "count": int(len(net)),
                    "mean_net": float(np.nanmean(net)),
                    "profit_factor": _profit_factor(net),
                    "win_rate": float(np.nanmean(net > 0)),
                }
            )
    return pd.DataFrame(rows)


def build_delay_stress(delay_trades: pd.DataFrame) -> pd.DataFrame:
    if delay_trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = _group_keys() + ["delay_bars"]
    for key, g in delay_trades.groupby(keys, sort=False):
        net = g["net_return"].to_numpy(dtype=float)
        rows.append(
            {
                "event_name": key[0], "family": key[1], "hypothesis": key[2], "variant": key[3], "direction": key[4],
                "exit_model": key[5], "exit_variant": key[6], "max_hold_bars": int(key[7]), "tp_pct": float(key[8]), "sl_pct": float(key[9]),
                "delay_bars": int(key[10]),
                "count": int(len(g)),
                "mean_net": float(np.nanmean(net)),
                "profit_factor": _profit_factor(net),
                "win_rate": float(np.nanmean(net > 0)),
            }
        )
    return pd.DataFrame(rows)


def _merge_fee_delay(summary: pd.DataFrame, cost_stress: pd.DataFrame, delay_stress: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    keys = _merge_keys_no_delay()
    if not cost_stress.empty:
        fee2 = cost_stress[np.isclose(cost_stress["cost_multiplier"].astype(float), 2.0)][keys + ["mean_net"]].rename(columns={"mean_net": "fee2_mean_net"})
        out = out.merge(fee2, on=keys, how="left")
    else:
        out["fee2_mean_net"] = np.nan
    if not delay_stress.empty:
        d1 = delay_stress[delay_stress["delay_bars"].astype(int) == 1][keys + ["mean_net"]].rename(columns={"mean_net": "delay1_mean_net"})
        out = out.merge(d1, on=keys, how="left")
    else:
        out["delay1_mean_net"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Baseline / decisions
# ---------------------------------------------------------------------------


def _baseline_candidate_groups(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return summary
    cond = (
        (summary["count"] >= int(args.min_count))
        & (summary["mean_net"] >= float(args.baseline_prefilter_mean_net))
        & (summary["profit_factor"] >= float(args.baseline_prefilter_pf))
    )
    return summary.loc[cond].copy().reset_index(drop=True)


def build_baseline_summary(
    args: argparse.Namespace,
    prelim: pd.DataFrame,
    base_trades: pd.DataFrame,
    all_event_times: set[pd.Timestamp],
    exit_specs: Sequence[ExitSpec],
) -> pd.DataFrame:
    if prelim.empty:
        print("[aggregate] matched baseline skipped: no groups passed preliminary filters", flush=True)
        return pd.DataFrame()
    print(f"[aggregate] matched baseline groups={len(prelim)}", flush=True)
    rng = np.random.default_rng(int(args.baseline_seed))
    chunks = _month_chunks(args.start_date, args.end_date)
    max_hold_needed = int(max(prelim["max_hold_bars"].astype(int).max(), max(_parse_csv_ints(args.max_hold_bars_list))))
    lookback_bars = int(args.cvd_norm_window) + max(_parse_csv_ints(args.impulse_bars_list)) + 20
    load_lookback = pd.Timedelta(seconds=max(lookback_bars * 5, 3600))
    load_forward = pd.Timedelta(seconds=(max_hold_needed + max(_parse_csv_ints(args.delay_bars_list)) + 5) * 5)
    rows: list[dict[str, object]] = []

    with ProgressReporter("[aggregate] matched baseline", total=len(prelim), every=max(1, len(prelim) // 10 or 1)) as pr:
        for gi, g in prelim.iterrows():
            event_name = str(g["event_name"])
            direction = str(g["direction"])
            exit_spec = ExitSpec(
                exit_model=str(g["exit_model"]),
                max_hold_bars=int(g["max_hold_bars"]),
                tp_pct=float(g["tp_pct"]),
                sl_pct=float(g["sl_pct"]),
                cvd_resume_mult=float(str(g.get("exit_variant", "rm0")).split("_rm")[-1].replace("p", ".")) if "_rm" in str(g.get("exit_variant", "")) else 0.0,
            )
            ev = base_trades[
                (base_trades["event_name"] == event_name)
                & (base_trades["direction"] == direction)
                & (base_trades["exit_variant"] == str(g["exit_variant"]))
            ]
            if ev.empty:
                pr.update(int(gi) + 1)
                continue
            sample_ev = ev.sample(n=min(len(ev), int(args.baseline_max_events_per_group)), random_state=int(args.baseline_seed))
            key_counts = sample_ev.groupby(list(MATCHED_BASELINE_COLUMNS), sort=False).size().to_dict()
            target_per_key = {k if isinstance(k, tuple) else (k,): int(v) * int(args.baseline_samples) for k, v in key_counts.items()}
            pseudo_parts: list[pd.DataFrame] = []
            for chunk_start, chunk_end in chunks:
                load_start = max(pd.Timestamp(args.warmup_start_date), chunk_start - load_lookback)
                load_end = chunk_end + load_forward
                raw = load_trade_bars_range(args, load_start, load_end)
                if raw.empty:
                    continue
                feat, _ = build_features(
                    raw,
                    cvd_norm_window=int(args.cvd_norm_window),
                    notional_window=int(args.notional_window),
                    range_window=int(args.range_window),
                    trend_window=int(args.trend_window),
                )
                in_chunk = (feat.index >= chunk_start) & (feat.index <= chunk_end)
                pos_all = np.flatnonzero(in_chunk)
                if pos_all.size == 0:
                    continue
                times = pd.to_datetime(feat.index[pos_all])
                not_event = ~pd.Series(times).isin(all_event_times).to_numpy(dtype=bool)
                pos_all = pos_all[not_event]
                if pos_all.size == 0:
                    continue
                key_frame = pd.DataFrame(
                    {
                        "year": feat["year"].iloc[pos_all].to_numpy(dtype=int),
                        "month": feat["month"].iloc[pos_all].to_numpy(dtype=int),
                        "session": feat["session"].iloc[pos_all].astype(str).to_numpy(),
                        "regime": feat["regime"].iloc[pos_all].astype(str).to_numpy(),
                        "volatility_bucket": feat["volatility_bucket"].iloc[pos_all].to_numpy(dtype=int),
                        "trend_bucket": feat["trend_bucket"].iloc[pos_all].astype(str).to_numpy(),
                        "direction": direction,
                    }
                )
                baseline_events: list[pd.DataFrame] = []
                for key, idxs in key_frame.groupby(list(MATCHED_BASELINE_COLUMNS), sort=False).indices.items():
                    kk = key if isinstance(key, tuple) else (key,)
                    need = target_per_key.get(kk, 0)
                    if need <= 0:
                        continue
                    pool_pos = pos_all[np.asarray(idxs, dtype=np.int64)]
                    take = min(len(pool_pos), need)
                    if take <= 0:
                        continue
                    chosen = rng.choice(pool_pos, size=take, replace=len(pool_pos) < take)
                    be = pd.DataFrame(
                        {
                            "event_name": event_name,
                            "family": "matched_baseline",
                            "hypothesis": "matched_baseline",
                            "variant": "matched_baseline",
                            "direction": direction,
                            "signal_time": feat.index[chosen],
                            "signal_pos": chosen.astype(np.int64),
                            "impulse_cvd_change": np.where(direction == "long", -1.0, 1.0),
                            "year": feat["year"].iloc[chosen].to_numpy(dtype=int),
                            "month": feat["month"].iloc[chosen].to_numpy(dtype=int),
                            "session": feat["session"].iloc[chosen].astype(str).to_numpy(),
                            "regime": feat["regime"].iloc[chosen].astype(str).to_numpy(),
                            "volatility_bucket": feat["volatility_bucket"].iloc[chosen].to_numpy(dtype=int),
                            "trend_bucket": feat["trend_bucket"].iloc[chosen].astype(str).to_numpy(),
                        }
                    )
                    baseline_events.append(be)
                if baseline_events:
                    bev = pd.concat(baseline_events, ignore_index=True)
                    pseudo = replay_positions(
                        feat,
                        bev,
                        exit_spec=exit_spec,
                        delay_bars=0,
                        round_trip_cost_pct=float(args.round_trip_cost_pct),
                        cost_multiplier=1.0,
                        include_detail=False,
                    )
                    if not pseudo.empty:
                        pseudo_parts.append(pseudo[["net_return", "gross_return", "tp_hit", "sl_hit"]])
            if pseudo_parts:
                p = pd.concat(pseudo_parts, ignore_index=True)
                event_mean = float(g["mean_net"])
                base_mean = float(p["net_return"].mean())
                rows.append(
                    {
                        "event_name": event_name,
                        "direction": direction,
                        "exit_model": str(g["exit_model"]),
                        "exit_variant": str(g["exit_variant"]),
                        "max_hold_bars": int(g["max_hold_bars"]),
                        "tp_pct": float(g["tp_pct"]),
                        "sl_pct": float(g["sl_pct"]),
                        "event_count": int(g["count"]),
                        "baseline_count": int(len(p)),
                        "event_mean_net": event_mean,
                        "baseline_mean_net": base_mean,
                        "matched_excess_mean_net": float(event_mean - base_mean),
                        "baseline_profit_factor": _profit_factor(p["net_return"].to_numpy(dtype=float)),
                        "baseline_win_rate": float((p["net_return"] > 0).mean()),
                        "baseline_tp_hit_rate": float(p["tp_hit"].mean()) if "tp_hit" in p else float("nan"),
                        "baseline_sl_hit_rate": float(p["sl_hit"].mean()) if "sl_hit" in p else float("nan"),
                    }
                )
            pr.update(int(gi) + 1)
    return pd.DataFrame(rows)


def build_decisions(summary: pd.DataFrame, baseline: pd.DataFrame, *, min_count: int, min_events_per_year: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    out = summary.copy()
    bkeys = _merge_keys_no_delay()
    if not baseline.empty:
        out = out.merge(
            baseline[bkeys + ["matched_excess_mean_net", "baseline_mean_net", "baseline_count"]],
            on=bkeys,
            how="left",
        )
    else:
        out["matched_excess_mean_net"] = np.nan
        out["baseline_mean_net"] = np.nan
        out["baseline_count"] = 0
    decisions: list[str] = []
    reasons: list[str] = []
    for _, r in out.iterrows():
        fails: list[str] = []
        if int(r["count"]) < min_count:
            fails.append("count_lt_min")
        if float(r["events_per_year"]) < min_events_per_year:
            fails.append("events_per_year_lt_min")
        if float(r["mean_net"]) <= 0:
            fails.append("mean_net_le_0")
        if float(r["profit_factor"]) < 1.15:
            fails.append("pf_lt_1p15")
        if pd.notna(r.get("fee2_mean_net")) and float(r["fee2_mean_net"]) <= 0:
            fails.append("fee2_le_0")
        if pd.notna(r.get("delay1_mean_net")) and float(r["delay1_mean_net"]) <= -0.0002:
            fails.append("delay1_too_weak")
        if int(r["positive_years"]) < 3:
            fails.append("positive_years_lt_3")
        if float(r["top5_winner_share"]) > 0.35:
            fails.append("top5_winner_share_gt_0p35")
        if float(r.get("same_bar_tp_sl_rate", 0.0)) > 0.10:
            fails.append("same_bar_tp_sl_rate_gt_0p10")
        if pd.notna(r.get("matched_excess_mean_net")) and float(r["matched_excess_mean_net"]) <= 0:
            fails.append("matched_excess_le_0")
        if not fails:
            decisions.append("promote_to_backtest_candidate")
            reasons.append("passed_research_filters")
        elif float(r["mean_net"]) > 0 and float(r["profit_factor"]) >= 1.05:
            decisions.append("research_continue")
            reasons.append(";".join(fails))
        else:
            decisions.append("rejected")
            reasons.append(";".join(fails))
    out["decision"] = decisions
    out["reason"] = reasons
    candidates = out[out["decision"] == "promote_to_backtest_candidate"].copy()
    rejected = out[out["decision"] == "rejected"].copy()
    return out.sort_values(["decision", "mean_net"], ascending=[True, False]), candidates, rejected


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def write_reports(
    args: argparse.Namespace,
    *,
    specs: Sequence[EventSpec],
    exit_specs: Sequence[ExitSpec],
    feature_columns: Sequence[str],
    input_rows: int,
    events: pd.DataFrame,
    base_trades: pd.DataFrame,
    summary: pd.DataFrame,
    cost_stress: pd.DataFrame,
    delay_stress: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    stage1_selection: pd.DataFrame | None = None,
    stage2_mode: str = "two_stage",
    event_summary_override: pd.DataFrame | None = None,
    event_count_override: int | None = None,
    replay_trade_count_override: int | None = None,
    report_table_overrides: dict[str, pd.DataFrame] | None = None,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary2 = _merge_fee_delay(summary, cost_stress, delay_stress)
    decisions, candidates, rejected = build_decisions(summary2, baseline_summary, min_count=int(args.min_count), min_events_per_year=float(args.min_events_per_year))
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "edge_is_tradable": False,
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "tp_pct_list": list(_parse_csv_floats(args.tp_pct_list)),
        "sl_pct_list": list(_parse_csv_floats(args.sl_pct_list)),
        "max_hold_bars_list": list(_parse_csv_ints(args.max_hold_bars_list)),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(_parse_csv_floats(args.cost_multipliers)),
        "delay_bars_list": list(_parse_csv_ints(args.delay_bars_list)),
        "input_rows_loaded_with_overlap": int(input_rows),
        "event_spec_count": int(len(specs) * 2),
        "exit_spec_count": int(len(exit_specs)),
        "stage2_mode": stage2_mode,
        "stage1_selected_event_count": int(stage1_selection["stage1_selected"].sum()) if stage1_selection is not None and not stage1_selection.empty and "stage1_selected" in stage1_selection.columns else None,
        "event_count": int(event_count_override) if event_count_override is not None else int(len(events)),
        "replay_trade_count": int(replay_trade_count_override) if replay_trade_count_override is not None else int(len(base_trades)),
        "replay_group_count": int(len(summary2)),
        "candidate_count": int(len(candidates)),
        "causal_lookahead_count": int(base_trades.get("lookahead_flag", pd.Series(dtype=int)).sum()) if not base_trades.empty else 0,
        "causal_policy": CAUSAL_POLICY,
        "matched_baseline_columns": list(MATCHED_BASELINE_COLUMNS),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    if event_summary_override is not None and not event_summary_override.empty:
        event_summary_override.to_csv(out_dir / "01_event_summary.csv", index=False)
    elif not events.empty:
        ev_summary = events.groupby(["hypothesis", "event_name", "family", "variant", "direction"], sort=False).agg(
            count=("signal_time", "size"), first_signal=("signal_time", "min"), last_signal=("signal_time", "max"),
            impulse_notional_ratio_mean=("impulse_notional_ratio", "mean"), impulse_delta_ratio_mean=("impulse_delta_ratio", "mean"),
        ).reset_index()
        ev_summary.to_csv(out_dir / "01_event_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "01_event_summary.csv", index=False)
    summary2.to_csv(out_dir / "02_replay_variant_summary.csv", index=False)
    if args.write_full_trades:
        base_trades.to_csv(out_dir / "03_replay_trades.csv", index=False)
    else:
        base_trades.head(int(args.trade_sample_size)).to_csv(out_dir / "03_replay_trades_sample.csv", index=False)
    override_names = ["04_yearly_breakdown.csv", "05_session_breakdown.csv", "06_regime_breakdown.csv", "15_hypothesis_environment_summary.csv", "16_exit_structure_summary.csv", "17_environment_breakdown.csv"]
    overrides = report_table_overrides or {}
    if overrides:
        for name in override_names:
            tbl = overrides.get(name, pd.DataFrame())
            tbl.to_csv(out_dir / name, index=False)
    elif not base_trades.empty:
        base_trades.groupby(["hypothesis", "event_name", "direction", "exit_variant", "year"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda s: float((s > 0).mean()))
        ).reset_index().to_csv(out_dir / "04_yearly_breakdown.csv", index=False)
        base_trades.groupby(["hypothesis", "direction", "session", "exit_model"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda s: float((s > 0).mean()))
        ).reset_index().to_csv(out_dir / "05_session_breakdown.csv", index=False)
        base_trades.groupby(["hypothesis", "direction", "regime", "exit_model"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda s: float((s > 0).mean()))
        ).reset_index().to_csv(out_dir / "06_regime_breakdown.csv", index=False)
        base_trades.groupby(["hypothesis", "environment", "direction", "exit_model"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), profit_factor=("net_return", _profit_factor)
        ).reset_index().to_csv(out_dir / "15_hypothesis_environment_summary.csv", index=False)
        base_trades.groupby(["exit_model", "tp_pct", "sl_pct", "max_hold_bars"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), profit_factor=("net_return", _profit_factor),
            tp_hit_rate=("tp_hit", "mean"), sl_hit_rate=("sl_hit", "mean"), same_bar_tp_sl_rate=("same_bar_tp_sl_flag", "mean"),
        ).reset_index().to_csv(out_dir / "16_exit_structure_summary.csv", index=False)
        base_trades.groupby(["hypothesis", "direction", "trend_bucket", "volatility_bucket"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), profit_factor=("net_return", _profit_factor)
        ).reset_index().to_csv(out_dir / "17_environment_breakdown.csv", index=False)
    else:
        for name in override_names:
            pd.DataFrame().to_csv(out_dir / name, index=False)
    cost_stress.to_csv(out_dir / "07_cost_stress.csv", index=False)
    delay_stress.to_csv(out_dir / "08_delay_stress.csv", index=False)
    baseline_summary.to_csv(out_dir / "09_matched_baseline_summary.csv", index=False)
    decisions.to_csv(out_dir / "10_research_decision.csv", index=False)
    candidates.to_csv(out_dir / "10_candidate_shortlist.csv", index=False)
    rejected.to_csv(out_dir / "11_rejected_candidates.csv", index=False)
    audit_cols = [
        "event_name", "direction", "hypothesis", "signal_time", "entry_time", "expected_entry_time", "entry_not_next_open_flag",
        "forward_window_valid_flag", "lookahead_flag", "exit_time", "exit_reason", "exit_model", "max_hold_bars", "delay_bars",
        "same_bar_tp_sl_flag", "gross_return", "net_return",
    ]
    base_trades[[c for c in audit_cols if c in base_trades.columns]].head(100000).to_csv(out_dir / "12_causal_audit.csv", index=False)
    events.head(int(args.event_sample_size)).to_csv(out_dir / "13_event_sample.csv", index=False)
    (out_dir / "14_feature_columns.json").write_text(json.dumps({"feature_columns": list(feature_columns)}, ensure_ascii=False, indent=2), encoding="utf-8")
    if stage1_selection is not None and not stage1_selection.empty:
        stage1_selection.to_csv(out_dir / "18_stage1_selection.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "18_stage1_selection.csv", index=False)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title="ETH HF CVD Impulse Spike Stall Reversal 5s V1.1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------



def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    event_specs = build_specs(args)
    full_exit_specs = build_exit_specs(args)
    probe_exit_specs = full_exit_specs if bool(args.disable_two_stage) else build_stage1_exit_specs(args)
    chunks = _month_chunks(args.start_date, args.end_date)
    lookback_bars = max(int(args.cvd_norm_window), int(args.notional_window), int(args.range_window), int(args.trend_window)) + max(_parse_csv_ints(args.impulse_bars_list)) + 20
    load_lookback = pd.Timedelta(seconds=max(lookback_bars * 5, 3600))
    full_hold_max = max(e.max_hold_bars for e in full_exit_specs)
    probe_hold_max = max(e.max_hold_bars for e in probe_exit_specs)
    full_load_forward = pd.Timedelta(seconds=(full_hold_max + max(delay_bars_list) + 5) * 5)
    probe_load_forward = pd.Timedelta(seconds=(probe_hold_max + 5) * 5)

    print(f"[run] {SCRIPT_NAME} version={SCRIPT_VERSION}", flush=True)
    print(f"[run] symbol={args.symbol} primary={args.primary_timeframe} range={args.start_date}->{args.end_date} warmup={args.warmup_start_date}", flush=True)
    stage2_mode = "direct_full_grid" if bool(args.disable_two_stage) else "two_stage_streaming_probe_then_full_survivors"

    input_rows = 0
    feature_columns: list[str] = []
    stage1_selection = pd.DataFrame()

    # Streaming containers. These are compact aggregate tables, not raw trade rows.
    event_summary_parts: list[pd.DataFrame] = []
    event_sample_parts: list[pd.DataFrame] = []
    base_stat_parts: list[pd.DataFrame] = []
    base_year_parts: list[pd.DataFrame] = []
    cost_stat_parts: list[pd.DataFrame] = []
    delay_stat_parts: list[pd.DataFrame] = []
    base_delay_stat_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    regime_parts: list[pd.DataFrame] = []
    hyp_env_parts: list[pd.DataFrame] = []
    exit_struct_parts: list[pd.DataFrame] = []
    env_breakdown_parts: list[pd.DataFrame] = []
    trade_sample_parts: list[pd.DataFrame] = []
    event_count_total = 0
    replay_count_total = 0
    delay_count_total = 0

    print(
        f"[setup] event_specs={len(event_specs) * 2} full_exit_specs={len(full_exit_specs)} "
        f"probe_exit_specs={len(probe_exit_specs)} monthly_chunks={len(chunks)} mode={stage2_mode}",
        flush=True,
    )

    if bool(args.disable_two_stage):
        selected_event_names: set[str] | None = None
    else:
        # Stage 1: stream probe trades into compact aggregate stats. Never materialize millions of rows.
        print("[stage1] streaming probe event structures with small predeclared exit set", flush=True)
        probe_stat_parts: list[pd.DataFrame] = []
        with ProgressReporter("[stage1] monthly chunks", total=len(chunks), every=max(1, int(args.progress_every))) as pr:
            for ci, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                load_start = max(pd.Timestamp(args.warmup_start_date), chunk_start - load_lookback)
                load_end = chunk_end + probe_load_forward
                raw = load_trade_bars_range(args, load_start, load_end)
                input_rows += int(len(raw))
                if raw.empty:
                    pr.update(ci)
                    continue
                feat, fcols = build_features(raw, cvd_norm_window=int(args.cvd_norm_window), notional_window=int(args.notional_window), range_window=int(args.range_window), trend_window=int(args.trend_window))
                feature_columns = fcols
                events = build_events_for_chunk(feat, event_specs, chunk_start=chunk_start, chunk_end=chunk_end, cooldown_bars=int(args.cooldown_bars))
                if not events.empty:
                    event_count_total += int(len(events))
                    event_summary_parts.append(summarize_events_chunk(events))
                    append_sample(event_sample_parts, events.drop(columns=["signal_pos"], errors="ignore"), int(args.event_sample_size))
                    for es in probe_exit_specs:
                        probe = replay_positions(feat, events, exit_spec=es, delay_bars=0, round_trip_cost_pct=float(args.round_trip_cost_pct), cost_multiplier=1.0, include_detail=False)
                        if not probe.empty:
                            probe_stat_parts.append(aggregate_trade_stats(probe, _group_keys()))
                pr.update(ci)
        probe_summary = combine_trade_stats(probe_stat_parts, _group_keys(), start_date=args.start_date, end_date=args.end_date)
        probe_yearly = pd.DataFrame()  # Stage 1 does not decide final edge; yearly stability is checked in Stage 2.
        if not probe_summary.empty:
            probe_summary["positive_years"] = 0
            probe_summary["year_count"] = 0
        selected_event_names, stage1_selection = select_stage2_event_names(probe_summary, args)
        print(
            f"[stage1] events={event_count_total:,} probe_groups={len(probe_summary):,} "
            f"selected_event_names={len(selected_event_names)} / {stage1_selection['event_name'].nunique() if not stage1_selection.empty else 0}",
            flush=True,
        )
        if not selected_event_names:
            print("[stage2] skipped: no event structures survived broad stage1 triage", flush=True)
            stage2_mode = "two_stage_probe_only_no_survivors"

    if bool(args.disable_two_stage) or selected_event_names:
        print("[stage2] streaming full exit replay on selected event structures", flush=True)
        stage2_event_count = 0
        with ProgressReporter("[stage2] monthly chunks", total=len(chunks), every=max(1, int(args.progress_every))) as pr:
            for ci, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                load_start = max(pd.Timestamp(args.warmup_start_date), chunk_start - load_lookback)
                load_end = chunk_end + full_load_forward
                raw = load_trade_bars_range(args, load_start, load_end)
                input_rows += int(len(raw))
                if raw.empty:
                    pr.update(ci)
                    continue
                feat, fcols = build_features(raw, cvd_norm_window=int(args.cvd_norm_window), notional_window=int(args.notional_window), range_window=int(args.range_window), trend_window=int(args.trend_window))
                feature_columns = fcols
                events = build_events_for_chunk(feat, event_specs, chunk_start=chunk_start, chunk_end=chunk_end, cooldown_bars=int(args.cooldown_bars))
                if selected_event_names is not None and not events.empty:
                    events = events[events["event_name"].astype(str).isin(selected_event_names)].copy()
                if not events.empty:
                    stage2_event_count += int(len(events))
                    # For direct mode, event count was not collected in stage1.
                    if bool(args.disable_two_stage):
                        event_count_total += int(len(events))
                        event_summary_parts.append(summarize_events_chunk(events))
                        append_sample(event_sample_parts, events.drop(columns=["signal_pos"], errors="ignore"), int(args.event_sample_size))
                    for es in full_exit_specs:
                        base = replay_positions(feat, events, exit_spec=es, delay_bars=0, round_trip_cost_pct=float(args.round_trip_cost_pct), cost_multiplier=1.0, include_detail=True)
                        if not base.empty:
                            replay_count_total += int(len(base))
                            base_stat_parts.append(aggregate_trade_stats(base, _group_keys()))
                            base_delay_stat_parts.append(aggregate_trade_stats(base, _group_keys() + ["delay_bars"]))
                            base_year_parts.append(aggregate_year_stats(base))
                            cost_stat_parts.append(aggregate_cost_stats(base, cost_multipliers, float(args.round_trip_cost_pct)))
                            session_parts.append(aggregate_trade_stats(base, ["hypothesis", "direction", "session", "exit_model"]))
                            regime_parts.append(aggregate_trade_stats(base, ["hypothesis", "direction", "regime", "exit_model"]))
                            hyp_env_parts.append(aggregate_trade_stats(base, ["hypothesis", "environment", "direction", "exit_model"]))
                            exit_struct_parts.append(aggregate_trade_stats(base, ["exit_model", "tp_pct", "sl_pct", "max_hold_bars"]))
                            env_breakdown_parts.append(aggregate_trade_stats(base, ["hypothesis", "direction", "trend_bucket", "volatility_bucket"]))
                            append_sample(trade_sample_parts, base, int(args.trade_sample_size))
                            if replay_count_total > int(args.max_stage2_replay_rows) and not bool(args.write_full_trades):
                                # This is a warning only: because we aggregate streaming, the row count no longer threatens memory.
                                pass
                        for db in delay_bars_list:
                            if db == 0:
                                continue
                            d = replay_positions(feat, events, exit_spec=es, delay_bars=db, round_trip_cost_pct=float(args.round_trip_cost_pct), cost_multiplier=1.0, include_detail=False)
                            if not d.empty:
                                delay_count_total += int(len(d))
                                delay_stat_parts.append(aggregate_trade_stats(d, _group_keys() + ["delay_bars"]))
                pr.update(ci)
        print(f"[stage2] selected_events={stage2_event_count:,} base_replay_rows={replay_count_total:,} delay_rows={delay_count_total:,}", flush=True)

    events_sample = pd.concat(event_sample_parts, ignore_index=True) if event_sample_parts else pd.DataFrame()
    trade_sample = pd.concat(trade_sample_parts, ignore_index=True) if trade_sample_parts else pd.DataFrame()
    event_summary = combine_event_summary(event_summary_parts)

    print(f"[events] rows={event_count_total:,} event_specs={len(event_specs) * 2}", flush=True)
    print(f"[forward] base replay rows={replay_count_total:,} delay rows={delay_count_total:,}", flush=True)

    print("[aggregate] summaries", flush=True)
    yearly_breakdown = combine_year_stats(base_year_parts)
    summary = combine_trade_stats(base_stat_parts, _group_keys(), start_date=args.start_date, end_date=args.end_date)
    summary = attach_positive_years(summary, yearly_breakdown)
    cost_stress = finalize_cost_stats(cost_stat_parts, start_date=args.start_date, end_date=args.end_date)
    delay_stress = finalize_delay_stats(delay_stat_parts + base_delay_stat_parts, start_date=args.start_date, end_date=args.end_date)
    summary_for_prefilter = _merge_fee_delay(summary, cost_stress, delay_stress)
    prelim = _baseline_candidate_groups(summary_for_prefilter, args)
    print("[aggregate] matched baseline prep", flush=True)
    # Baseline is deliberately skipped in streaming fix2 unless there are very few prelim rows and full trade rows were retained.
    # This avoids reintroducing large materialized DataFrames. Candidate rows still require manual follow-up replay if any survive.
    if prelim.empty:
        baseline_summary = build_baseline_summary(args, prelim, trade_sample, set(), full_exit_specs)
    else:
        print("[aggregate] matched baseline skipped in streaming mode: prelim survivors require focused follow-up replay", flush=True)
        baseline_summary = pd.DataFrame()

    report_table_overrides = {
        "04_yearly_breakdown.csv": yearly_breakdown[[c for c in ["hypothesis", "event_name", "direction", "exit_variant", "year", "count", "mean_net", "win_rate"] if c in yearly_breakdown.columns]] if not yearly_breakdown.empty else pd.DataFrame(),
        "05_session_breakdown.csv": finalize_breakdown_stats(session_parts, ["hypothesis", "direction", "session", "exit_model"], start_date=args.start_date, end_date=args.end_date),
        "06_regime_breakdown.csv": finalize_breakdown_stats(regime_parts, ["hypothesis", "direction", "regime", "exit_model"], start_date=args.start_date, end_date=args.end_date),
        "15_hypothesis_environment_summary.csv": finalize_breakdown_stats(hyp_env_parts, ["hypothesis", "environment", "direction", "exit_model"], start_date=args.start_date, end_date=args.end_date),
        "16_exit_structure_summary.csv": finalize_breakdown_stats(exit_struct_parts, ["exit_model", "tp_pct", "sl_pct", "max_hold_bars"], start_date=args.start_date, end_date=args.end_date),
        "17_environment_breakdown.csv": finalize_breakdown_stats(env_breakdown_parts, ["hypothesis", "direction", "trend_bucket", "volatility_bucket"], start_date=args.start_date, end_date=args.end_date),
    }

    print("[write] report files", flush=True)
    write_reports(
        args,
        specs=event_specs,
        exit_specs=full_exit_specs,
        feature_columns=feature_columns,
        input_rows=input_rows,
        events=events_sample,
        base_trades=trade_sample,
        summary=summary,
        cost_stress=cost_stress,
        delay_stress=delay_stress,
        baseline_summary=baseline_summary,
        stage1_selection=stage1_selection,
        stage2_mode=stage2_mode,
        event_summary_override=event_summary,
        event_count_override=event_count_total,
        replay_trade_count_override=replay_count_total,
        report_table_overrides=report_table_overrides,
    )
    print(f"[done] report_dir={Path(args.out_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
