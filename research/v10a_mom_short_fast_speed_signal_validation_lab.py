#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Momentum Short Fast-Speed Signal Validation Lab
====================================================

Research-only validation for the V10A candidate:

    Current V10:
        Block independent MOMENTUM_V3 LONG entries when the same-side range-footprint
        micro action is NOT_ALIGNED_RISK_REDUCED.

    V10A candidate:
        Current V10 + block independent MOMENTUM_V3 SHORT entries when range speed is FAST_Q4.

Purpose
-------
Portfolio trade count is small, so this script validates the V10A fast-speed
hypothesis at the raw-signal level:

1. All raw MOMENTUM_V3 SHORT signals, grouped by past-only range-speed regime.
2. Yearly signal stats, to see whether FAST_Q4 is broadly weaker or only one year.
3. Cooldown-deduped signal stats, to reduce repeated counting in the same trend leg.
4. Whether FAST signals are selected / executed by V10 versus V10A.
5. When V10A blocks an actual V10 Momentum Short trade, whether Bear catches up later.

Lookahead safety
----------------
- Input features for the rule use only the completed signal bar and shifted rolling
  past-only thresholds via `add_past_only_micro_features`.
- Future returns / MFE / MAE and future Bear catch-up are analysis labels only;
  they are not used to compute the rule.
- This script is validation, not a final strategy file.

Outputs
-------
- v10a_mom_short_signal_speed_overview.csv
- v10a_mom_short_signal_speed_yearly.csv
- v10a_mom_short_signal_speed_by_horizon.csv
- v10a_mom_short_signal_speed_deduped_overview.csv
- v10a_mom_short_signal_selection_execution_overlap.csv
- v10a_mom_short_fast_bear_followup_signal_audit.csv
- v10a_blocked_fast_mom_short_trade_diff.csv
- v10a_signal_validation_meta.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research import v9e_engine_router_variants_lab as router  # noqa: E402
from research import v10_microstructure_feature_discovery_lab as discovery  # noqa: E402
from research import v10_microstructure_router_variants_lab as micro_router  # noqa: E402

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
BASELINE_V10 = "baseline_v10_micro_filter"
CANDIDATE_V10A = "v10_plus_mom_short_fast_speed_block"


def _parse_horizons(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n > 0:
            out.append(n)
    return sorted(set(out)) or [1, 3, 6, 12]


def ensure_args(args: Any) -> Any:
    args = micro_router.ensure_discovery_defaults(args)
    if not hasattr(args, "horizons"):
        setattr(args, "horizons", "1,3,6,12")
    if not hasattr(args, "primary_horizon"):
        setattr(args, "primary_horizon", 12)
    if not hasattr(args, "dedupe_bars"):
        setattr(args, "dedupe_bars", 3)
    if not hasattr(args, "bear_followup_bars"):
        setattr(args, "bear_followup_bars", 3)
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v10a_mom_short_fast_speed_signal_validation_lab"
    return args


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _pf(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _safe_pct(num: float, den: float) -> float:
    try:
        if den == 0 or not math.isfinite(float(den)):
            return float("nan")
        return float(num) / float(den) * 100.0
    except Exception:
        return float("nan")


def _stats(df: pd.DataFrame, group_cols: list[str], horizon: int) -> pd.DataFrame:
    ret_col = f"ret_{horizon}bar_pct"
    win_col = f"win_{horizon}bar"
    mfe_col = f"mfe_{horizon}bar_pct"
    mae_col = f"mae_{horizon}bar_pct"
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_arg: Any = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, g in df.groupby(group_arg, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = {col: key for col, key in zip(group_cols, keys)}
        ret = pd.to_numeric(g.get(ret_col), errors="coerce")
        win = g.get(win_col)
        item.update({
            "horizon_bars": horizon,
            "count": int(len(g)),
            "win_rate_pct": float(pd.Series(win).astype("float64").mean() * 100.0) if win is not None and len(g) else np.nan,
            "avg_ret_pct": float(ret.mean()) if ret.notna().any() else np.nan,
            "median_ret_pct": float(ret.median()) if ret.notna().any() else np.nan,
            "profit_factor": _pf(ret),
            "avg_mfe_pct": float(pd.to_numeric(g.get(mfe_col), errors="coerce").mean()) if mfe_col in g else np.nan,
            "avg_mae_pct": float(pd.to_numeric(g.get(mae_col), errors="coerce").mean()) if mae_col in g else np.nan,
            "best_ret_pct": float(ret.max()) if ret.notna().any() else np.nan,
            "worst_ret_pct": float(ret.min()) if ret.notna().any() else np.nan,
        })
        rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = [c for c in group_cols if c in out.columns]
        out = out.sort_values(sort_cols + ["horizon_bars"]).reset_index(drop=True)
    return out


def _multi_horizon_stats(df: pd.DataFrame, group_cols: list[str], horizons: Iterable[int]) -> pd.DataFrame:
    frames = [_stats(df, group_cols, h) for h in horizons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dedupe_by_cooldown(events: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.sort_values(["bar_i", "timestamp"]).copy()
    keep: list[bool] = []
    last_keep = -10**12
    for _, row in out.iterrows():
        bi = int(row.get("bar_i", -10**9))
        if bi - last_keep > int(cooldown_bars):
            keep.append(True)
            last_keep = bi
        else:
            keep.append(False)
    out["dedupe_keep"] = keep
    return out.loc[out["dedupe_keep"]].copy()


def _trade_signal_ts(entry_time: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(entry_time, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts - pd.Timedelta(hours=4)


def _trade_key_frame(trades_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["signal_ts", f"executed_by_{scenario}"])
    t = trades_df.copy()
    t["signal_ts"] = t["entry_time"].apply(_trade_signal_ts)
    t["trade_side"] = t.get("type", "").astype(str).str.upper()
    t["trade_engine"] = t.get("engine", "").astype(str)
    t = t[t["signal_ts"].notna()].copy()
    cols = ["signal_ts", "entry_time", "exit_time", "trade_side", "trade_engine", "return_pct", "pnl", "capital", "note"]
    cols = [c for c in cols if c in t.columns]
    t = t[cols].copy()
    t[f"executed_by_{scenario}"] = True
    return t


def _annotate_selection(events: pd.DataFrame, features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = events.copy()
    selected_engine = features.get("selected_engine", pd.Series("NONE", index=features.index)).astype(str)
    sig = pd.to_numeric(features.get("signal", pd.Series(0, index=features.index)), errors="coerce").fillna(0).astype(int)
    s = pd.DataFrame({
        "timestamp": features.index,
        f"selected_engine_{prefix}": selected_engine.to_numpy(),
        f"selected_signal_{prefix}": sig.to_numpy(),
    })
    out = out.merge(s, on="timestamp", how="left")
    out[f"selected_by_{prefix}"] = out[f"selected_engine_{prefix}"].eq(ENGINE_MOM) & out[f"selected_signal_{prefix}"].eq(-1)
    return out


def _annotate_trades(events: pd.DataFrame, trades_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = events.copy()
    t = _trade_key_frame(trades_df, scenario)
    if t.empty:
        out[f"executed_by_{scenario}"] = False
        return out
    t = t[(t["trade_engine"].eq(ENGINE_MOM)) & (t["trade_side"].eq("SHORT"))].copy()
    suffix = f"_{scenario}"
    keep_cols = ["signal_ts", f"executed_by_{scenario}"]
    rename: dict[str, str] = {}
    for c in ["entry_time", "exit_time", "return_pct", "pnl", "capital", "note"]:
        if c in t.columns:
            rename[c] = c + suffix
            keep_cols.append(c)
    t = t[keep_cols].rename(columns={"signal_ts": "timestamp", **rename})
    out = out.merge(t, on="timestamp", how="left")
    out[f"executed_by_{scenario}"] = out[f"executed_by_{scenario}"].fillna(False).astype(bool)
    return out


def _bear_followup_flags(events: pd.DataFrame, raw: dict[str, pd.DataFrame], baseline_idx: pd.Index, lookahead_bars: int) -> pd.DataFrame:
    out = events.copy()
    bear_sig = pd.to_numeric(raw[router.ENGINE_BEAR].reindex(baseline_idx).get("signal", pd.Series(0, index=baseline_idx)), errors="coerce").fillna(0).astype(int)
    bear_short = bear_sig.eq(-1)
    ts_to_i = {pd.Timestamp(ts): i for i, ts in enumerate(baseline_idx)}
    same: list[bool] = []
    n1: list[bool] = []
    n2: list[bool] = []
    n3: list[bool] = []
    first_delta: list[float] = []
    for ts in out["timestamp"]:
        i = ts_to_i.get(pd.Timestamp(ts), None)
        if i is None:
            same.append(False); n1.append(False); n2.append(False); n3.append(False); first_delta.append(np.nan); continue
        flags = []
        first = np.nan
        for k in range(0, max(lookahead_bars, 3) + 1):
            j = i + k
            val = bool(j < len(bear_short) and bear_short.iloc[j])
            flags.append(val)
            if val and math.isnan(first):
                first = float(k)
        same.append(flags[0] if len(flags) > 0 else False)
        n1.append(any(flags[:2]))
        n2.append(any(flags[:3]))
        n3.append(any(flags[:4]))
        first_delta.append(first)
    out["bear_short_same_bar_signal"] = same
    out["bear_short_within_1bar_signal"] = n1
    out["bear_short_within_2bar_signal"] = n2
    out["bear_short_within_3bar_signal"] = n3
    out["bear_short_first_delta_bars"] = first_delta
    return out


def _build_blocked_trade_diff(v10_trades: pd.DataFrame, v10a_trades: pd.DataFrame, mom_short_events: pd.DataFrame, catchup_bars: int) -> pd.DataFrame:
    if v10_trades.empty:
        return pd.DataFrame()
    vt = v10_trades.copy()
    vt["signal_ts"] = vt["entry_time"].apply(_trade_signal_ts)
    vt = vt[(vt.get("engine", "").astype(str).eq(ENGINE_MOM)) & (vt.get("type", "").astype(str).str.upper().eq("SHORT"))].copy()
    fast_ts = set(pd.to_datetime(mom_short_events.loc[mom_short_events["is_fast_speed"], "timestamp"]))
    vt = vt[vt["signal_ts"].isin(fast_ts)].copy()
    if vt.empty:
        return pd.DataFrame()
    va = v10a_trades.copy() if v10a_trades is not None else pd.DataFrame()
    if va.empty:
        vt["v10a_followup_found"] = False
        return vt
    va["entry_time_ts"] = pd.to_datetime(va["entry_time"], errors="coerce")
    rows = []
    for _, tr in vt.iterrows():
        entry = pd.to_datetime(tr.get("entry_time"), errors="coerce")
        end = entry + pd.Timedelta(hours=4 * int(catchup_bars))
        cand = va[(va["entry_time_ts"].ge(entry)) & (va["entry_time_ts"].le(end))].copy()
        # Prefer same-side short, then Bear, then any.
        short = cand[cand.get("type", "").astype(str).str.upper().eq("SHORT")].copy() if not cand.empty else cand
        bear = short[short.get("engine", "").astype(str).eq(ENGINE_BEAR)].copy() if not short.empty else short
        pick = bear.iloc[0] if not bear.empty else (short.iloc[0] if not short.empty else (cand.iloc[0] if not cand.empty else None))
        item = {f"v10_{c}": tr.get(c, np.nan) for c in vt.columns if c != "entry_time_ts"}
        item["v10a_followup_found"] = pick is not None
        if pick is not None:
            for c in ["entry_time", "exit_time", "type", "engine", "return_pct", "pnl", "capital", "note"]:
                item[f"v10a_followup_{c}"] = pick.get(c, np.nan)
            item["followup_delay_hours"] = (pd.to_datetime(pick.get("entry_time")) - entry).total_seconds() / 3600.0
            try:
                item["delta_return_pct_followup_minus_v10"] = float(pick.get("return_pct", np.nan)) - float(tr.get("return_pct", np.nan))
            except Exception:
                item["delta_return_pct_followup_minus_v10"] = np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def main() -> int:
    args = router.parse_args()
    args = ensure_args(args)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = _parse_horizons(getattr(args, "horizons", "1,3,6,12"))
    primary_h = int(getattr(args, "primary_horizon", 12))
    if primary_h not in horizons:
        horizons = sorted(set(horizons + [primary_h]))

    print("Building V10A signal validation dataset...", flush=True)
    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    baseline_idx = baseline.index

    # Raw event study: all raw signals, not just executed portfolio trades.
    events = discovery.build_signal_events(baseline, raw, args, horizons)
    if events.empty:
        raise RuntimeError("No signal events were generated.")
    ts_to_i = {pd.Timestamp(ts): i for i, ts in enumerate(baseline_idx)}
    events["bar_i"] = events["timestamp"].map(lambda x: ts_to_i.get(pd.Timestamp(x), np.nan))

    mom_short = events[(events["engine"].eq(ENGINE_MOM)) & (events["side"].eq("SHORT"))].copy()
    mom_short["speed_group"] = mom_short["rf_speed_bin"].astype(str).replace({"nan": "NA"})
    mom_short["is_fast_speed"] = mom_short["speed_group"].eq("FAST_Q4")
    mom_short["is_normal_speed"] = mom_short["speed_group"].eq("NORMAL_Q2_Q3")
    mom_short["is_slow_speed"] = mom_short["speed_group"].eq("SLOW_Q1")
    mom_short["v10a_rule_group"] = np.where(mom_short["is_fast_speed"], "BLOCKED_FAST_Q4", "KEPT_NON_FAST")

    # Add Bear same/nearby signal flags. Future nearby flags are diagnostics only.
    mom_short = _bear_followup_flags(mom_short, raw, baseline_idx, int(getattr(args, "bear_followup_bars", 3)))

    # Build V10/V10A portfolio scenarios to label selected/executed overlap.
    cond = micro_router.build_microstructure_flags(baseline, raw, args)
    scenarios = micro_router.build_scenarios(baseline, raw, args, cond)
    if BASELINE_V10 not in scenarios or CANDIDATE_V10A not in scenarios:
        raise RuntimeError("Required V10/V10A scenarios were not found.")
    v10_features, _ = scenarios[BASELINE_V10]
    v10a_features, _ = scenarios[CANDIDATE_V10A]

    mom_short = _annotate_selection(mom_short, v10_features, "v10")
    mom_short = _annotate_selection(mom_short, v10a_features, "v10a")

    print("Running V10 and V10A portfolio backtests for execution labels...", flush=True)
    _, v10_trades_df, _, _ = router.run_variant(BASELINE_V10, v10_features, cfg, engine_cfgs, args, addon_mode="off", extra={"variant_type": "baseline_v10"})
    _, v10a_trades_df, _, _ = router.run_variant(CANDIDATE_V10A, v10a_features, cfg, engine_cfgs, args, addon_mode="off", extra={"variant_type": "candidate_v10a"})
    mom_short = _annotate_trades(mom_short, v10_trades_df, "v10")
    mom_short = _annotate_trades(mom_short, v10a_trades_df, "v10a")

    # Key outputs.
    mom_short.to_csv(out_dir / "v10a_mom_short_all_signal_events.csv", index=False, encoding="utf-8-sig")

    overview = _stats(mom_short, ["v10a_rule_group"], primary_h)
    overview.to_csv(out_dir / "v10a_mom_short_signal_speed_overview.csv", index=False, encoding="utf-8-sig")

    by_horizon = _multi_horizon_stats(mom_short, ["speed_group"], horizons)
    by_horizon.to_csv(out_dir / "v10a_mom_short_signal_speed_by_horizon.csv", index=False, encoding="utf-8-sig")

    yearly = _stats(mom_short, ["year", "v10a_rule_group"], primary_h)
    yearly.to_csv(out_dir / "v10a_mom_short_signal_speed_yearly.csv", index=False, encoding="utf-8-sig")

    deduped = _dedupe_by_cooldown(mom_short, int(getattr(args, "dedupe_bars", 3)))
    deduped.to_csv(out_dir / "v10a_mom_short_deduped_signal_events.csv", index=False, encoding="utf-8-sig")
    deduped_overview = _stats(deduped, ["v10a_rule_group"], primary_h)
    deduped_overview.to_csv(out_dir / "v10a_mom_short_signal_speed_deduped_overview.csv", index=False, encoding="utf-8-sig")
    deduped_yearly = _stats(deduped, ["year", "v10a_rule_group"], primary_h)
    deduped_yearly.to_csv(out_dir / "v10a_mom_short_signal_speed_deduped_yearly.csv", index=False, encoding="utf-8-sig")

    overlap_rows: list[dict[str, Any]] = []
    for group, g in mom_short.groupby("v10a_rule_group", dropna=False):
        item = {
            "v10a_rule_group": group,
            "signal_count": int(len(g)),
            "selected_by_v10_count": int(g["selected_by_v10"].sum()) if "selected_by_v10" in g else 0,
            "selected_by_v10a_count": int(g["selected_by_v10a"].sum()) if "selected_by_v10a" in g else 0,
            "executed_by_v10_count": int(g["executed_by_v10"].sum()) if "executed_by_v10" in g else 0,
            "executed_by_v10a_count": int(g["executed_by_v10a"].sum()) if "executed_by_v10a" in g else 0,
            "bear_same_bar_count": int(g["bear_short_same_bar_signal"].sum()) if "bear_short_same_bar_signal" in g else 0,
            "bear_within_3bar_count": int(g["bear_short_within_3bar_signal"].sum()) if "bear_short_within_3bar_signal" in g else 0,
        }
        item["selected_by_v10_pct"] = _safe_pct(item["selected_by_v10_count"], item["signal_count"])
        item["executed_by_v10_pct"] = _safe_pct(item["executed_by_v10_count"], item["signal_count"])
        item["bear_same_bar_pct"] = _safe_pct(item["bear_same_bar_count"], item["signal_count"])
        item["bear_within_3bar_pct"] = _safe_pct(item["bear_within_3bar_count"], item["signal_count"])
        overlap_rows.append(item)
    overlap = pd.DataFrame(overlap_rows).sort_values("v10a_rule_group") if overlap_rows else pd.DataFrame()
    overlap.to_csv(out_dir / "v10a_mom_short_signal_selection_execution_overlap.csv", index=False, encoding="utf-8-sig")

    bear_follow = _stats(mom_short, ["v10a_rule_group", "bear_short_same_bar_signal"], primary_h)
    bear_follow.to_csv(out_dir / "v10a_mom_short_fast_bear_followup_signal_audit.csv", index=False, encoding="utf-8-sig")

    blocked_diff = _build_blocked_trade_diff(v10_trades_df, v10a_trades_df, mom_short, int(getattr(args, "bear_followup_bars", 3)))
    blocked_diff.to_csv(out_dir / "v10a_blocked_fast_mom_short_trade_diff.csv", index=False, encoding="utf-8-sig")

    # Compact decision aid, not a final promotion verdict.
    decision_rows = []
    fast = mom_short[mom_short["is_fast_speed"]]
    nonfast = mom_short[~mom_short["is_fast_speed"]]
    fast_d = deduped[deduped["is_fast_speed"]] if not deduped.empty else pd.DataFrame()
    nonfast_d = deduped[~deduped["is_fast_speed"]] if not deduped.empty else pd.DataFrame()
    ret_col = f"ret_{primary_h}bar_pct"
    decision_rows.append({
        "check": "raw_signal_fast_vs_nonfast",
        "fast_count": int(len(fast)),
        "nonfast_count": int(len(nonfast)),
        "fast_avg_ret_pct": float(pd.to_numeric(fast.get(ret_col), errors="coerce").mean()) if not fast.empty else np.nan,
        "nonfast_avg_ret_pct": float(pd.to_numeric(nonfast.get(ret_col), errors="coerce").mean()) if not nonfast.empty else np.nan,
        "fast_pf": _pf(fast.get(ret_col, pd.Series(dtype=float))) if not fast.empty else np.nan,
        "nonfast_pf": _pf(nonfast.get(ret_col, pd.Series(dtype=float))) if not nonfast.empty else np.nan,
    })
    decision_rows.append({
        "check": "deduped_signal_fast_vs_nonfast",
        "fast_count": int(len(fast_d)),
        "nonfast_count": int(len(nonfast_d)),
        "fast_avg_ret_pct": float(pd.to_numeric(fast_d.get(ret_col), errors="coerce").mean()) if not fast_d.empty else np.nan,
        "nonfast_avg_ret_pct": float(pd.to_numeric(nonfast_d.get(ret_col), errors="coerce").mean()) if not nonfast_d.empty else np.nan,
        "fast_pf": _pf(fast_d.get(ret_col, pd.Series(dtype=float))) if not fast_d.empty else np.nan,
        "nonfast_pf": _pf(nonfast_d.get(ret_col, pd.Series(dtype=float))) if not nonfast_d.empty else np.nan,
    })
    pd.DataFrame(decision_rows).to_csv(out_dir / "v10a_signal_validation_decision_aid.csv", index=False, encoding="utf-8-sig")

    if args.write_trades:
        v10_trades_df.to_csv(out_dir / "baseline_v10_micro_filter_trades.csv", index=False, encoding="utf-8-sig")
        v10a_trades_df.to_csv(out_dir / "v10_plus_mom_short_fast_speed_block_trades.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v10a_mom_short_fast_speed_signal_validation_lab.py",
        "mode": "raw_signal_event_study_plus_portfolio_selection_execution_overlap",
        "baseline": BASELINE_V10,
        "candidate": CANDIDATE_V10A,
        "primary_horizon_bars": primary_h,
        "horizons": horizons,
        "dedupe_bars": int(getattr(args, "dedupe_bars", 3)),
        "bear_followup_bars": int(getattr(args, "bear_followup_bars", 3)),
        "no_lookahead_note": "Rule features use completed signal-bar range context plus shifted rolling past-only thresholds. Future returns and Bear follow-up are labels only.",
        "range_data_dependency_note": "The V10A fast-speed rule depends on range/footprint context, specifically rf_bar_count compared to a shifted rolling past Q75 threshold.",
        "outputs": [
            "v10a_mom_short_signal_speed_overview.csv",
            "v10a_mom_short_signal_speed_yearly.csv",
            "v10a_mom_short_signal_speed_by_horizon.csv",
            "v10a_mom_short_signal_speed_deduped_overview.csv",
            "v10a_mom_short_signal_selection_execution_overlap.csv",
            "v10a_mom_short_fast_bear_followup_signal_audit.csv",
            "v10a_blocked_fast_mom_short_trade_diff.csv",
        ],
        "args": vars(args),
        "output_dir": str(out_dir),
    }
    (out_dir / "v10a_signal_validation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V10A Momentum Short Fast-Speed Signal Validation Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v10a_mom_short_signal_speed_overview.csv")
    print("  - v10a_mom_short_signal_speed_yearly.csv")
    print("  - v10a_mom_short_signal_speed_deduped_overview.csv")
    print("  - v10a_mom_short_signal_selection_execution_overlap.csv")
    print("  - v10a_blocked_fast_mom_short_trade_diff.csv")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
