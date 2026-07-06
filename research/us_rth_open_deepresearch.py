#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
US RTH Open Deep Research Miner
===============================

Second-stage discovery miner for ``research/us_rth_open_event_study.py`` outputs.
It does not load raw market data.  It only mines causal event rows already created
by the event-study script, so full-session max-up/max-down labels are never used as
entry features.

Main outputs:
- existing event-name stability
- causal atomic feature predicates
- 2/3-condition interaction rules
- yearly stability and daily de-duplicated estimates

Example:
python research/us_rth_open_deepresearch.py --input-dir data/reports/research/us_rth_open_event_study --out-dir data/reports/research/us_rth_open_deepresearch --target-horizon 60
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.research_common.progress import ProgressReporter  # type: ignore
except Exception:  # pragma: no cover
    class ProgressReporter:  # type: ignore
        def __init__(self, total: int, desc: str = "progress", every: int = 100):
            self.total = max(int(total), 1)
            self.desc = desc
            self.every = max(int(every), 1)
            self.n = 0
        def update(self, n: int = 1):
            self.n += n
            if self.n == self.total or self.n % self.every == 0:
                print(f"[{self.desc}] {self.n}/{self.total}")
        def close(self):
            pass


@dataclass(frozen=True)
class RuleMask:
    rule: str
    features_key: tuple[str, ...]
    mask: np.ndarray
    conditions: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine US RTH open event-study outputs for causal rule candidates.")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory produced by us_rth_open_event_study.py")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--target-horizon", type=int, default=60, choices=[5, 15, 30, 60, 120, 240, 390])
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--min-year-count", type=int, default=12)
    p.add_argument("--top-atomic", type=int, default=30)
    p.add_argument("--top-pairs", type=int, default=200)
    p.add_argument("--top-triples", type=int, default=0)
    p.add_argument("--max-numeric-features", type=int, default=20)
    p.add_argument("--no-triples", action="store_true")
    return p.parse_args(argv)


def profit_factor(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses <= 0:
        return math.inf if gains > 0 else 0.0
    return float(gains / losses)


def summarize_returns(df: pd.DataFrame, metric: str, min_count: int = 0) -> dict[str, float | int | bool]:
    r = pd.to_numeric(df[metric], errors="coerce").dropna()
    count = int(len(r))
    if count == 0:
        return {
            "count": 0, "eligible": False, "mean": np.nan, "median": np.nan, "win_rate": np.nan,
            "profit_factor": np.nan, "payoff_ratio": np.nan, "top5_winner_share": np.nan,
            "p05": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan,
        }
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = -losses.mean() if len(losses) else np.nan
    pf = profit_factor(r)
    top5 = np.nan
    if len(wins) and wins.sum() > 0:
        top5 = float(wins.nlargest(min(5, len(wins))).sum() / wins.sum())
    return {
        "count": count,
        "eligible": bool(count >= min_count),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": pf,
        "payoff_ratio": float(avg_win / avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss > 0 else np.nan,
        "top5_winner_share": top5,
        "p05": float(r.quantile(0.05)),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p95": float(r.quantile(0.95)),
    }


def yearly_stats(df: pd.DataFrame, metric: str, group_cols: list[str], min_year_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols + ["year"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: val for col, val in zip(group_cols + ["year"], keys)}
        base.update(summarize_returns(sub, metric, min_count=min_year_count))
        rows.append(base)
    out = pd.DataFrame(rows)
    return out


def stability_from_yearly(y: pd.DataFrame, group_cols: list[str], min_year_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if y.empty:
        return pd.DataFrame()
    valid = y[y["count"] >= min_year_count].copy()
    for keys, sub in valid.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec["tested_years"] = int(sub["year"].nunique())
        rec["positive_years"] = int((sub["mean"] > 0).sum())
        rec["yearly_positive_rate"] = float(rec["positive_years"] / rec["tested_years"]) if rec["tested_years"] else np.nan
        rec["worst_year_mean"] = float(sub["mean"].min()) if len(sub) else np.nan
        rec["best_year_mean"] = float(sub["mean"].max()) if len(sub) else np.nan
        rec["min_year_count_observed"] = int(sub["count"].min()) if len(sub) else 0
        rows.append(rec)
    return pd.DataFrame(rows)


def rank_score(row: pd.Series) -> float:
    mean = row.get("mean", np.nan)
    pf = row.get("profit_factor", np.nan)
    wr = row.get("win_rate", np.nan)
    yr = row.get("yearly_positive_rate", np.nan)
    top5 = row.get("top5_winner_share", np.nan)
    count = row.get("count", 0)
    score = 0.0
    if pd.notna(mean):
        score += float(mean) / 0.0005
    if pd.notna(pf) and np.isfinite(pf):
        score += max(0.0, float(pf) - 1.0) * 2.0
    if pd.notna(wr):
        score += (float(wr) - 0.5) * 1.5
    if pd.notna(yr):
        score += (float(yr) - 0.5) * 1.0
    if pd.notna(top5):
        score -= max(0.0, float(top5) - 0.25) * 2.0
    score += min(math.log1p(max(float(count), 0.0)) / 10.0, 0.7)
    return float(score)


def load_events(input_dir: Path, horizon: int) -> tuple[pd.DataFrame, str]:
    path = input_dir / "01_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    df = pd.read_csv(path)
    metric = f"next_open_ret_h{horizon}_net"
    if metric not in df.columns:
        raise KeyError(f"missing target metric {metric}; available next_open cols={[c for c in df.columns if c.startswith('next_open_ret_h')]}")
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df["year"] = df["entry_time"].dt.year
    elif "signal_time" in df.columns:
        df["signal_time"] = pd.to_datetime(df["signal_time"], errors="coerce")
        df["year"] = df["signal_time"].dt.year
    else:
        raise KeyError("events must contain entry_time or signal_time")
    if "session_date" not in df.columns:
        df["session_date"] = df["entry_time"].dt.date.astype(str)
    return df, metric


FORBIDDEN_FEATURE_PATTERNS = [
    "next_open_ret", "mfe", "mae", "full_rth", "daily_event_group", "max_up", "max_down",
    "abs_max_down", "rth_close", "close_time", "entry_", "exit_", "profit", "return_label",
]


def numeric_feature_columns(df: pd.DataFrame, max_features: int) -> list[str]:
    prefixes = ("preopen_", "sig_", "open_window_")
    cols: list[str] = []
    for c in df.columns:
        if not c.startswith(prefixes):
            continue
        lc = c.lower()
        if any(p in lc for p in FORBIDDEN_FEATURE_PATTERNS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nunique = df[c].replace([np.inf, -np.inf], np.nan).nunique(dropna=True)
            if nunique >= 8:
                cols.append(c)
    # Prefer orderflow and opening-window features first, then common price context.
    def priority(c: str) -> tuple[int, str]:
        score = 9
        if "tb_" in c:
            score = 0
        elif c.startswith("open_window_"):
            score = 1
        elif "volume" in c or "atr" in c or "rv" in c:
            score = 2
        elif "ret" in c or "dist_ema" in c:
            score = 3
        return (score, c)
    return sorted(cols, key=priority)[:max_features]


def make_atomic_masks(df: pd.DataFrame, numeric_cols: list[str], min_count: int) -> list[RuleMask]:
    masks: list[RuleMask] = []
    n = len(df)
    # Categorical structural conditions.
    categorical_cols = [c for c in ["side_name", "event_family", "open_window_min"] if c in df.columns]
    if "event_name" in df.columns:
        categorical_cols.append("event_name")
    for c in categorical_cols:
        vc = df[c].value_counts(dropna=False)
        for val, cnt in vc.items():
            if cnt < min_count:
                continue
            mask = (df[c] == val).to_numpy(dtype=bool)
            masks.append(RuleMask(f"{c} == {val}", (c,), mask, 1))

    quantiles = [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() < min_count:
            continue
        qs = s.quantile(quantiles).dropna().to_dict()
        thresholds: list[tuple[str, float]] = []
        for q, val in qs.items():
            if np.isfinite(val):
                thresholds.append((f"q{int(q*100):02d}", float(val)))
        if s.min(skipna=True) < 0 < s.max(skipna=True):
            thresholds.append(("zero", 0.0))
        seen: set[tuple[str, float, str]] = set()
        arr = s.to_numpy(dtype=float)
        ok = np.isfinite(arr)
        for label, th in thresholds:
            for op in [">=", "<="]:
                key = (c, round(th, 12), op)
                if key in seen:
                    continue
                seen.add(key)
                if op == ">=":
                    mask = ok & (arr >= th)
                else:
                    mask = ok & (arr <= th)
                cnt = int(mask.sum())
                if cnt < min_count or cnt > n - max(10, min_count // 3):
                    continue
                masks.append(RuleMask(f"{c} {op} {th:.8g} ({label})", (c,), mask, 1))
    return masks


def evaluate_mask(df: pd.DataFrame, metric: str, rule: str, features_key: tuple[str, ...], mask: np.ndarray, min_count: int, min_year_count: int) -> dict[str, object] | None:
    cnt = int(mask.sum())
    if cnt < min_count:
        return None
    sub = df.loc[mask]
    rec: dict[str, object] = {"rule": rule, "features_key": "|".join(features_key), "conditions": int(rule.count(" AND ") + 1)}
    rec.update(summarize_returns(sub, metric, min_count=min_count))
    y = yearly_stats(sub, metric, [], min_year_count=min_year_count)
    y_valid = y[y["count"] >= min_year_count]
    rec["tested_years"] = int(y_valid["year"].nunique()) if not y_valid.empty else 0
    rec["positive_years"] = int((y_valid["mean"] > 0).sum()) if not y_valid.empty else 0
    rec["yearly_positive_rate"] = float(rec["positive_years"] / rec["tested_years"]) if rec["tested_years"] else np.nan
    rec["worst_year_mean"] = float(y_valid["mean"].min()) if not y_valid.empty else np.nan
    rec["min_year_count_observed"] = int(y_valid["count"].min()) if not y_valid.empty else 0
    # simple deep candidate flag; still discovery only
    rec["candidate_flag"] = bool(
        rec["count"] >= min_count
        and pd.notna(rec["mean"]) and float(rec["mean"]) > 0.00035
        and pd.notna(rec["profit_factor"]) and np.isfinite(float(rec["profit_factor"])) and float(rec["profit_factor"]) >= 1.12
        and pd.notna(rec["yearly_positive_rate"]) and float(rec["yearly_positive_rate"]) >= 0.75
        and pd.notna(rec["top5_winner_share"]) and float(rec["top5_winner_share"]) <= 0.35
        and int(rec["tested_years"]) >= 3
    )
    rec["rank_score"] = rank_score(pd.Series(rec))
    return rec


def mine_atomic(df: pd.DataFrame, metric: str, masks: list[RuleMask], min_count: int, min_year_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prog = ProgressReporter(total=len(masks), desc="atomic rules", every=max(1, len(masks)//20))
    for m in masks:
        rec = evaluate_mask(df, metric, m.rule, m.features_key, m.mask, min_count, min_year_count)
        if rec is not None:
            rows.append(rec)
        prog.update(1)
    prog.close()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["candidate_flag", "rank_score", "mean"], ascending=[False, False, False])
    return out


def same_feature_family(a: RuleMask, b: RuleMask) -> bool:
    # Avoid combinations like x>=q80 AND x>=q90, and duplicated event-name/event-family pairs when too specific.
    if set(a.features_key) & set(b.features_key):
        return True
    return False


def combine_rules(df: pd.DataFrame, metric: str, base_masks: list[RuleMask], min_count: int, min_year_count: int, top_pairs: int, top_triples: int, no_triples: bool) -> tuple[pd.DataFrame, list[RuleMask]]:
    pair_rows: list[dict[str, object]] = []
    pair_masks: list[RuleMask] = []
    combos = list(itertools.combinations(base_masks, 2))
    prog = ProgressReporter(total=len(combos), desc="pair rules", every=max(1, len(combos)//20))
    for a, b in combos:
        if same_feature_family(a, b):
            prog.update(1); continue
        mask = a.mask & b.mask
        if int(mask.sum()) >= min_count:
            rule = f"{a.rule} AND {b.rule}"
            features_key = tuple(sorted(set(a.features_key + b.features_key)))
            rec = evaluate_mask(df, metric, rule, features_key, mask, min_count, min_year_count)
            if rec is not None:
                pair_rows.append(rec)
                pair_masks.append(RuleMask(rule, features_key, mask, 2))
        prog.update(1)
    prog.close()
    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(["candidate_flag", "rank_score", "mean"], ascending=[False, False, False]).head(top_pairs)
        keep_rules = set(pair_df["rule"].astype(str))
        pair_masks = [m for m in pair_masks if m.rule in keep_rules]

    if no_triples or not pair_masks or top_triples <= 0:
        return pair_df, []

    # Build triples by adding one high-quality atomic to high-quality pairs.
    triple_rows: list[dict[str, object]] = []
    triple_masks: list[RuleMask] = []
    max_pair_seed = min(len(pair_masks), max(top_triples * 3, 200))
    seeds = pair_masks[:max_pair_seed]
    combos2 = [(p, a) for p in seeds for a in base_masks]
    prog = ProgressReporter(total=len(combos2), desc="triple rules", every=max(1, len(combos2)//20))
    seen: set[tuple[str, ...]] = set()
    for p, a in combos2:
        if set(a.features_key) & set(p.features_key):
            prog.update(1); continue
        features_key = tuple(sorted(set(p.features_key + a.features_key)))
        if features_key in seen:
            prog.update(1); continue
        mask = p.mask & a.mask
        if int(mask.sum()) >= min_count:
            rule = f"{p.rule} AND {a.rule}"
            rec = evaluate_mask(df, metric, rule, features_key, mask, min_count, min_year_count)
            if rec is not None:
                triple_rows.append(rec)
                triple_masks.append(RuleMask(rule, features_key, mask, 3))
                seen.add(features_key)
        prog.update(1)
    prog.close()
    triple_df = pd.DataFrame(triple_rows)
    if not triple_df.empty:
        triple_df = triple_df.sort_values(["candidate_flag", "rank_score", "mean"], ascending=[False, False, False]).head(top_triples)
    all_df = pd.concat([pair_df, triple_df], ignore_index=True) if not pair_df.empty or not triple_df.empty else pd.DataFrame()
    if not all_df.empty:
        all_df = all_df.sort_values(["candidate_flag", "rank_score", "mean"], ascending=[False, False, False])
    return all_df, triple_masks


def existing_event_stats(df: pd.DataFrame, metric: str, min_count: int, min_year_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "event_family", "open_window_min", "side_name"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec.update(summarize_returns(sub, metric, min_count=min_count))
        rows.append(rec)
    stats = pd.DataFrame(rows)
    y = yearly_stats(df, metric, group_cols, min_year_count)
    stab = stability_from_yearly(y, group_cols, min_year_count)
    out = stats.merge(stab, on=group_cols, how="left")
    out["tested_years"] = out["tested_years"].fillna(0).astype(int)
    out["positive_years"] = out["positive_years"].fillna(0).astype(int)
    out["candidate_flag"] = (
        (out["count"] >= min_count)
        & (out["mean"] > 0.00035)
        & (out["profit_factor"] >= 1.12)
        & (out["yearly_positive_rate"] >= 0.75)
        & (out["top5_winner_share"] <= 0.35)
        & (out["tested_years"] >= 3)
    )
    out["rank_score"] = out.apply(rank_score, axis=1)
    out = out.sort_values(["candidate_flag", "rank_score", "mean"], ascending=[False, False, False])
    return out, y


def rule_to_mask(df: pd.DataFrame, rule: str, available_masks: dict[str, np.ndarray]) -> np.ndarray | None:
    return available_masks.get(rule)


def daily_dedup_for_rule(df: pd.DataFrame, metric: str, rule_name: str, mask: np.ndarray, min_count: int) -> dict[str, object]:
    sub = df.loc[mask].copy()
    if sub.empty:
        return {"rule": rule_name, "daily_count": 0}
    sort_cols = [c for c in ["session_date", "entry_time"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols)
    one = sub.groupby("session_date", as_index=False).head(1) if "session_date" in sub.columns else sub
    rec = {"rule": rule_name, "raw_event_count": int(mask.sum()), "daily_count": int(len(one)), "unique_sessions": int(one["session_date"].nunique()) if "session_date" in one.columns else int(len(one))}
    rec.update({f"daily_{k}": v for k, v in summarize_returns(one, metric, min_count=min_count).items()})
    return rec


def write_brief(out_dir: Path, meta: dict[str, object], existing: pd.DataFrame, atomic: pd.DataFrame, combo: pd.DataFrame, daily_dedup: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# US RTH Open Deep Research Report")
    lines.append("")
    lines.append("This is still discovery-only. It mines only causal event rows created by the first event-study script.")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    def block(title: str, df: pd.DataFrame, cols: list[str], n: int = 15):
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("No rows.")
        else:
            lines.append("```text")
            lines.append(df[cols].head(n).to_string(index=False))
            lines.append("```")
        lines.append("")
    common_cols = ["rule", "count", "mean", "median", "win_rate", "profit_factor", "tested_years", "yearly_positive_rate", "top5_winner_share", "candidate_flag", "rank_score"]
    if not existing.empty:
        tmp = existing.rename(columns={"event_name": "rule"})
        block("Existing event-name candidates", tmp, [c for c in common_cols if c in tmp.columns])
    block("Atomic causal feature rules", atomic, [c for c in common_cols if c in atomic.columns])
    block("Interaction rules", combo, [c for c in common_cols if c in combo.columns])
    if not daily_dedup.empty:
        block("Daily de-duplicated estimates", daily_dedup, [c for c in ["rule", "raw_event_count", "daily_count", "daily_mean", "daily_median", "daily_win_rate", "daily_profit_factor"] if c in daily_dedup.columns])
    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append("- Promote nothing directly to live trading.")
    lines.append("- Rules with high mean but low count, high top5 winner share, or missing yearly coverage are weak candidates.")
    lines.append("- Next step should be formal replay with one position at a time, next-open audit, fee/slippage/delay stress, and parameter-neighbourhood checks.")
    (out_dir / "20_deepresearch_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, metric = load_events(args.input_dir, args.target_horizon)
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = numeric_feature_columns(df, args.max_numeric_features)

    print(f"[load] events={len(df):,} metric={metric} years={sorted(df['year'].dropna().unique().tolist())}")
    print(f"[features] numeric causal features={len(numeric_cols)}")

    existing, existing_yearly = existing_event_stats(df, metric, args.min_count, args.min_year_count)
    existing.to_csv(args.out_dir / "00_existing_event_name_stats_deep.csv", index=False)
    existing_yearly.to_csv(args.out_dir / "01_existing_event_name_yearly_deep.csv", index=False)

    atomic_masks_all = make_atomic_masks(df, numeric_cols, args.min_count)
    print(f"[rules] atomic masks={len(atomic_masks_all):,}")
    atomic = mine_atomic(df, metric, atomic_masks_all, args.min_count, args.min_year_count)
    atomic.to_csv(args.out_dir / "02_atomic_rule_stats.csv", index=False)

    # Use top-quality atomic masks for interactions. Prefer candidate or positive PF/mean.
    if not atomic.empty:
        usable_rules = set(atomic.head(args.top_atomic)["rule"].astype(str))
        atomic_lookup = {m.rule: m for m in atomic_masks_all}
        seed_masks = [atomic_lookup[r] for r in atomic[atomic["rule"].isin(usable_rules)]["rule"].astype(str).tolist() if r in atomic_lookup]
    else:
        seed_masks = atomic_masks_all[:args.top_atomic]
    combo, _ = combine_rules(df, metric, seed_masks, args.min_count, args.min_year_count, args.top_pairs, args.top_triples, args.no_triples)
    combo.to_csv(args.out_dir / "03_interaction_rule_stats.csv", index=False)

    # Daily de-dup for the best discovery rules.
    daily_rows: list[dict[str, object]] = []
    all_mask_lookup = {m.rule: m.mask for m in atomic_masks_all}
    for m in seed_masks:
        all_mask_lookup[m.rule] = m.mask
    # Reconstruct combo masks only for top rows to keep memory small.
    top_rules = []
    for table in [existing.rename(columns={"event_name": "rule"}).head(30), atomic.head(30), combo.head(40) if not combo.empty else combo]:
        if table is not None and not table.empty and "rule" in table.columns:
            top_rules.extend(table["rule"].astype(str).tolist())
    seen_rules: set[str] = set()
    for rule in top_rules:
        if rule in seen_rules:
            continue
        seen_rules.add(rule)
        if " AND " in rule:
            parts = [p.strip() for p in rule.split(" AND ")]
            masks = []
            ok = True
            for part in parts:
                if part not in all_mask_lookup:
                    ok = False; break
                masks.append(all_mask_lookup[part])
            if not ok:
                continue
            mask = masks[0].copy()
            for m in masks[1:]:
                mask &= m
        elif rule in all_mask_lookup:
            mask = all_mask_lookup[rule]
        elif "event_name" in df.columns and rule in set(df["event_name"].astype(str).unique()):
            mask = (df["event_name"].astype(str) == rule).to_numpy(dtype=bool)
        else:
            continue
        daily_rows.append(daily_dedup_for_rule(df, metric, rule, mask, max(20, args.min_count // 2)))
    daily_dedup = pd.DataFrame(daily_rows)
    if not daily_dedup.empty:
        daily_dedup = daily_dedup.sort_values(["daily_profit_factor", "daily_mean"], ascending=[False, False])
    daily_dedup.to_csv(args.out_dir / "04_daily_dedup_rule_stats.csv", index=False)

    meta = {
        "input_dir": str(args.input_dir),
        "target_metric": metric,
        "events": int(len(df)),
        "min_count": int(args.min_count),
        "min_year_count": int(args.min_year_count),
        "numeric_features": int(len(numeric_cols)),
        "atomic_masks": int(len(atomic_masks_all)),
        "atomic_rows": int(len(atomic)),
        "interaction_rows": int(len(combo)) if not combo.empty else 0,
        "discovery_warning": "Discovery only; do not promote without formal replay/stress/audit.",
    }
    (args.out_dir / "99_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_brief(args.out_dir, meta, existing, atomic, combo, daily_dedup)
    print(f"[done] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
