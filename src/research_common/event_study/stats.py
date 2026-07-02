#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Statistical summaries for event-study outputs."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def profit_factor(returns: pd.Series) -> float:
    """Return gross-profit / gross-loss for signed returns."""
    x = pd.to_numeric(returns, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x <= 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def payoff_ratio(returns: pd.Series) -> float:
    """Return avg winner / avg loser abs value for signed returns."""
    x = pd.to_numeric(returns, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    wins = x[x > 0]
    losses = x[x <= 0]
    if wins.empty or losses.empty:
        return float("nan")
    avg_loss = abs(float(losses.mean()))
    return float(wins.mean()) / avg_loss if avg_loss > 0 else float("nan")


def top_winner_dependency(returns: pd.Series, *, top_n: int = 5) -> float:
    """Return the share of total positive return contributed by top winners."""
    x = pd.to_numeric(returns, errors="coerce").dropna()
    winners = x[x > 0].sort_values(ascending=False)
    gross_profit = float(winners.sum())
    if gross_profit <= 0 or winners.empty:
        return float("nan")
    return float(winners.head(int(top_n)).sum()) / gross_profit


def summarize_returns(returns: pd.Series, *, name: str | None = None, min_count: int = 0) -> dict[str, object]:
    """Summarize a signed-return series with robust event-study metrics."""
    x = pd.to_numeric(returns, errors="coerce").dropna()
    count = int(len(x))
    if count == 0:
        return {
            "metric": name or getattr(returns, "name", "return"),
            "count": 0,
            "eligible": False,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "payoff_ratio": np.nan,
            "top5_winner_share": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    pf = profit_factor(x)
    return {
        "metric": name or getattr(returns, "name", "return"),
        "count": count,
        "eligible": bool(count >= int(min_count)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": pf if math.isfinite(pf) else "inf",
        "payoff_ratio": payoff_ratio(x),
        "top5_winner_share": top_winner_dependency(x, top_n=5),
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
    }


def summarize_many(df: pd.DataFrame, return_cols: Iterable[str], *, group_cols: Iterable[str] = (), min_count: int = 0) -> pd.DataFrame:
    """Summarize one or more return columns, optionally by groups."""
    groups = list(group_cols)
    rows: list[dict[str, object]] = []
    if groups:
        grouped = df.groupby(groups, dropna=False, observed=False)
        for key, part in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            group_values = dict(zip(groups, key_tuple, strict=False))
            for col in return_cols:
                if col not in part.columns:
                    continue
                row = summarize_returns(part[col], name=col, min_count=min_count)
                row.update(group_values)
                rows.append(row)
    else:
        for col in return_cols:
            if col not in df.columns:
                continue
            rows.append(summarize_returns(df[col], name=col, min_count=min_count))
    return pd.DataFrame(rows)


def condition_contrast(df: pd.DataFrame, *, condition_col: str, return_col: str, min_count: int = 0) -> pd.DataFrame:
    """Compare return quality when a boolean condition is true vs false."""
    if condition_col not in df.columns:
        raise KeyError(f"condition column not found: {condition_col}")
    if return_col not in df.columns:
        raise KeyError(f"return column not found: {return_col}")
    cond = df[condition_col].astype("boolean").fillna(False).astype(bool)
    rows = []
    for label, mask in (("true", cond), ("false", ~cond)):
        row = summarize_returns(df.loc[mask, return_col], name=return_col, min_count=min_count)
        row["condition"] = condition_col
        row["condition_value"] = label
        rows.append(row)
    return pd.DataFrame(rows)
