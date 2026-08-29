from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PortfolioSelectionKey


def _safe_pf(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _max_consecutive_losing_days(daily_equity: pd.DataFrame) -> int:
    """Maximum consecutive calendar days with negative mark-to-market return.

    Flat/no-change days break the streak. This is intentionally equity-path based
    rather than exit-day based so long-held positions cannot hide multi-day pain.
    """
    if daily_equity.empty or "equity" not in daily_equity:
        return 0
    ret = pd.to_numeric(daily_equity["equity"], errors="coerce").pct_change()
    best = cur = 0
    for value in ret.to_numpy(dtype=float):
        if np.isfinite(value) and value < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _max_flat_days(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if end <= start:
        return 0.0
    if trades.empty:
        return float((end - start).total_seconds() / 86400.0)
    x = trades.copy()
    x["entry_time"] = pd.to_datetime(x["entry_time"], errors="coerce")
    x["exit_time"] = pd.to_datetime(x["exit_time"], errors="coerce")
    x = x.dropna(subset=["entry_time", "exit_time"]).sort_values("entry_time")
    if x.empty:
        return float((end - start).total_seconds() / 86400.0)
    gaps = [max((x.iloc[0]["entry_time"] - start).total_seconds(), 0.0)]
    prev_exit = x.iloc[0]["exit_time"]
    for _, row in x.iloc[1:].iterrows():
        gaps.append(max((row["entry_time"] - prev_exit).total_seconds(), 0.0))
        prev_exit = max(prev_exit, row["exit_time"])
    gaps.append(max((end - prev_exit).total_seconds(), 0.0))
    return float(max(gaps) / 86400.0)


def _max_drawdown_pct(daily_equity: pd.DataFrame) -> float:
    if daily_equity.empty or "equity" not in daily_equity:
        return 0.0
    eq = pd.to_numeric(daily_equity["equity"], errors="coerce").dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax().replace(0.0, np.nan)
    dd = 1.0 - eq / peak
    return float(dd.max() * 100.0)


def calculate_metrics(
    trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    *,
    initial_capital: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    intrabar_max_drawdown_pct: float | None = None,
) -> dict[str, Any]:
    final_equity = float(daily_equity["equity"].iloc[-1]) if not daily_equity.empty else initial_capital
    total_ret = final_equity / initial_capital - 1.0
    years = max((end - start).total_seconds() / (365.25 * 86400.0), 1.0 / 365.25)
    cagr = (final_equity / initial_capital) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    pf = _safe_pf(pnl)
    win_rate = float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0
    mdd = _max_drawdown_pct(daily_equity)
    if intrabar_max_drawdown_pct is not None:
        mdd = max(mdd, float(intrabar_max_drawdown_pct))

    yearly = pd.DataFrame()
    monthly = pd.DataFrame()
    if not daily_equity.empty:
        de = daily_equity.copy()
        de.index = pd.to_datetime(de.index)
        rets = de["equity"].pct_change().fillna(0.0)
        yearly_ret = rets.groupby(de.index.year).apply(lambda s: (1.0 + s).prod() - 1.0)
        monthly_ret = rets.groupby(de.index.to_period("M")).apply(lambda s: (1.0 + s).prod() - 1.0)
        yearly = yearly_ret
        monthly = monthly_ret
    positive_years = int((yearly > 0).sum()) if len(yearly) else 0
    positive_month_ratio = float((monthly > 0).mean()) if len(monthly) else 0.0
    worst_month_pct = float(monthly.min() * 100.0) if len(monthly) else 0.0

    return {
        "total_trades": int(len(trades)),
        "long_trades": int((trades.get("side", pd.Series(dtype=int)) == 1).sum()) if not trades.empty else 0,
        "short_trades": int((trades.get("side", pd.Series(dtype=int)) == -1).sum()) if not trades.empty else 0,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "total_return_pct": total_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": mdd,
        "max_flat_days": _max_flat_days(trades, start, end),
        "max_consecutive_losing_days": _max_consecutive_losing_days(daily_equity),
        "positive_years": positive_years,
        "positive_month_ratio": positive_month_ratio,
        "worst_month_pct": worst_month_pct,
        "final_equity": final_equity,
    }


def yearly_table(strategy_id: str, daily_equity: pd.DataFrame) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame(columns=["strategy_id", "year", "return_pct"])
    de = daily_equity.copy()
    de.index = pd.to_datetime(de.index)
    ret = de["equity"].pct_change().fillna(0.0)
    rows = []
    for year, s in ret.groupby(de.index.year):
        rows.append({"strategy_id": strategy_id, "year": int(year), "return_pct": ((1.0 + s).prod() - 1.0) * 100.0})
    return pd.DataFrame(rows)


def monthly_table(strategy_id: str, daily_equity: pd.DataFrame) -> pd.DataFrame:
    if daily_equity.empty:
        return pd.DataFrame(columns=["strategy_id", "month", "return_pct"])
    de = daily_equity.copy()
    de.index = pd.to_datetime(de.index)
    ret = de["equity"].pct_change().fillna(0.0)
    rows = []
    for month, s in ret.groupby(de.index.to_period("M")):
        rows.append({"strategy_id": strategy_id, "month": str(month), "return_pct": ((1.0 + s).prod() - 1.0) * 100.0})
    return pd.DataFrame(rows)


def top_trade_dependency(trades: pd.DataFrame, top_ns: tuple[int, ...] = (1, 5, 10)) -> list[dict[str, Any]]:
    if trades.empty or "return_on_equity" not in trades:
        return [{"remove_top_n": n, "total_return_pct": 0.0} for n in top_ns]
    r = pd.to_numeric(trades["return_on_equity"], errors="coerce").fillna(0.0)
    order = r.sort_values(ascending=False).index
    rows: list[dict[str, Any]] = []
    for n in top_ns:
        keep = r.copy()
        keep.loc[order[: min(n, len(order))]] = 0.0
        total = float(np.prod(1.0 + keep.to_numpy(dtype=float)) - 1.0)
        rows.append({"remove_top_n": int(n), "total_return_pct": total * 100.0})
    return rows


def passes_survivor_gate(metrics: dict[str, Any], *, max_mdd_pct: float, min_pf: float, min_positive_years: int, min_trades: int) -> tuple[bool, str]:
    reasons: list[str] = []
    if float(metrics.get("total_return_pct", -1.0)) <= 0:
        reasons.append("total_return<=0")
    pf = float(metrics.get("profit_factor", 0.0))
    if not math.isfinite(pf):
        pf = 999.0
    if pf <= min_pf:
        reasons.append(f"PF<={min_pf}")
    if float(metrics.get("max_drawdown_pct", 999.0)) > max_mdd_pct:
        reasons.append(f"MDD>{max_mdd_pct}")
    if int(metrics.get("positive_years", 0)) < min_positive_years:
        reasons.append(f"positive_years<{min_positive_years}")
    if int(metrics.get("total_trades", 0)) < min_trades:
        reasons.append(f"trades<{min_trades}")
    return not reasons, ";".join(reasons) if reasons else "PASS"


def selection_key(metrics: dict[str, Any]) -> PortfolioSelectionKey:
    return PortfolioSelectionKey.from_metrics(metrics)
