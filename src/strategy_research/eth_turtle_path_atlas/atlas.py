from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurtleEpisode:
    episode_id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl: float
    max_units: int
    adds: int
    n_value: float
    entry_equity: float


def _strict_context_n(context: pd.DataFrame, entry_time: pd.Timestamp) -> float:
    ctx = context.sort_values("available_time")
    times = pd.DatetimeIndex(pd.to_datetime(ctx["available_time"]))
    pos = int(times.searchsorted(pd.Timestamp(entry_time), side="left") - 1)
    if pos < 0:
        return float("nan")
    return float(ctx.iloc[pos]["N"])


def parse_episodes(events: pd.DataFrame, equity: pd.Series, context: pd.DataFrame) -> list[TurtleEpisode]:
    if events.empty:
        return []
    ev = events.copy()
    ev["time"] = pd.to_datetime(ev["time"])
    ev = ev.sort_values("time", kind="mergesort").reset_index(drop=True)
    out: list[TurtleEpisode] = []
    current: list[pd.Series] = []
    eid = 0
    for _, row in ev.iterrows():
        kind = str(row["event"])
        if kind == "ENTRY":
            current = [row]
        elif current:
            current.append(row)
        if kind == "EXIT" and current:
            entry = current[0]
            exit_row = current[-1]
            adds = [r for r in current if str(r["event"]) == "ADD"]
            entry_time = pd.Timestamp(entry["time"])
            eq_idx = pd.DatetimeIndex(equity.index)
            p = int(eq_idx.searchsorted(entry_time, side="left"))
            entry_equity = float(equity.iloc[max(p - 1, 0)])
            eid += 1
            out.append(
                TurtleEpisode(
                    episode_id=eid,
                    entry_time=entry_time,
                    exit_time=pd.Timestamp(exit_row["time"]),
                    side=int(entry["side"]),
                    entry_price=float(entry["price"]),
                    exit_price=float(exit_row["price"]),
                    exit_reason=str(exit_row["reason"]),
                    pnl=float(exit_row["pnl"]),
                    max_units=int(max([int(r["units"]) for r in current])),
                    adds=len(adds),
                    n_value=_strict_context_n(context, entry_time),
                    entry_equity=entry_equity,
                )
            )
            current = []
    return out


def _directional_path(bars: pd.DataFrame, side: int, entry: float, n_value: float) -> pd.DataFrame:
    out = pd.DataFrame(index=bars.index)
    if side == 1:
        fav_price = bars["high"].to_numpy(float)
        adv_price = bars["low"].to_numpy(float)
        close_move = bars["close"].to_numpy(float) - entry
        fav_move = fav_price - entry
        adv_move = entry - adv_price
    else:
        fav_price = bars["low"].to_numpy(float)
        adv_price = bars["high"].to_numpy(float)
        close_move = entry - bars["close"].to_numpy(float)
        fav_move = entry - fav_price
        adv_move = adv_price - entry
    denom = n_value if np.isfinite(n_value) and n_value > 0 else np.nan
    out["close_n"] = close_move / denom
    out["fav_n"] = fav_move / denom
    out["adv_n"] = adv_move / denom
    out["running_mfe_n"] = np.maximum.accumulate(np.nan_to_num(out["fav_n"].to_numpy(float), nan=-np.inf))
    out["running_mae_n"] = np.maximum.accumulate(np.nan_to_num(out["adv_n"].to_numpy(float), nan=-np.inf))
    out["running_mfe_n"] = out["running_mfe_n"].replace(-np.inf, np.nan)
    out["running_mae_n"] = out["running_mae_n"].replace(-np.inf, np.nan)
    return out


def build_path_tables(
    one_minute: pd.DataFrame,
    events: pd.DataFrame,
    equity: pd.Series,
    context: pd.DataFrame,
    *,
    discovery_end: str,
    checkpoints_minutes: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = parse_episodes(events, equity, context)
    event_times = events.copy()
    if not event_times.empty:
        event_times["time"] = pd.to_datetime(event_times["time"])
    summary_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    add_rows: list[dict] = []
    split_boundary = pd.Timestamp(discovery_end)

    for ep in episodes:
        bars = one_minute.loc[ep.entry_time:ep.exit_time, ["open", "high", "low", "close"]].copy()
        if bars.empty:
            continue
        path = _directional_path(bars, ep.side, ep.entry_price, ep.n_value)
        mfe_n = float(path["running_mfe_n"].max())
        mae_n = float(path["running_mae_n"].max())
        mfe_ts = path["fav_n"].idxmax() if path["fav_n"].notna().any() else pd.NaT
        mae_ts = path["adv_n"].idxmax() if path["adv_n"].notna().any() else pd.NaT
        final_move_n = float(ep.side * (ep.exit_price - ep.entry_price) / ep.n_value) if ep.n_value > 0 else float("nan")
        pnl_pct_equity = 100.0 * ep.pnl / ep.entry_equity if ep.entry_equity else float("nan")
        ep_events = event_times[(event_times["time"] >= ep.entry_time) & (event_times["time"] <= ep.exit_time)]
        adds = ep_events[ep_events["event"] == "ADD"].copy()
        add_times = list(pd.to_datetime(adds["time"]))
        split = "DISCOVERY_2023_2024" if ep.entry_time <= split_boundary else "VALIDATION_2025"
        if ep.pnl > 0 and ep.exit_reason == "20D_EXIT":
            path_class = "TREND_CAPTURE"
        elif ep.pnl <= 0 and ep.max_units == 1:
            path_class = "NO_FOLLOW_THROUGH"
        elif ep.pnl <= 0 and ep.max_units >= 3:
            path_class = "PYRAMID_THEN_FAIL"
        elif ep.pnl <= 0:
            path_class = "PARTIAL_PROOF_THEN_FAIL"
        else:
            path_class = "OTHER_WIN"
        summary_rows.append(
            {
                "episode_id": ep.episode_id,
                "split": split,
                "entry_time": ep.entry_time,
                "exit_time": ep.exit_time,
                "side": "LONG" if ep.side == 1 else "SHORT",
                "entry_price": ep.entry_price,
                "exit_price": ep.exit_price,
                "exit_reason": ep.exit_reason,
                "pnl": ep.pnl,
                "pnl_pct_entry_equity": pnl_pct_equity,
                "duration_hours": (ep.exit_time - ep.entry_time).total_seconds() / 3600.0,
                "n_value": ep.n_value,
                "max_units": ep.max_units,
                "adds": ep.adds,
                "mfe_n": mfe_n,
                "mae_n": mae_n,
                "final_move_n": final_move_n,
                "giveback_from_mfe_n": mfe_n - final_move_n,
                "time_to_mfe_hours": (pd.Timestamp(mfe_ts) - ep.entry_time).total_seconds() / 3600.0 if pd.notna(mfe_ts) else np.nan,
                "time_to_mae_hours": (pd.Timestamp(mae_ts) - ep.entry_time).total_seconds() / 3600.0 if pd.notna(mae_ts) else np.nan,
                "time_to_unit2_hours": (add_times[0] - ep.entry_time).total_seconds() / 3600.0 if len(add_times) >= 1 else np.nan,
                "time_to_unit3_hours": (add_times[1] - ep.entry_time).total_seconds() / 3600.0 if len(add_times) >= 2 else np.nan,
                "time_to_unit4_hours": (add_times[2] - ep.entry_time).total_seconds() / 3600.0 if len(add_times) >= 3 else np.nan,
                "path_class": path_class,
            }
        )

        for minute in checkpoints_minutes:
            target = ep.entry_time + pd.Timedelta(minutes=int(minute))
            if target > ep.exit_time:
                continue
            p = int(pd.DatetimeIndex(path.index).searchsorted(target, side="right") - 1)
            if p < 0:
                continue
            checkpoint_rows.append(
                {
                    "episode_id": ep.episode_id,
                    "split": split,
                    "checkpoint_min": int(minute),
                    "checkpoint_time": path.index[p],
                    "close_n": float(path.iloc[p]["close_n"]),
                    "running_mfe_n": float(path.iloc[p]["running_mfe_n"]),
                    "running_mae_n": float(path.iloc[p]["running_mae_n"]),
                    "eventual_pnl": ep.pnl,
                    "eventual_win": bool(ep.pnl > 0),
                    "eventual_max_units": ep.max_units,
                    "eventual_exit_reason": ep.exit_reason,
                }
            )

        for stage_no, (_, add) in enumerate(adds.iterrows(), start=2):
            at = pd.Timestamp(add["time"])
            p = int(pd.DatetimeIndex(path.index).searchsorted(at, side="right") - 1)
            if p >= 0:
                add_rows.append(
                    {
                        "episode_id": ep.episode_id,
                        "split": split,
                        "unit_reached": stage_no,
                        "add_time": at,
                        "hours_from_entry": (at - ep.entry_time).total_seconds() / 3600.0,
                        "running_mfe_n_at_add": float(path.iloc[p]["running_mfe_n"]),
                        "running_mae_n_at_add": float(path.iloc[p]["running_mae_n"]),
                        "eventual_pnl": ep.pnl,
                        "eventual_win": bool(ep.pnl > 0),
                        "eventual_exit_reason": ep.exit_reason,
                    }
                )

    return pd.DataFrame(summary_rows), pd.DataFrame(checkpoint_rows), pd.DataFrame(add_rows)


def grouped_episode_stats(episodes: pd.DataFrame) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    group_cols = [
        ("ALL", pd.Series(True, index=episodes.index)),
        ("DISCOVERY_2023_2024", episodes["split"].eq("DISCOVERY_2023_2024")),
        ("VALIDATION_2025", episodes["split"].eq("VALIDATION_2025")),
        ("LONG", episodes["side"].eq("LONG")),
        ("SHORT", episodes["side"].eq("SHORT")),
        ("MAX_UNIT_1", episodes["max_units"].eq(1)),
        ("MAX_UNIT_2", episodes["max_units"].eq(2)),
        ("MAX_UNIT_3", episodes["max_units"].eq(3)),
        ("MAX_UNIT_4", episodes["max_units"].eq(4)),
    ]
    for label, mask in group_cols:
        g = episodes.loc[mask]
        if g.empty:
            continue
        rows.append(
            {
                "group": label,
                "episodes": int(len(g)),
                "win_rate_pct": 100.0 * float((g["pnl"] > 0).mean()),
                "sum_pnl": float(g["pnl"].sum()),
                "mean_pnl_pct_entry_equity": float(g["pnl_pct_entry_equity"].mean()),
                "median_mfe_n": float(g["mfe_n"].median()),
                "median_mae_n": float(g["mae_n"].median()),
                "median_giveback_n": float(g["giveback_from_mfe_n"].median()),
                "median_duration_hours": float(g["duration_hours"].median()),
            }
        )
    return pd.DataFrame(rows)


def checkpoint_outcome_stats(checkpoints: pd.DataFrame) -> pd.DataFrame:
    if checkpoints.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (split, minute), g in checkpoints.groupby(["split", "checkpoint_min"], sort=True):
        winners = g[g["eventual_win"]]
        losers = g[~g["eventual_win"]]
        rows.append(
            {
                "split": split,
                "checkpoint_min": int(minute),
                "episodes_alive": int(len(g)),
                "eventual_win_rate_pct": 100.0 * float(g["eventual_win"].mean()),
                "winner_median_close_n": float(winners["close_n"].median()) if not winners.empty else np.nan,
                "loser_median_close_n": float(losers["close_n"].median()) if not losers.empty else np.nan,
                "winner_median_mae_n": float(winners["running_mae_n"].median()) if not winners.empty else np.nan,
                "loser_median_mae_n": float(losers["running_mae_n"].median()) if not losers.empty else np.nan,
                "winner_median_mfe_n": float(winners["running_mfe_n"].median()) if not winners.empty else np.nan,
                "loser_median_mfe_n": float(losers["running_mfe_n"].median()) if not losers.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)
