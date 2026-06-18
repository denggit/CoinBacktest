#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streaming helpers for high-frequency backtests.

This module is intentionally data-layer only.  Strategy-specific signal logic and
execution state stay in the HF backtest files, while this file provides reusable
streaming primitives:

- raw trade/tick chunk iteration;
- optional per-day raw tick frames;
- per-day second-bar aggregation without keeping the whole date range in memory;
- warmup windows for rolling features;
- execution windows that preserve next-bar execution across day boundaries.

The main rule is: never accumulate the full multi-year HF dataset in memory.
At most one day plus a bounded warmup tail should be resident at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable, Iterator, Sequence

import pandas as pd

from src.backtest_common.data import aggregate_trades_to_seconds, merge_second_bars
from src.data_feed.okx_tick_loader import OKXTickLoader

DEFAULT_WARMUP_EXTRA_SECONDS = 300
DEFAULT_TIME_COLS = ("timestamp", "ts", "datetime", "time", "created_time", "ts_ms")


@dataclass(frozen=True)
class HFTradeChunk:
    """One normalized raw trade/tick chunk from OKXTickLoader."""

    day: date
    chunk: pd.DataFrame
    chunk_index: int
    rows: int


@dataclass(frozen=True)
class HFTradeDay:
    """One UTC day's raw trade/tick data.

    Use this only for strategies that really need a full day of ticks at once.
    For maximum memory safety, prefer iter_trade_chunks() or
    iter_trade_chunk_windows().
    """

    day: date
    ticks: pd.DataFrame
    chunks: int
    rows: int


@dataclass(frozen=True)
class HFBarDay:
    """One UTC day's aggregated bar data."""

    day: date
    bars: pd.DataFrame
    trade_rows: int
    bar_chunks: int


@dataclass(frozen=True)
class HFDataWindow:
    """A bounded feature/execution window for streaming HF backtests.

    current:
        Data for the current chunk/day only.
    work:
        warmup tail + current.  Build rolling features on this frame.
    execute_start/execute_end:
        Slice features to this range before passing to a stateful chunk runner.
        The execution slice includes the previous unprocessed last row when
        available, so a signal generated on day D's last bar can enter on day
        D+1's first bar without lookahead.
    """

    key: Any
    current: pd.DataFrame
    work: pd.DataFrame
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    execute_start: pd.Timestamp
    execute_end: pd.Timestamp
    warmup_rows: int
    current_rows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def slice_current(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only the current chunk/day part from a feature frame."""
        if frame.empty:
            return frame
        return frame.loc[self.current_start : self.current_end]

    def slice_execution(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows that should be passed to run_backtest_chunk().

        The returned frame is designed for next-bar execution loops that iterate
        ``for i in range(len(rows) - 1)``.  The final row is kept as the next
        price for the previous signal, but its own signal will naturally not be
        processed until the following streaming window.
        """
        if frame.empty:
            return frame
        return frame.loc[self.execute_start : self.execute_end]


@dataclass
class FrameWarmupBuffer:
    """Bounded FIFO warmup buffer for DataFrame windows.

    It works for both DatetimeIndex frames and raw tick frames with a timestamp
    column.  For rolling features, seconds-based trimming is preferred.  rows can
    be used as an additional hard cap.
    """

    seconds: int = 0
    rows: int | None = None
    time_col: str | None = None
    _frame: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def rows_count(self) -> int:
        return len(self._frame)

    def prepend(self, current: pd.DataFrame) -> pd.DataFrame:
        """Return warmup + current without mutating the buffer."""
        current = ensure_time_ordered(current, time_col=self.time_col)
        if self._frame.empty:
            return current
        out = pd.concat([self._frame, current], axis=0)
        out = ensure_time_ordered(out, time_col=self.time_col)
        return out

    def update_from_work(self, work: pd.DataFrame) -> None:
        """Keep only the bounded tail of the already-combined work frame."""
        self._frame = tail_frame(work, seconds=self.seconds, rows=self.rows, time_col=self.time_col)

    def update_from_current(self, current: pd.DataFrame) -> None:
        """Append current to existing warmup and keep the bounded tail."""
        self.update_from_work(self.prepend(current))

    def clear(self) -> None:
        self._frame = pd.DataFrame()


def iter_trade_chunks(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
) -> Iterator[HFTradeChunk]:
    """Yield raw normalized trade chunks, day by day, without aggregation."""
    loader = OKXTickLoader(symbol=symbol, data_dir=data_dir, trades_url_template=trades_url_template)
    chunk_counts: dict[date, int] = {}
    for day, chunk in loader.iter_trades(start_date, end_date, chunksize=chunksize):
        idx = chunk_counts.get(day, 0) + 1
        chunk_counts[day] = idx
        yield HFTradeChunk(day=day, chunk=chunk, chunk_index=idx, rows=len(chunk))


def iter_trade_days(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
    max_day_rows: int | None = None,
) -> Iterator[HFTradeDay]:
    """Yield one day of raw ticks at a time.

    This still avoids multi-year memory usage, but a single high-volume day can
    be large.  Set max_day_rows to guard against accidentally loading a huge day.
    """
    current_day: date | None = None
    parts: list[pd.DataFrame] = []
    chunk_count = 0
    row_count = 0

    def flush() -> HFTradeDay | None:
        nonlocal current_day, parts, chunk_count, row_count
        if current_day is None or not parts:
            return None
        ticks = pd.concat(parts, ignore_index=True)
        ticks = ensure_time_ordered(ticks, time_col="timestamp")
        out = HFTradeDay(day=current_day, ticks=ticks, chunks=chunk_count, rows=row_count)
        parts = []
        chunk_count = 0
        row_count = 0
        return out

    for item in iter_trade_chunks(
        symbol,
        start_date,
        end_date,
        chunksize=chunksize,
        trades_url_template=trades_url_template,
        data_dir=data_dir,
    ):
        if current_day is not None and item.day != current_day:
            flushed = flush()
            if flushed is not None:
                yield flushed
        current_day = item.day
        row_count += item.rows
        if max_day_rows is not None and row_count > int(max_day_rows):
            raise MemoryError(f"raw tick day {item.day} exceeded max_day_rows={max_day_rows}")
        chunk_count += 1
        parts.append(item.chunk)

    flushed = flush()
    if flushed is not None:
        yield flushed


def iter_trade_chunk_windows(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
    warmup_seconds: int = 0,
    warmup_rows: int | None = None,
    time_col: str = "timestamp",
) -> Iterator[HFDataWindow]:
    """Yield raw tick windows: warmup tail + current tick chunk.

    This is for future tick-level strategies.  It does not aggregate to bars.
    The strategy can build tick-level features on window.work, then use
    window.slice_execution(features) to continue across chunk boundaries.
    """
    buffer = FrameWarmupBuffer(seconds=max(0, int(warmup_seconds)), rows=_effective_warmup_rows(warmup_rows, min_rows=1, seconds=max(0, int(warmup_seconds))), time_col=time_col)
    pending_start: pd.Timestamp | None = None

    for item in iter_trade_chunks(
        symbol,
        start_date,
        end_date,
        chunksize=chunksize,
        trades_url_template=trades_url_template,
        data_dir=data_dir,
    ):
        current = ensure_time_ordered(item.chunk, time_col=time_col)
        if current.empty:
            continue
        work = buffer.prepend(current)
        current_start, current_end = frame_time_bounds(current, time_col=time_col)
        execute_start = pending_start or current_start
        window = HFDataWindow(
            key=(item.day, item.chunk_index),
            current=current,
            work=work,
            current_start=current_start,
            current_end=current_end,
            execute_start=execute_start,
            execute_end=current_end,
            warmup_rows=buffer.rows_count,
            current_rows=len(current),
            meta={"day": item.day, "chunk_index": item.chunk_index, "kind": "ticks", "source_rows": item.rows},
        )
        yield window
        buffer.update_from_work(work)
        pending_start = current_end


def iter_second_bars_by_day(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    cfg: Any,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
    progress: bool = False,
) -> Iterator[HFBarDay]:
    """Yield one day's aggregated second bars at a time.

    Only the current day's intermediate bar parts are accumulated.  The full
    requested date range is never kept in memory.
    """
    current_day: date | None = None
    bar_parts: list[pd.DataFrame] = []
    trade_rows = 0
    bar_chunks = 0

    def flush() -> HFBarDay | None:
        nonlocal current_day, bar_parts, trade_rows, bar_chunks
        if current_day is None or not bar_parts:
            return None
        bars = merge_second_bars(bar_parts)
        out = HFBarDay(day=current_day, bars=bars, trade_rows=trade_rows, bar_chunks=bar_chunks)
        bar_parts = []
        trade_rows = 0
        bar_chunks = 0
        return out

    for item in iter_trade_chunks(
        symbol,
        start_date,
        end_date,
        chunksize=chunksize,
        trades_url_template=trades_url_template,
        data_dir=data_dir,
    ):
        if current_day is not None and item.day != current_day:
            flushed = flush()
            if flushed is not None and not flushed.bars.empty:
                yield flushed
        current_day = item.day
        bars = aggregate_trades_to_seconds(item.chunk, cfg)
        trade_rows += item.rows
        if not bars.empty:
            bar_parts.append(bars)
            bar_chunks += 1
        if progress:
            print(f"stream tick chunk day={item.day} chunk={item.chunk_index} rows={item.rows} bars={len(bars)}")
        del bars

    flushed = flush()
    if flushed is not None and not flushed.bars.empty:
        yield flushed


def iter_second_bar_windows_by_day(
    symbol: str,
    start_date: str | date,
    end_date: str | date,
    cfg: Any,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
    warmup_seconds: int | None = None,
    warmup_rows: int | None = None,
    progress: bool = False,
) -> Iterator[HFDataWindow]:
    """Yield warmup+current windows over per-day second bars.

    Typical usage in a strategy file:

        state = init_state(cfg)
        for window in iter_second_bar_windows_by_day(...):
            features = build_features(window.work, cfg)
            exec_features = window.slice_execution(features)
            state = run_backtest_chunk(exec_features, cfg, state)

    run_backtest_chunk should use next-bar execution and loop over
    ``range(len(rows) - 1)``.  That naturally leaves the final row pending for
    the next streaming window.
    """
    warmup_sec = compute_warmup_seconds(cfg, warmup_seconds=warmup_seconds)
    buffer = FrameWarmupBuffer(seconds=warmup_sec, rows=_effective_warmup_rows(warmup_rows, min_rows=1, seconds=warmup_sec), time_col=None)
    pending_start: pd.Timestamp | None = None

    for bar_day in iter_second_bars_by_day(
        symbol,
        start_date,
        end_date,
        cfg,
        chunksize=chunksize,
        trades_url_template=trades_url_template,
        data_dir=data_dir,
        progress=progress,
    ):
        current = ensure_time_ordered(bar_day.bars)
        if current.empty:
            continue
        work = buffer.prepend(current)
        current_start, current_end = frame_time_bounds(current)
        execute_start = pending_start or current_start
        window = HFDataWindow(
            key=bar_day.day,
            current=current,
            work=work,
            current_start=current_start,
            current_end=current_end,
            execute_start=execute_start,
            execute_end=current_end,
            warmup_rows=buffer.rows_count,
            current_rows=len(current),
            meta={
                "day": bar_day.day,
                "kind": "second_bars",
                "trade_rows": bar_day.trade_rows,
                "bar_chunks": bar_day.bar_chunks,
                "warmup_seconds": warmup_sec,
            },
        )
        yield window
        buffer.update_from_work(work)
        pending_start = current_end


def load_second_bars_one_day(
    symbol: str,
    day: str | date,
    cfg: Any,
    *,
    chunksize: int = 100_000,
    trades_url_template: str = "",
    data_dir: str | None = None,
) -> pd.DataFrame:
    """Convenience helper for diagnostics: load/aggregate exactly one day."""
    start = _parse_date(day)
    end = start
    for item in iter_second_bars_by_day(
        symbol,
        start,
        end,
        cfg,
        chunksize=chunksize,
        trades_url_template=trades_url_template,
        data_dir=data_dir,
    ):
        return item.bars
    return pd.DataFrame()


def compute_warmup_seconds(
    cfg: Any,
    *,
    warmup_seconds: int | None = None,
    extra_seconds: int = DEFAULT_WARMUP_EXTRA_SECONDS,
    candidate_attrs: Sequence[str] | None = None,
) -> int:
    """Infer a safe rolling-feature warmup window from common HF config names."""
    if warmup_seconds is not None:
        return max(0, int(warmup_seconds))

    attrs = candidate_attrs or (
        "context_seconds",
        "range_seconds",
        "confirm_seconds",
        "signal_seconds",
        "local_lookback_seconds",
        "trade_window_seconds",
        "reclaim_window_seconds",
        "no_progress_seconds",
        "max_hold_seconds",
        "sweep_context_seconds",
        "momentum_context_seconds",
        "compression_context_seconds",
        "compression_range_seconds",
    )
    vals: list[int] = []
    for attr in attrs:
        if hasattr(cfg, attr):
            try:
                value = int(getattr(cfg, attr))
            except Exception:
                continue
            if value > 0:
                vals.append(value)
    base = max(vals) if vals else 0
    return max(0, int(base) + int(extra_seconds))


def build_streaming_feature_runner(
    feature_builder: Callable[[pd.DataFrame, Any], pd.DataFrame],
    chunk_runner: Callable[[pd.DataFrame, Any, Any], Any],
) -> Callable[[Iterable[HFDataWindow], Any, Any], Any]:
    """Small adapter for strategy files that want less boilerplate.

    It intentionally does not know the shape of the strategy state.
    """

    def run(windows: Iterable[HFDataWindow], cfg: Any, state: Any) -> Any:
        for window in windows:
            features = feature_builder(window.work, cfg)
            exec_features = window.slice_execution(features)
            if len(exec_features) >= 2:
                state = chunk_runner(exec_features, cfg, state)
        return state

    return run


def ensure_time_ordered(df: pd.DataFrame, *, time_col: str | None = None) -> pd.DataFrame:
    """Return a frame sorted by time with duplicate timestamps preserved."""
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index()
        if out.index.tz is None:
            out = out.copy()
            out.index = out.index.tz_localize("UTC")
        else:
            out = out.copy()
            out.index = out.index.tz_convert("UTC")
        return out

    col = time_col or infer_time_column(df)
    out = df.copy()
    if col == "ts_ms":
        ts = pd.to_datetime(pd.to_numeric(out[col], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.loc[ts.notna()].copy()
    out["_hf_stream_ts"] = ts.loc[ts.notna()]
    out = out.sort_values("_hf_stream_ts").drop(columns=["_hf_stream_ts"])
    return out


def frame_time_bounds(df: pd.DataFrame, *, time_col: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if df.empty:
        raise ValueError("cannot get time bounds from an empty frame")
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        start = idx[0]
        end = idx[-1]
    else:
        col = time_col or infer_time_column(df)
        if col == "ts_ms":
            ts = pd.to_datetime(pd.to_numeric(df[col], errors="coerce"), unit="ms", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        ts = ts.loc[ts.notna()].sort_values()
        if ts.empty:
            raise ValueError("frame has no valid timestamps")
        start = ts.iloc[0]
        end = ts.iloc[-1]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    return start, end


def tail_frame(df: pd.DataFrame, *, seconds: int = 0, rows: int | None = None, time_col: str | None = None) -> pd.DataFrame:
    """Keep a bounded FIFO tail by seconds and/or row count."""
    if df.empty:
        return pd.DataFrame()
    out = ensure_time_ordered(df, time_col=time_col)

    if seconds and seconds > 0:
        _, end = frame_time_bounds(out, time_col=time_col)
        cutoff = end - pd.Timedelta(seconds=int(seconds))
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.loc[out.index >= cutoff]
        else:
            col = time_col or infer_time_column(out)
            if col == "ts_ms":
                ts = pd.to_datetime(pd.to_numeric(out[col], errors="coerce"), unit="ms", utc=True, errors="coerce")
            else:
                ts = pd.to_datetime(out[col], utc=True, errors="coerce")
            out = out.loc[ts >= cutoff]

    if rows is not None and int(rows) > 0 and len(out) > int(rows):
        out = out.tail(int(rows))
    return out.copy()


def infer_time_column(df: pd.DataFrame) -> str:
    for col in DEFAULT_TIME_COLS:
        if col in df.columns:
            return col
    raise RuntimeError(f"no timestamp column found; expected one of {DEFAULT_TIME_COLS}, got {list(df.columns)}")


def _effective_warmup_rows(rows: int | None, *, min_rows: int, seconds: int) -> int | None:
    """Return a hard row cap only when requested, while preserving overlap.

    If seconds > 0 and rows is None, do not cap by rows; otherwise a 6-hour
    time warmup would accidentally shrink to one row.  If there is no
    seconds-based warmup, keep at least min_rows rows so next-bar execution can
    cross chunk/day boundaries.
    """
    if rows is None:
        return None if int(seconds) > 0 else int(min_rows)
    return max(int(rows), int(min_rows))


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def date_range(start_date: str | date, end_date: str | date) -> Iterator[date]:
    cur = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < cur:
        raise ValueError("end_date must be >= start_date")
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


@dataclass
class HFBacktestState:
    """Generic mutable state for stateful HF chunk backtests.

    Strategy files still own their signal/exit rules, but this state prevents
    capital/position/cooldown from resetting at day or chunk boundaries.
    """

    capital: float
    peak: float
    trades: list[dict[str, Any]] = field(default_factory=list)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)

    in_pos: bool = False
    side: int = 0
    entry_time: pd.Timestamp | None = None
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    qty_eth: float = 0.0
    entry_fee: float = 0.0
    max_fav: float = 0.0
    max_adv: float = 0.0
    last_exit_time: pd.Timestamp | None = None
    last_ts: pd.Timestamp | None = None
    last_close: float = 0.0

    @classmethod
    def initial(cls, initial_capital: float) -> "HFBacktestState":
        cap = float(initial_capital)
        return cls(capital=cap, peak=cap)

    def record_equity(self, ts: pd.Timestamp) -> None:
        self.peak = max(self.peak, self.capital)
        dd = (self.peak - self.capital) / self.peak if self.peak > 0 else 0.0
        self.equity_rows.append({"time": ts, "capital": self.capital, "drawdown_pct": dd})

    def to_result(self) -> tuple[list[dict[str, Any]], pd.DataFrame]:
        equity = pd.DataFrame(self.equity_rows).set_index("time") if self.equity_rows else pd.DataFrame()
        return self.trades, equity
