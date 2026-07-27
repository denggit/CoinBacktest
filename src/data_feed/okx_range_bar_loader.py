#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build and load OKX trade-derived Range Bars with SQLite cache.

Range bars are event bars: a bar starts at the first trade price and closes once
price moves at least ``range_pct`` away from the bar open.  The loader streams
raw OKX official trades ZIP files, never loading a multi-year tick dataset into
memory.

Default tables for ETH-USDT-SWAP:
    ETH_USDT_SWAP_range_bars_r0015
    ETH_USDT_SWAP_range_bars_r0020
    ETH_USDT_SWAP_range_bars_r0025
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

try:
    from src.data_feed.okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader
except ImportError:  # pragma: no cover
    from okx_tick_loader import DEFAULT_OKX_TRADES_URL_TEMPLATE, OKXTickLoader

try:
    from src.utils.log import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

DEFAULT_RANGE_PCTS = (0.0015, 0.0020, 0.0025)
DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD = 100_000.0
BAR_ID_MULT = 1_000_000


def safe_symbol(symbol: str) -> str:
    return str(symbol).replace("-", "_").replace("/", "_").replace("\\", "_").upper()


def range_code(range_pct: float) -> str:
    # 0.15% = 0.0015 = 15 bps -> r0015
    bps = int(round(float(range_pct) * 10_000))
    return f"r{bps:04d}"


def table_name_for_range_bars(symbol: str, range_pct: float) -> str:
    return f"{safe_symbol(symbol)}_range_bars_{range_code(range_pct)}"


def parse_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterator[date]:
    if end < start:
        raise ValueError(f"end_date must be >= start_date, got {start} -> {end}")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parse_timezone_offset_hours(value: str) -> int:
    text = str(value or "+0").strip()
    if not text:
        return 0
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("+-")
    try:
        return sign * int(float(text))
    except ValueError:
        return 0


def timestamp_to_db_text(ts: Any) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def normalize_trade_chunk_fast(raw: pd.DataFrame) -> pd.DataFrame:
    """Fast normalizer for official OKX trade CSV chunks."""
    rename: dict[str, str] = {}
    for col in raw.columns:
        low = str(col).strip().lower()
        if low in {"ts", "timestamp", "time", "datetime", "created_time", "createdtime", "create_time", "created_at", "createdat"}:
            rename[col] = "ts_ms"
        elif low in {"px", "price"}:
            rename[col] = "price"
        elif low in {"sz", "size", "qty", "amount"}:
            rename[col] = "size"
        elif low == "side":
            rename[col] = "side"
    df = raw.rename(columns=rename)
    required = {"ts_ms", "price", "size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"trade file missing columns {sorted(missing)}; columns={list(raw.columns)}")

    ts_num = pd.to_numeric(df["ts_ms"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    size = pd.to_numeric(df["size"], errors="coerce")

    if ts_num.notna().mean() > 0.9:
        ok = ts_num.notna() & price.notna() & size.notna()
        if not ok.any():
            return pd.DataFrame(columns=["timestamp", "price", "size", "side"])
        ts_num = ts_num.loc[ok]
        if ts_num.max() < 10_000_000_000:
            ts_num = ts_num * 1000
        timestamp = pd.to_datetime(ts_num.astype("int64"), unit="ms", utc=True)
    else:
        ts = pd.to_datetime(df["ts_ms"], utc=True, errors="coerce")
        ok = ts.notna() & price.notna() & size.notna()
        if not ok.any():
            return pd.DataFrame(columns=["timestamp", "price", "size", "side"])
        timestamp = ts.loc[ok]

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "price": price.loc[ok].astype(float),
            "size": size.loc[ok].astype(float),
            "side": df.get("side", "").loc[ok].astype(str).str.lower() if "side" in df.columns else "",
        }
    )
    # Official OKX files are normally already chronological.  Avoid allocating
    # a full argsort index for every million-row chunk unless it is actually
    # needed; this is the allocation that commonly fails under memory pressure.
    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp", kind="stable")
    return out.reset_index(drop=True)


def iter_trade_csv_chunks(path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    wanted = {
        "ts", "timestamp", "time", "datetime", "created_time", "createdtime",
        "create_time", "created_at", "createdat", "px", "price", "sz", "size",
        "qty", "amount", "side",
    }

    def use_column(name: str) -> bool:
        return str(name).strip().lower() in wanted

    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if not members:
                raise RuntimeError(f"empty OKX trade ZIP: {p}")
            for name in members:
                with zf.open(name) as f:
                    yield from pd.read_csv(f, chunksize=chunksize, usecols=use_column)
        return
    yield from pd.read_csv(p, chunksize=chunksize, usecols=use_column)


@dataclass
class _ActiveRangeBar:
    bar_id: int
    start_ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    notional: float = 0.0
    trades_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    buy_trades_count: int = 0
    sell_trades_count: int = 0
    large_buy_notional: float = 0.0
    large_sell_notional: float = 0.0
    large_trades_count: int = 0
    price_size_sum: float = 0.0
    max_trade_notional: float = 0.0
    footprints: dict[float, dict[str, float]] = field(default_factory=dict)


class RangeBarBuilder:
    """Stateful range-bar builder that can be fed streaming trade chunks."""

    def __init__(
        self,
        *,
        range_pct: float,
        contract_value: float,
        large_trade_notional_threshold: float = DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
        price_step: float | None = None,
    ):
        if range_pct <= 0:
            raise ValueError("range_pct must be > 0")
        if price_step is not None and price_step <= 0:
            raise ValueError("price_step must be > 0")
        self.range_pct = float(range_pct)
        self.contract_value = float(contract_value)
        self.large_trade_notional_threshold = float(large_trade_notional_threshold)
        self.price_step = None if price_step is None else float(price_step)
        self.active: _ActiveRangeBar | None = None
        self.day_seq: dict[str, int] = {}
        self.cvd_volume = 0.0
        self.cvd_notional = 0.0

    def export_state(self) -> dict[str, Any]:
        """Return a JSON-serializable exact continuation checkpoint.

        Range bars are path dependent.  Recreating a builder one day before a
        gap is not sufficient because the active bar can cross UTC midnight and
        CVD is cumulative.  The checkpoint therefore contains every mutable
        field required to continue with byte-for-byte equivalent bar boundaries.
        """

        active: dict[str, Any] | None = None
        if self.active is not None:
            active = {
                "bar_id": int(self.active.bar_id),
                "start_ts": pd.Timestamp(self.active.start_ts).isoformat(),
                "open": float(self.active.open),
                "high": float(self.active.high),
                "low": float(self.active.low),
                "close": float(self.active.close),
                "volume": float(self.active.volume),
                "notional": float(self.active.notional),
                "trades_count": int(self.active.trades_count),
                "buy_volume": float(self.active.buy_volume),
                "sell_volume": float(self.active.sell_volume),
                "buy_notional": float(self.active.buy_notional),
                "sell_notional": float(self.active.sell_notional),
                "buy_trades_count": int(self.active.buy_trades_count),
                "sell_trades_count": int(self.active.sell_trades_count),
                "large_buy_notional": float(self.active.large_buy_notional),
                "large_sell_notional": float(self.active.large_sell_notional),
                "large_trades_count": int(self.active.large_trades_count),
                "price_size_sum": float(self.active.price_size_sum),
                "max_trade_notional": float(self.active.max_trade_notional),
                "footprints": {
                    repr(float(bucket)): {str(k): float(v) for k, v in values.items()}
                    for bucket, values in self.active.footprints.items()
                },
            }
        return {
            "version": 1,
            "config": {
                "range_pct": self.range_pct,
                "contract_value": self.contract_value,
                "large_trade_notional_threshold": self.large_trade_notional_threshold,
                "price_step": self.price_step,
            },
            "active": active,
            "day_seq": {str(k): int(v) for k, v in self.day_seq.items()},
            "cvd_volume": float(self.cvd_volume),
            "cvd_notional": float(self.cvd_notional),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore an exact continuation checkpoint into this builder."""

        if int(state.get("version", 0)) != 1:
            raise ValueError(f"unsupported RangeBarBuilder checkpoint version: {state.get('version')!r}")
        config = dict(state.get("config") or {})
        expected = {
            "range_pct": self.range_pct,
            "contract_value": self.contract_value,
            "large_trade_notional_threshold": self.large_trade_notional_threshold,
            "price_step": self.price_step,
        }
        for key, expected_value in expected.items():
            actual = config.get(key)
            if expected_value is None:
                if actual is not None:
                    raise ValueError(f"checkpoint config mismatch for {key}: {actual!r} != None")
            elif actual is None or not math.isclose(float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"checkpoint config mismatch for {key}: {actual!r} != {expected_value!r}")

        self.day_seq = {str(k): int(v) for k, v in dict(state.get("day_seq") or {}).items()}
        self.cvd_volume = float(state.get("cvd_volume", 0.0))
        self.cvd_notional = float(state.get("cvd_notional", 0.0))
        active = state.get("active")
        if not active:
            self.active = None
            return
        footprints = {
            float(bucket): {str(k): float(v) for k, v in dict(values).items()}
            for bucket, values in dict(active.get("footprints") or {}).items()
        }
        self.active = _ActiveRangeBar(
            bar_id=int(active["bar_id"]),
            start_ts=pd.Timestamp(active["start_ts"]),
            open=float(active["open"]),
            high=float(active["high"]),
            low=float(active["low"]),
            close=float(active["close"]),
            volume=float(active.get("volume", 0.0)),
            notional=float(active.get("notional", 0.0)),
            trades_count=int(active.get("trades_count", 0)),
            buy_volume=float(active.get("buy_volume", 0.0)),
            sell_volume=float(active.get("sell_volume", 0.0)),
            buy_notional=float(active.get("buy_notional", 0.0)),
            sell_notional=float(active.get("sell_notional", 0.0)),
            buy_trades_count=int(active.get("buy_trades_count", 0)),
            sell_trades_count=int(active.get("sell_trades_count", 0)),
            large_buy_notional=float(active.get("large_buy_notional", 0.0)),
            large_sell_notional=float(active.get("large_sell_notional", 0.0)),
            large_trades_count=int(active.get("large_trades_count", 0)),
            price_size_sum=float(active.get("price_size_sum", 0.0)),
            max_trade_notional=float(active.get("max_trade_notional", 0.0)),
            footprints=footprints,
        )

    def process_chunk(self, trades: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bars: list[dict[str, Any]] = []
        footprints: list[dict[str, Any]] = []
        if trades.empty:
            return bars, footprints

        for row in trades.itertuples(index=False):
            ts = pd.Timestamp(row.timestamp)
            price = float(row.price)
            size = float(row.size)
            side = str(row.side).lower()
            if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size <= 0:
                continue
            if self.active is None:
                self.active = self._new_bar(ts, price)
            self._add_trade(self.active, ts, price, size, side)
            if self._should_close(self.active, price):
                bar, fps = self._close_bar(self.active, ts)
                bars.append(bar)
                footprints.extend(fps)
                self.active = None
        return bars, footprints

    def _new_bar(self, ts: pd.Timestamp, price: float) -> _ActiveRangeBar:
        d = ts.strftime("%Y%m%d")
        seq = self.day_seq.get(d, 0) + 1
        self.day_seq[d] = seq
        return _ActiveRangeBar(
            bar_id=int(d) * BAR_ID_MULT + seq,
            start_ts=ts,
            open=price,
            high=price,
            low=price,
            close=price,
        )

    def _add_trade(self, bar: _ActiveRangeBar, ts: pd.Timestamp, price: float, size: float, side: str) -> None:
        notional = price * size * self.contract_value
        is_buy = side == "buy"
        is_sell = side == "sell"
        is_large = abs(notional) >= self.large_trade_notional_threshold

        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        bar.volume += size
        bar.notional += notional
        bar.trades_count += 1
        bar.price_size_sum += price * size
        bar.max_trade_notional = max(bar.max_trade_notional, abs(notional))
        if is_buy:
            bar.buy_volume += size
            bar.buy_notional += notional
            bar.buy_trades_count += 1
            if is_large:
                bar.large_buy_notional += notional
        elif is_sell:
            bar.sell_volume += size
            bar.sell_notional += notional
            bar.sell_trades_count += 1
            if is_large:
                bar.large_sell_notional += notional
        if is_large:
            bar.large_trades_count += 1

        if self.price_step is not None:
            bucket = self._price_bucket(price)
            fp = bar.footprints.setdefault(bucket, self._new_footprint_bucket(bucket))
            fp["volume"] += size
            fp["notional"] += notional
            fp["trades_count"] += 1
            fp["max_trade_notional"] = max(fp["max_trade_notional"], abs(notional))
            if is_buy:
                fp["buy_volume"] += size
                fp["buy_notional"] += notional
                fp["buy_trades_count"] += 1
                if is_large:
                    fp["large_buy_notional"] += notional
            elif is_sell:
                fp["sell_volume"] += size
                fp["sell_notional"] += notional
                fp["sell_trades_count"] += 1
                if is_large:
                    fp["large_sell_notional"] += notional
            if is_large:
                fp["large_trades_count"] += 1

    def _should_close(self, bar: _ActiveRangeBar, price: float) -> bool:
        up = bar.open * (1.0 + self.range_pct)
        down = bar.open * (1.0 - self.range_pct)
        return price >= up or price <= down

    def _close_bar(self, bar: _ActiveRangeBar, end_ts: pd.Timestamp) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        delta_volume = bar.buy_volume - bar.sell_volume
        delta_notional = bar.buy_notional - bar.sell_notional
        self.cvd_volume += delta_volume
        self.cvd_notional += delta_notional
        duration = max(0.0, (end_ts - bar.start_ts).total_seconds())
        range_size = bar.high - bar.low
        actual_range_pct = range_size / bar.open if bar.open > 0 else 0.0
        large_delta = bar.large_buy_notional - bar.large_sell_notional
        direction = 1 if bar.close > bar.open else (-1 if bar.close < bar.open else 0)
        vwap = bar.price_size_sum / bar.volume if bar.volume > 0 else bar.close
        out = {
            "bar_id": bar.bar_id,
            "start_ts": bar.start_ts,
            "end_ts": end_ts,
            "duration_seconds": duration,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "range_pct": actual_range_pct,
            "range_size": range_size,
            "direction": direction,
            "volume": bar.volume,
            "notional": bar.notional,
            "trades_count": bar.trades_count,
            "buy_volume": bar.buy_volume,
            "sell_volume": bar.sell_volume,
            "buy_notional": bar.buy_notional,
            "sell_notional": bar.sell_notional,
            "buy_trades_count": bar.buy_trades_count,
            "sell_trades_count": bar.sell_trades_count,
            "delta_volume": delta_volume,
            "delta_notional": delta_notional,
            "cvd_volume": self.cvd_volume,
            "cvd_notional": self.cvd_notional,
            "taker_buy_ratio": bar.buy_volume / bar.volume if bar.volume > 0 else 0.0,
            "large_buy_notional": bar.large_buy_notional,
            "large_sell_notional": bar.large_sell_notional,
            "large_delta_notional": large_delta,
            "large_trades_count": bar.large_trades_count,
            "vwap": vwap,
            "max_trade_notional": bar.max_trade_notional,
        }
        fps: list[dict[str, Any]] = []
        for bucket, fp in sorted(bar.footprints.items()):
            fp_delta_volume = fp["buy_volume"] - fp["sell_volume"]
            fp_delta_notional = fp["buy_notional"] - fp["sell_notional"]
            fps.append(
                {
                    "bar_id": bar.bar_id,
                    "start_ts": bar.start_ts,
                    "end_ts": end_ts,
                    "price_bucket": bucket,
                    "volume": fp["volume"],
                    "notional": fp["notional"],
                    "trades_count": fp["trades_count"],
                    "buy_volume": fp["buy_volume"],
                    "sell_volume": fp["sell_volume"],
                    "buy_notional": fp["buy_notional"],
                    "sell_notional": fp["sell_notional"],
                    "buy_trades_count": fp["buy_trades_count"],
                    "sell_trades_count": fp["sell_trades_count"],
                    "delta_volume": fp_delta_volume,
                    "delta_notional": fp_delta_notional,
                    "large_buy_notional": fp["large_buy_notional"],
                    "large_sell_notional": fp["large_sell_notional"],
                    "large_delta_notional": fp["large_buy_notional"] - fp["large_sell_notional"],
                    "large_trades_count": fp["large_trades_count"],
                    "max_trade_notional": fp["max_trade_notional"],
                }
            )
        return out, fps

    def _price_bucket(self, price: float) -> float:
        assert self.price_step is not None
        return round(math.floor(price / self.price_step) * self.price_step, 8)

    def _new_footprint_bucket(self, bucket: float) -> dict[str, float]:
        return {
            "price_bucket": bucket,
            "volume": 0.0,
            "notional": 0.0,
            "trades_count": 0.0,
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "buy_trades_count": 0.0,
            "sell_trades_count": 0.0,
            "delta_volume": 0.0,
            "delta_notional": 0.0,
            "large_buy_notional": 0.0,
            "large_sell_notional": 0.0,
            "large_delta_notional": 0.0,
            "large_trades_count": 0.0,
            "max_trade_notional": 0.0,
        }


class OKXRangeBarLoader:
    """Load/build OKX range bars from raw trades ZIP files."""

    BASE_COLUMNS = [
        "bar_id",
        "start_ts",
        "end_ts",
        "duration_seconds",
        "open",
        "high",
        "low",
        "close",
        "range_pct",
        "range_size",
        "direction",
        "volume",
        "notional",
        "trades_count",
        "buy_volume",
        "sell_volume",
        "buy_notional",
        "sell_notional",
        "buy_trades_count",
        "sell_trades_count",
        "delta_volume",
        "delta_notional",
        "cvd_volume",
        "cvd_notional",
        "taker_buy_ratio",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
        "large_trades_count",
        "vwap",
        "max_trade_notional",
    ]

    NUMERIC_COLUMNS = [c for c in BASE_COLUMNS if c not in {"bar_id", "start_ts", "end_ts"}]

    def __init__(
        self,
        symbol: str = "ETH-USDT-SWAP",
        range_pct: float = 0.0020,
        data_dir: str | os.PathLike[str] | None = None,
        db_name: str = "okx_range_bars.db",
        trades_url_template: str = DEFAULT_OKX_TRADES_URL_TEMPLATE,
        contract_value: float | None = None,
        large_trade_notional_threshold: float = DEFAULT_LARGE_TRADE_NOTIONAL_THRESHOLD,
        align_with_okx_loader_timezone: bool = True,
        initialize_db: bool = True,
    ):
        self.symbol = symbol
        self.range_pct = float(range_pct)
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / db_name
        self.trades_url_template = trades_url_template
        self.contract_value = self._infer_contract_value(symbol) if contract_value is None else float(contract_value)
        self.large_trade_notional_threshold = float(large_trade_notional_threshold)
        self.align_with_okx_loader_timezone = bool(align_with_okx_loader_timezone)
        self.timezone_offset_hours = parse_timezone_offset_hours(TIMEZONE) if self.align_with_okx_loader_timezone else 0
        self.tick_loader = OKXTickLoader(symbol=self.symbol, data_dir=self.data_dir, trades_url_template=self.trades_url_template)
        self.table_name = table_name_for_range_bars(self.symbol, self.range_pct)
        self.coverage_table_name = "range_bar_coverage"
        if initialize_db:
            self._init_db()

    def fetch_data_by_date_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
        cvd_mode: str = "range",
    ) -> pd.DataFrame:
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        if end_ts < start_ts:
            raise ValueError(f"end_date must be >= start_date, got {start_ts} -> {end_ts}")
        self.ensure_cached_range(start_ts, end_ts, chunksize=chunksize, force_rebuild=force_rebuild)
        df = self.load_local_data(start_date=start_ts, end_date=end_ts)
        if not df.empty and cvd_mode == "range":
            df = self._recompute_cvd(df)
        return df

    def fetch_historical_data(
        self,
        limit: int = 50_000,
        *,
        end_date: str | datetime | pd.Timestamp | None = None,
        lookback_days: int = 30,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> pd.DataFrame:
        end_ts = self._normalize_query_timestamp(end_date or pd.Timestamp.now(), is_end=True)
        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
        df = self.fetch_data_by_date_range(start_ts, end_ts, chunksize=chunksize, force_rebuild=force_rebuild)
        return df.tail(limit)

    def ensure_cached_range(
        self,
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        *,
        chunksize: int = 300_000,
        force_rebuild: bool = False,
    ) -> None:
        start_ts = self._normalize_query_timestamp(start_date, is_end=False)
        end_ts = self._normalize_query_timestamp(end_date, is_end=True)
        days = list(self._required_utc_days_for_local_range(start_ts, end_ts))
        missing = [d for d in days if force_rebuild or not self._has_coverage(d)]
        if not missing:
            return
        self.build_range_to_db(days, chunksize=chunksize, force_rebuild=force_rebuild)

    def ensure_cached_days(self, utc_days: Iterable[date], *, chunksize: int = 300_000, force_rebuild: bool = False) -> None:
        days = [parse_date(d) for d in utc_days]
        missing = [d for d in days if force_rebuild or not self._has_coverage(d)]
        if missing:
            self.build_range_to_db(days, chunksize=chunksize, force_rebuild=force_rebuild)

    def build_range_to_db(self, utc_days: Sequence[date], *, chunksize: int = 300_000, force_rebuild: bool = False) -> dict[str, int]:
        days = [parse_date(d) for d in utc_days]
        if not days:
            return {"bars_written": 0, "chunks_read": 0, "days": 0}
        self._assert_legacy_build_is_safe(days, force_rebuild=force_rebuild)
        if force_rebuild:
            self._delete_cached_days(days)
        builder = RangeBarBuilder(
            range_pct=self.range_pct,
            contract_value=self.contract_value,
            large_trade_notional_threshold=self.large_trade_notional_threshold,
            price_step=None,
        )
        bars_written = 0
        chunks_read = 0
        for day in days:
            if not force_rebuild and self._has_coverage(day):
                logger.info("[RANGE-BAR-SKIP] symbol=%s range=%s utc_day=%s", self.symbol, range_code(self.range_pct), day)
                continue
            raw_file = self._ensure_raw_trade_file(day)
            logger.info("[RANGE-BAR-DAY-START] symbol=%s range=%s utc_day=%s raw=%s", self.symbol, range_code(self.range_pct), day, raw_file)
            day_rows = 0
            for raw in iter_trade_csv_chunks(raw_file, chunksize=chunksize):
                chunks_read += 1
                chunk = normalize_trade_chunk_fast(raw)
                bars, _ = builder.process_chunk(chunk)
                if bars:
                    df = self._bars_to_frame(bars)
                    self._upsert_bars(df)
                    bars_written += len(df)
                    day_rows += len(df)
            self._mark_coverage(day, rows=day_rows)
            logger.info("[RANGE-BAR-DAY-DONE] symbol=%s range=%s utc_day=%s rows=%s", self.symbol, range_code(self.range_pct), day, day_rows)
        return {"bars_written": bars_written, "chunks_read": chunks_read, "days": len(days)}

    def aggregate_trades_df(self, trades: pd.DataFrame) -> pd.DataFrame:
        chunk = normalize_trade_chunk_fast(trades) if "ts_ms" in trades.columns or "created_time" in trades.columns else trades
        builder = RangeBarBuilder(
            range_pct=self.range_pct,
            contract_value=self.contract_value,
            large_trade_notional_threshold=self.large_trade_notional_threshold,
            price_step=None,
        )
        bars, _ = builder.process_chunk(chunk)
        return self._bars_to_frame(bars)

    def load_local_data(
        self,
        start_date: Any | None = None,
        end_date: Any | None = None,
        *,
        connection: sqlite3.Connection | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        selected_columns = list(self.BASE_COLUMNS if columns is None else columns)
        unknown = sorted(set(selected_columns) - set(self.BASE_COLUMNS))
        if unknown:
            raise ValueError(f"unknown range-bar columns: {unknown}")
        if not selected_columns:
            raise ValueError("columns must not be empty")
        where = []
        params: list[Any] = []
        if start_date is not None:
            where.append("end_ts >= ?")
            params.append(timestamp_to_db_text(start_date))
        if end_date is not None:
            where.append("start_ts <= ?")
            params.append(timestamp_to_db_text(end_date))
        sql = f"SELECT {', '.join(selected_columns)} FROM {self.table_name}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY end_ts, bar_id"
        owns_connection = connection is None
        conn = self._get_read_db_connection() if owns_connection else connection
        try:
            parse_dates = [name for name in ("start_ts", "end_ts") if name in selected_columns]
            df = pd.read_sql_query(sql, conn, params=params, parse_dates=parse_dates)
        except Exception as exc:
            logger.warning("读取 range bar DB 失败 table=%s error=%s", self.table_name, exc)
            return pd.DataFrame(columns=selected_columns)
        finally:
            if owns_connection:
                conn.close()
        if df.empty:
            return pd.DataFrame(columns=selected_columns)
        if columns is None:
            return self._finalize_return_df(df)
        out = df.copy()
        for col in ("start_ts", "end_ts"):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        for col in set(selected_columns).intersection(self.NUMERIC_COLUMNS):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        if "bar_id" in out.columns:
            out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
        if "end_ts" in out.columns:
            out = out.dropna(subset=["end_ts"])
            sort_columns = [name for name in ("end_ts", "bar_id") if name in out.columns]
            out = out.sort_values(sort_columns, kind="mergesort")
            out = out.set_index("end_ts", drop=False)
            out.index.name = "end_ts"
        return out[selected_columns]

    def delete_cache(self) -> None:
        with self._get_db_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
            conn.execute(f"DELETE FROM {self.coverage_table_name} WHERE cache_key = ?", (self._cache_key(),))
            conn.commit()

    def _ensure_raw_trade_file(self, day: date) -> Path:
        raw_file = self.tick_loader.find_local_trade_file(day, template=self.trades_url_template)
        if raw_file is not None and raw_file.exists() and raw_file.stat().st_size > 0:
            return raw_file
        return self.tick_loader.download_official_trade_file(day, self.trades_url_template)

    def _bars_to_frame(self, bars: list[dict[str, Any]]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        df = pd.DataFrame(bars)
        if self.timezone_offset_hours:
            df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True) + pd.Timedelta(hours=self.timezone_offset_hours)
            df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True) + pd.Timedelta(hours=self.timezone_offset_hours)
        else:
            df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
            df["end_ts"] = pd.to_datetime(df["end_ts"], utc=True)
        df["start_ts"] = df["start_ts"].dt.tz_localize(None)
        df["end_ts"] = df["end_ts"].dt.tz_localize(None)
        return self._finalize_return_df(df)

    def _finalize_return_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        out = df.copy()
        for col in ["start_ts", "end_ts"]:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        for col in self.NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
        out = out.dropna(subset=["start_ts", "end_ts"]).sort_values(["end_ts", "bar_id"])
        out = out.set_index("end_ts", drop=False)
        out.index.name = "end_ts"
        return out[self.BASE_COLUMNS]

    def _recompute_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["cvd_volume"] = out["delta_volume"].cumsum()
        out["cvd_notional"] = out["delta_notional"].cumsum()
        return out

    def open_read_connection(self) -> sqlite3.Connection:
        """Return a query-only shared connection for multi-table research reads.

        The caller owns the returned connection and must close it.
        """
        return self._get_read_db_connection()

    def _get_read_db_connection(self) -> sqlite3.Connection:
        """Open a query-only connection without negotiating WAL mode.

        Repeated research reads must not request a journal-mode transition. On
        some Windows/network filesystems that transition can wait behind a
        prior reader even though no write is needed.
        """
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")
        conn.execute("PRAGMA mmap_size=268435456")
        return conn

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")  # about 256 MiB page cache when possible
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA wal_autocheckpoint=10000")
        return conn

    def _init_db(self) -> None:
        numeric_cols_sql = ",\n                    ".join(f"{col} REAL" for col in self.NUMERIC_COLUMNS)
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    bar_id INTEGER PRIMARY KEY,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    {numeric_cols_sql}
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_end_ts ON {self.table_name}(end_ts)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_start_ts ON {self.table_name}(start_ts)")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.coverage_table_name} (
                    cache_key TEXT NOT NULL,
                    utc_day TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    range_pct REAL NOT NULL,
                    table_name TEXT NOT NULL,
                    rows INTEGER NOT NULL DEFAULT 0,
                    params_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (cache_key, utc_day)
                )
                """
            )
            conn.commit()

    def _upsert_bars(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        clean = df.reset_index(drop=True).copy()
        clean["start_ts"] = pd.to_datetime(clean["start_ts"]).map(timestamp_to_db_text)
        clean["end_ts"] = pd.to_datetime(clean["end_ts"]).map(timestamp_to_db_text)
        for col in self.NUMERIC_COLUMNS:
            clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)
        clean["bar_id"] = pd.to_numeric(clean["bar_id"], errors="coerce").astype("int64")
        db_cols = self.BASE_COLUMNS
        placeholders = ",".join(["?"] * len(db_cols))
        update_cols = [c for c in db_cols if c != "bar_id"]
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in update_cols])
        sql = f"""
            INSERT INTO {self.table_name} ({', '.join(db_cols)})
            VALUES ({placeholders})
            ON CONFLICT(bar_id) DO UPDATE SET {update_sql}
        """
        with self._get_db_connection() as conn:
            conn.executemany(sql, clean[db_cols].itertuples(index=False, name=None))
            conn.commit()

    def _delete_cached_days(self, days: Sequence[date]) -> None:
        if not days:
            return
        with self._get_db_connection() as conn:
            for day in days:
                start_ts, end_ts = self._utc_day_db_bounds(day)
                conn.execute(
                    f"DELETE FROM {self.table_name} WHERE end_ts >= ? AND end_ts < ?",
                    (timestamp_to_db_text(start_ts), timestamp_to_db_text(end_ts)),
                )
                conn.execute(f"DELETE FROM {self.coverage_table_name} WHERE cache_key = ? AND utc_day = ?", (self._cache_key(), day.isoformat()))
            conn.commit()

    def _utc_day_db_bounds(self, utc_day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = pd.Timestamp(utc_day)
        if self.timezone_offset_hours:
            start += pd.Timedelta(hours=self.timezone_offset_hours)
        return start, start + pd.Timedelta(days=1)

    def _earliest_coverage_day(self) -> date | None:
        with self._get_db_connection() as conn:
            row = conn.execute(
                f"SELECT MIN(utc_day) FROM {self.coverage_table_name} WHERE cache_key = ?",
                (self._cache_key(),),
            ).fetchone()
        return None if not row or not row[0] else date.fromisoformat(str(row[0]))

    def _latest_coverage_day(self) -> date | None:
        with self._get_db_connection() as conn:
            row = conn.execute(
                f"SELECT MAX(utc_day) FROM {self.coverage_table_name} WHERE cache_key = ?",
                (self._cache_key(),),
            ).fetchone()
        return None if not row or not row[0] else date.fromisoformat(str(row[0]))

    def _assert_legacy_build_is_safe(self, days: Sequence[date], *, force_rebuild: bool) -> None:
        """Prevent path-dependent incremental builds without exact state.

        The loader-level builder has no persistent active-bar checkpoint.  It is
        safe only for a brand-new cache or a full-cache rebuild.  Incremental
        updates must go through tools/prebuild_okx_range_all.py.
        """

        ordered_days = sorted(set(days))
        if len(ordered_days) != (ordered_days[-1] - ordered_days[0]).days + 1:
            raise RuntimeError("Unsafe non-contiguous range-bar build blocked; UTC days must be contiguous.")
        first = self._earliest_coverage_day()
        if first is None:
            return
        last = self._latest_coverage_day()
        assert last is not None
        requested_first = ordered_days[0]
        requested_last = ordered_days[-1]
        if force_rebuild and requested_first <= first and requested_last >= last:
            return
        missing = [day for day in ordered_days if not self._has_coverage(day)]
        if not missing and not force_rebuild:
            return
        raise RuntimeError(
            "Unsafe incremental range-bar build blocked: range bars are path dependent and this loader "
            "cannot restore the cross-day active bar/CVD state. Run "
            "python tools/prebuild_okx_range_all.py with the requested date range instead."
        )

    def _has_coverage(self, utc_day: date) -> bool:
        with self._get_db_connection() as conn:
            row = conn.execute(
                f"SELECT rows FROM {self.coverage_table_name} WHERE cache_key = ? AND utc_day = ?",
                (self._cache_key(), utc_day.isoformat()),
            ).fetchone()
        return row is not None

    def _mark_coverage(self, utc_day: date, *, rows: int) -> None:
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        params = {
            "symbol": self.symbol,
            "range_pct": self.range_pct,
            "contract_value": self.contract_value,
            "large_trade_notional_threshold": self.large_trade_notional_threshold,
            "align_with_okx_loader_timezone": self.align_with_okx_loader_timezone,
            "timezone": TIMEZONE,
        }
        with self._get_db_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.coverage_table_name}
                    (cache_key, utc_day, symbol, range_pct, table_name, rows, params_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key, utc_day) DO UPDATE SET
                    rows=excluded.rows,
                    params_json=excluded.params_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self._cache_key(),
                    utc_day.isoformat(),
                    self.symbol,
                    self.range_pct,
                    self.table_name,
                    int(rows),
                    json.dumps(params, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()

    def _cache_key(self) -> str:
        return f"{self.symbol}|{range_code(self.range_pct)}|cv={self.contract_value}|large={self.large_trade_notional_threshold}|tz={self.timezone_offset_hours}"

    def _required_utc_days_for_local_range(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> Iterator[date]:
        if self.timezone_offset_hours:
            raw_start = start_ts - pd.Timedelta(hours=self.timezone_offset_hours)
            raw_end = end_ts - pd.Timedelta(hours=self.timezone_offset_hours)
        else:
            raw_start = start_ts
            raw_end = end_ts
        yield from date_range(raw_start.date(), raw_end.date())

    def _normalize_query_timestamp(self, value: Any, *, is_end: bool) -> pd.Timestamp:
        if isinstance(value, str) and len(value) == 10:
            suffix = " 23:59:59.999" if is_end else " 00:00:00.000"
            return pd.Timestamp(value + suffix)
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts

    def _infer_contract_value(self, symbol: str) -> float:
        s = str(symbol).upper()
        if s.endswith("-SWAP") and s.startswith("ETH-"):
            return 0.1
        return 1.0
