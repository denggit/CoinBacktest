#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data service for analyze_tool.

The service is a thin adapter over existing ``src.data_feed`` loaders.  It keeps
all chart-specific normalization in ``analyze_tool`` so no research/backtest
logic is pushed back into data_feed.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402

TIME_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
TRADE_BAR_TIMEFRAMES = ["1s", "5s", "10s", "15s", "30s", "1m", "5m", "15m", "30m", "1H", "4H", "1D"]
RANGE_PRESETS = [0.0015, 0.0020, 0.0025]
MAX_RETURN_BARS = 2000000


@dataclass(frozen=True)
class LoadRequest:
    data_type: str = "normal"
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    range_pct: float = 0.0020
    start: str = "2026-06-01 00:00:00"
    end: str = "2026-06-30 23:59:59"
    limit: int = 5000
    local_only: bool = True
    chunksize: int = 300_000


def config_payload() -> dict[str, Any]:
    return {
        "data_types": [
            {"id": "normal", "label": "普通K线 / OKXDataLoader", "time_based": True},
            {"id": "trade_bar", "label": "Trade Bar / OKXTradeBarLoader", "time_based": True},
            {"id": "range_bar", "label": "Range Bar / OKXRangeBarLoader", "time_based": False},
        ],
        "timeframes": {
            "normal": TIME_TIMEFRAMES,
            "trade_bar": TRADE_BAR_TIMEFRAMES,
        },
        "range_pct_presets": RANGE_PRESETS,
        "defaults": {
            "symbol": "ETH-USDT-SWAP",
            "data_type": "normal",
            "timeframe": "1m",
            "range_pct": 0.0020,
            "limit": 5000,
            "local_only": True,
        },
    }


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_request(params: dict[str, Any]) -> LoadRequest:
    data_type = str(params.get("data_type", "normal")).strip() or "normal"
    if data_type not in {"normal", "trade_bar", "range_bar"}:
        raise ValueError("data_type must be one of normal/trade_bar/range_bar")
    symbol = str(params.get("symbol", "ETH-USDT-SWAP")).strip() or "ETH-USDT-SWAP"
    timeframe = str(params.get("timeframe", "1m")).strip() or "1m"
    if data_type == "normal" and timeframe not in TIME_TIMEFRAMES:
        raise ValueError(f"normal timeframe only supports: {TIME_TIMEFRAMES}")
    if data_type == "trade_bar" and timeframe not in TRADE_BAR_TIMEFRAMES:
        # The loader supports more generic values, but the UI keeps a safe list.
        raise ValueError(f"trade_bar timeframe only supports UI presets: {TRADE_BAR_TIMEFRAMES}")
    range_pct = float(params.get("range_pct", 0.0020))
    if range_pct <= 0:
        raise ValueError("range_pct must be > 0")
    start = str(params.get("start") or params.get("start_date") or "2026-06-01 00:00:00").replace("T", " ")
    end = str(params.get("end") or params.get("end_date") or "2026-06-30 23:59:59").replace("T", " ")
    # Validate early so loader error messages are less confusing.
    pd.Timestamp(start)
    pd.Timestamp(end)
    limit = int(params.get("limit", 5000))
    if limit <= 0:
        raise ValueError("limit must be > 0")
    limit = min(limit, MAX_RETURN_BARS)
    chunksize = int(params.get("chunksize", 300_000))
    chunksize = max(10_000, min(chunksize, 2_000_000))
    local_only = parse_bool(params.get("local_only"), default=True)
    return LoadRequest(
        data_type=data_type,
        symbol=symbol,
        timeframe=timeframe,
        range_pct=range_pct,
        start=start,
        end=end,
        limit=limit,
        local_only=local_only,
        chunksize=chunksize,
    )


def load_dataframe(req: LoadRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load bars through the existing data_feed loaders."""
    meta: dict[str, Any] = {
        "data_type": req.data_type,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "range_pct": req.range_pct,
        "local_only": req.local_only,
    }
    if req.data_type == "normal":
        loader = OKXDataLoader(symbol=req.symbol, timeframe=req.timeframe)
        meta["loader"] = "src.data_feed.okx_loader.OKXDataLoader"
        meta["db_path"] = str(Path(loader.db_path))
        meta["table_name"] = loader.table_name
        if req.local_only:
            df = loader.load_local_data()
            df = _slice_indexed(df, req.start, req.end)
        else:
            df = loader.fetch_data_by_date_range(req.start, req.end)
    elif req.data_type == "trade_bar":
        loader = OKXTradeBarLoader(symbol=req.symbol, timeframe=req.timeframe)
        meta["loader"] = "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader"
        meta["db_path"] = str(loader.db_path)
        meta["table_name"] = loader.table_name
        if req.local_only:
            df = loader.load_local_data(req.start, req.end)
        else:
            df = loader.fetch_data_by_date_range(req.start, req.end, chunksize=req.chunksize, build_missing=True)
    else:
        loader = OKXRangeBarLoader(symbol=req.symbol, range_pct=req.range_pct)
        meta["loader"] = "src.data_feed.okx_range_bar_loader.OKXRangeBarLoader"
        meta["db_path"] = str(loader.db_path)
        meta["table_name"] = loader.table_name
        if req.local_only:
            df = loader.load_local_data(req.start, req.end)
        else:
            df = loader.fetch_data_by_date_range(req.start, req.end, chunksize=req.chunksize)
    df = _prepare_ohlcv_dataframe(df, req.data_type).tail(req.limit)
    meta["rows"] = int(len(df))
    if not df.empty:
        meta["start"] = pd.Timestamp(df.index[0]).strftime("%Y-%m-%d %H:%M:%S")
        meta["end"] = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d %H:%M:%S")
    return df, meta


def _slice_indexed(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return out[(out.index >= start_ts) & (out.index <= end_ts)]


def _prepare_ohlcv_dataframe(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    if data_type == "range_bar" and "end_ts" in out.columns:
        out.index = pd.to_datetime(out["end_ts"])
        out.index.name = "timestamp"
    else:
        out.index = pd.to_datetime(out.index)
        out.index.name = "timestamp"
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            raise ValueError(f"loaded dataframe missing required OHLCV column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out.sort_index()


def dataframe_to_candles(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    candles: list[dict[str, Any]] = []
    if df is None or df.empty:
        return {"candles": candles, "meta": meta}
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        extra: dict[str, Any] = {}
        for col, value in row.items():
            if col in {"open", "high", "low", "close", "volume"}:
                continue
            extra[col] = to_json_value(value)
        candles.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "time": int(ts.timestamp() * 1000),
                "open": to_json_value(row["open"]),
                "high": to_json_value(row["high"]),
                "low": to_json_value(row["low"]),
                "close": to_json_value(row["close"]),
                "volume": to_json_value(row.get("volume", 0.0)),
                "extra": extra,
            }
        )
    return {"candles": candles, "meta": meta}


def to_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if pd.isna(value):
        return None
    return str(value)
