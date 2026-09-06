from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader

from .config import ResearchConfig


UTC = "UTC"
BEIJING = "Asia/Shanghai"

CNBC_CHART_ENDPOINT = "https://webql-redesign.cnbcfm.com/graphql"
CNBC_CHART_QUERY_HASH = "9e1670c29a10707c417a1efd327d4b2b1d456b77f1426e7e84fb7d399416bb6b"
OKX_HISTORY_ENDPOINT = "https://www.okx.com/api/v5/market/history-candles"


@dataclass(frozen=True)
class Coverage:
    source: str
    dataset: str
    rows: int
    start_utc: str | None
    end_utc: str | None
    notes: str = ""


def _utc_index(values: Iterable[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))


def _target_midpoint(target_range: object) -> float:
    text = str(target_range or "").strip().replace("%", "")
    parts = text.split("-")
    if len(parts) != 2:
        return float("nan")
    try:
        return (float(parts[0]) + float(parts[1])) / 2.0
    except ValueError:
        return float("nan")


def load_monitor_observations(db_path: str | Path) -> pd.DataFrame:
    """Read only the metrics needed by this research from the live monitor DB."""

    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    metrics = (
        "fedwatch_cut_probability",
        "fedwatch_hold_probability",
        "fedwatch_hike_probability",
        "fedwatch_expected_rate",
        "fedwatch_target_probability",
        "us2y_yield",
        "us10y_yield",
        "us10y_2y_spread",
        "dxy_index",
    )
    placeholders = ",".join("?" for _ in metrics)
    sql = f"""
        SELECT timestamp_utc, source, metric, meeting_date, target_range, value, status
        FROM observations
        WHERE status='ok' AND metric IN ({placeholders})
        ORDER BY timestamp_utc, id
    """
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(sql, connection, params=metrics)
    if frame.empty:
        return frame
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["timestamp_utc", "value"])


def _fedwatch_frame(observations: pd.DataFrame) -> pd.DataFrame:
    fed = observations.loc[observations["metric"].str.startswith("fedwatch_")].copy()
    if fed.empty:
        return pd.DataFrame()

    simple_metrics = {
        "fedwatch_cut_probability": "fedwatch_cut_pct",
        "fedwatch_hold_probability": "fedwatch_hold_pct",
        "fedwatch_hike_probability": "fedwatch_hike_pct",
        "fedwatch_expected_rate": "fedwatch_expected_rate_pct",
    }
    simple = fed.loc[fed["metric"].isin(simple_metrics)].copy()
    simple = simple.drop_duplicates(["timestamp_utc", "meeting_date", "metric"], keep="last")
    if simple.empty:
        wide = pd.DataFrame()
    else:
        wide = simple.pivot(index=["timestamp_utc", "meeting_date"], columns="metric", values="value")
        wide = wide.rename(columns=simple_metrics).reset_index()

    targets = fed.loc[fed["metric"].eq("fedwatch_target_probability")].copy()
    expected = pd.DataFrame(columns=["timestamp_utc", "meeting_date", "expected_rate_rebuilt"])
    if not targets.empty:
        targets["midpoint"] = targets["target_range"].map(_target_midpoint)
        targets = targets.dropna(subset=["midpoint"])
        targets = targets.drop_duplicates(
            ["timestamp_utc", "meeting_date", "target_range"], keep="last"
        )
        targets["weighted"] = targets["midpoint"] * targets["value"]
        expected = (
            targets.groupby(["timestamp_utc", "meeting_date"], as_index=False)
            .agg(weighted=("weighted", "sum"), probability_sum=("value", "sum"))
        )
        expected["expected_rate_rebuilt"] = np.where(
            expected["probability_sum"].between(99.0, 101.0),
            expected["weighted"] / expected["probability_sum"],
            np.nan,
        )
        expected = expected[["timestamp_utc", "meeting_date", "expected_rate_rebuilt"]]

    if wide.empty:
        wide = expected
    elif not expected.empty:
        wide = wide.merge(expected, on=["timestamp_utc", "meeting_date"], how="outer")
    if wide.empty:
        return wide
    if "fedwatch_expected_rate_pct" not in wide:
        wide["fedwatch_expected_rate_pct"] = np.nan
    wide["fedwatch_expected_rate_pct"] = wide["fedwatch_expected_rate_pct"].fillna(
        wide.get("expected_rate_rebuilt")
    )
    required = ["fedwatch_cut_pct", "fedwatch_hold_pct", "fedwatch_hike_pct"]
    for column in required:
        if column not in wide:
            wide[column] = np.nan
    wide["fedwatch_policy_bias_pct"] = wide["fedwatch_cut_pct"] - wide["fedwatch_hike_pct"]
    wide = wide.sort_values(["timestamp_utc", "meeting_date"])
    # The nearest meeting is the one written by the monitor. If duplicate
    # meeting rows exist at a timestamp, retain the lexicographically earliest
    # date, matching the monitor's nearest-meeting contract.
    return wide.drop_duplicates("timestamp_utc", keep="first").set_index("timestamp_utc")


def build_monitor_panel(observations: pd.DataFrame) -> pd.DataFrame:
    """Create an as-of macro state panel without treating polls as new facts."""

    if observations.empty:
        return pd.DataFrame()
    fed = _fedwatch_frame(observations)
    metric_names = {
        "us2y_yield": "us2y_yield_pct",
        "us10y_yield": "us10y_yield_pct",
        "us10y_2y_spread": "curve_10y_2y_pct",
        "dxy_index": "dxy_index",
    }
    series_frames: list[pd.DataFrame] = []
    if not fed.empty:
        series_frames.append(fed)
    for metric, column in metric_names.items():
        part = observations.loc[observations["metric"].eq(metric), ["timestamp_utc", "value"]].copy()
        if part.empty:
            continue
        part = part.drop_duplicates("timestamp_utc", keep="last").set_index("timestamp_utc")
        part = part.rename(columns={"value": column})
        series_frames.append(part)
    if not series_frames:
        return pd.DataFrame()
    timeline = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in series_frames))))
    panel = pd.DataFrame(index=timeline)
    for frame in series_frames:
        for column in frame.columns:
            if column in panel:
                continue
            panel[column] = frame[column].reindex(timeline).ffill()
    panel.index.name = "timestamp_utc"
    panel = panel.sort_index()
    # Repeated browser/monitor instances can produce dense identical rows.
    # Keep the union timeline here for accurate window lookup; event thinning
    # happens only after changes are calculated.
    return panel


def alpaca_loader(config: ResearchConfig, symbol: str, timeframe: str) -> AlpacaStockLoader:
    return AlpacaStockLoader(
        symbol=symbol,
        timeframe=timeframe,
        feed="sip",
        adjustment="split",
        data_dir=config.research_data_dir,
    )


def fetch_alpaca_bars(
    config: ResearchConfig,
    symbol: str,
    timeframe: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    loader = alpaca_loader(config, symbol, timeframe)
    frame = loader.fetch_remote(start, end, request_pause_seconds=0.15)
    if not frame.empty:
        loader.save_local_data(frame)
    return frame


def load_alpaca_bars(config: ResearchConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    frame = alpaca_loader(config, symbol, timeframe).load_local_data()
    if frame.empty:
        return frame
    frame.index = _utc_index(frame.index)
    frame.index.name = "timestamp_utc"
    return frame.sort_index()


def _normalize_okx_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    # CoinBacktest's legacy OKX SQLite loader stores timezone-naive Beijing
    # wall-clock timestamps. Do not reinterpret those values as UTC.
    if index.tz is None:
        index = index.tz_localize(BEIJING, ambiguous="NaT", nonexistent="shift_forward").tz_convert(UTC)
    else:
        index = index.tz_convert(UTC)
    result.index = index
    result.index.name = "timestamp_utc"
    return result.loc[~result.index.isna()].sort_index()


def load_existing_okx_bars(
    config: ResearchConfig,
    symbol: str,
    timeframe: str,
    start_utc: str | pd.Timestamp | None = None,
    end_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe, db_dir=config.existing_market_db.parent)
    start_bjt = None
    end_bjt = None
    if start_utc is not None:
        start_bjt = pd.Timestamp(start_utc).tz_convert(BEIJING).tz_localize(None)
    if end_utc is not None:
        end_bjt = pd.Timestamp(end_utc).tz_convert(BEIJING).tz_localize(None)
    frame = loader.load_local_data_range(start_bjt, end_bjt)
    return _normalize_okx_index(frame)


def fetch_okx_bars(
    config: ResearchConfig,
    symbol: str,
    timeframe: str,
    start_utc: str | pd.Timestamp,
    end_utc: str | pd.Timestamp,
) -> pd.DataFrame:
    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe, db_dir=config.research_data_dir)
    start = pd.Timestamp(start_utc).tz_convert(BEIJING).tz_localize(None)
    end = pd.Timestamp(end_utc).tz_convert(BEIJING).tz_localize(None)
    return _normalize_okx_index(loader.fetch_data_by_date_range(start, end))


def load_research_okx_bars(config: ResearchConfig, symbol: str, timeframe: str) -> pd.DataFrame:
    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe, db_dir=config.research_data_dir)
    return _normalize_okx_index(loader.load_local_data())


def yahoo_daily_path(config: ResearchConfig, symbol: str) -> Path:
    safe = _safe_symbol(symbol)
    return config.raw_dir / f"yahoo_{safe}_daily.csv"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("=", "_").replace("^", "_").replace("-", "_").replace(".", "_")


def fetch_yahoo_daily(
    config: ResearchConfig,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start, tz=UTC).timestamp()) if pd.Timestamp(start).tzinfo is None else int(pd.Timestamp(start).timestamp())
    end_value = pd.Timestamp(end)
    if end_value.tzinfo is None:
        end_value = end_value.tz_localize(UTC)
    end_ts = int((end_value + pd.Timedelta(days=1)).timestamp())
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0 CoinBacktest research"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json().get("chart", {})
    if payload.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {payload['error']}")
    result = (payload.get("result") or [None])[0]
    if not result:
        return pd.DataFrame()
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose", [])
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(result.get("timestamp", []), unit="s", utc=True),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        }
    )
    if adjusted and len(adjusted) == len(frame):
        frame["adjusted_close"] = adjusted
        # Keep the OHLC path internally consistent through ETF splits. Yahoo
        # supplies only adjusted close, so apply its close ratio to all four
        # price fields. This is essential for SOXL's split history.
        raw_close = pd.to_numeric(frame["close"], errors="coerce")
        factor = pd.to_numeric(frame["adjusted_close"], errors="coerce") / raw_close
        frame["raw_close"] = raw_close
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["timestamp_utc", "close"]).drop_duplicates("timestamp_utc", keep="last")
    path = yahoo_daily_path(config, symbol)
    frame.to_csv(path, index=False)
    return frame.set_index("timestamp_utc").sort_index()


def load_yahoo_daily(config: ResearchConfig, symbol: str) -> pd.DataFrame:
    path = yahoo_daily_path(config, symbol)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp_utc"]).set_index("timestamp_utc").sort_index()


def yahoo_intraday_path(
    config: ResearchConfig,
    symbol: str,
    *,
    interval: str = "5m",
) -> Path:
    return config.raw_dir / f"yahoo_{_safe_symbol(symbol)}_{interval}.csv"


def parse_yahoo_chart_payload(payload: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    """Parse Yahoo's public chart response without relying on network in tests."""

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo chart error for {symbol}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result:
        return pd.DataFrame()
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True, errors="coerce"),
            "open": quote.get("open") or [None] * len(timestamps),
            "high": quote.get("high") or [None] * len(timestamps),
            "low": quote.get("low") or [None] * len(timestamps),
            "close": quote.get("close") or [None] * len(timestamps),
            "volume": quote.get("volume") or [None] * len(timestamps),
        }
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = symbol
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def fetch_yahoo_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    range_value: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "range": range_value,
            "interval": interval,
            "events": "history",
            "includePrePost": "true",
        },
        headers={"User-Agent": "Mozilla/5.0 CoinBacktest research"},
        timeout=45,
    )
    response.raise_for_status()
    frame = parse_yahoo_chart_payload(response.json(), symbol)
    if not frame.empty:
        frame.reset_index().to_csv(yahoo_intraday_path(config, symbol, interval=interval), index=False)
    return frame


def load_yahoo_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    interval: str = "5m",
) -> pd.DataFrame:
    path = yahoo_intraday_path(config, symbol, interval=interval)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def cnbc_yield_path(
    config: ResearchConfig,
    symbol: str,
    *,
    time_range: str = "5D",
) -> Path:
    return config.raw_dir / f"cnbc_{symbol.lower()}_{time_range.lower()}.csv"


def parse_cnbc_chart_payload(payload: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    """Parse CNBC/Tradeweb Treasury yield bars from a fixed response payload."""

    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError(f"CNBC chart error for {symbol}: {errors[0]}")
    chart = ((payload.get("data") or {}).get("chartData") or {})
    bars = chart.get("priceBars") or []
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame(bars)
    millis = pd.to_numeric(frame.get("tradeTimeinMills"), errors="coerce")
    frame["timestamp_utc"] = pd.to_datetime(millis, unit="ms", utc=True, errors="coerce")
    if frame["timestamp_utc"].isna().all() and "tradeTime" in frame:
        frame["timestamp_utc"] = pd.to_datetime(
            frame["tradeTime"], format="%Y%m%d%H%M%S", utc=True, errors="coerce"
        )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["symbol"] = symbol
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def fetch_cnbc_yield_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    time_range: str = "5D",
) -> pd.DataFrame:
    variables = json.dumps({"symbol": symbol, "timeRange": time_range}, separators=(",", ":"))
    extensions = json.dumps(
        {"persistedQuery": {"version": 1, "sha256Hash": CNBC_CHART_QUERY_HASH}},
        separators=(",", ":"),
    )
    response = requests.get(
        CNBC_CHART_ENDPOINT,
        params={
            "operationName": "getQuoteChartData",
            "variables": variables,
            "extensions": extensions,
        },
        headers={
            "User-Agent": "Mozilla/5.0 CoinBacktest research",
            "Referer": f"https://www.cnbc.com/quotes/{symbol}",
        },
        timeout=45,
    )
    response.raise_for_status()
    frame = parse_cnbc_chart_payload(response.json(), symbol)
    if not frame.empty:
        frame.reset_index().to_csv(cnbc_yield_path(config, symbol, time_range=time_range), index=False)
    return frame


def load_cnbc_yield_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    time_range: str = "5D",
) -> pd.DataFrame:
    path = cnbc_yield_path(config, symbol, time_range=time_range)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def okx_intraday_path(config: ResearchConfig, symbol: str, *, bar: str = "5m") -> Path:
    return config.raw_dir / f"okx_{_safe_symbol(symbol).lower()}_{bar}.csv"


def parse_okx_candles(rows: Iterable[Iterable[object]], symbol: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        values = list(row)
        if len(values) < 5:
            continue
        records.append(
            {
                "timestamp_utc": pd.to_datetime(
                    pd.to_numeric(values[0], errors="coerce"),
                    unit="ms",
                    utc=True,
                    errors="coerce",
                ),
                "open": values[1],
                "high": values[2],
                "low": values[3],
                "close": values[4],
                "volume": values[5] if len(values) > 5 else None,
                "confirmed": values[8] if len(values) > 8 else None,
                "symbol": symbol,
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def fetch_okx_public_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    start_utc: str | pd.Timestamp,
    end_utc: str | pd.Timestamp,
    bar: str = "5m",
    request_pause_seconds: float = 0.04,
) -> pd.DataFrame:
    """Download public OKX candles backwards with bounded pagination."""

    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    start = start.tz_localize(UTC) if start.tzinfo is None else start.tz_convert(UTC)
    end = end.tz_localize(UTC) if end.tzinfo is None else end.tz_convert(UTC)
    cursor: int | None = int(end.timestamp() * 1000)
    pages: list[pd.DataFrame] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 CoinBacktest research"})
    last_oldest: int | None = None
    for _ in range(500):
        params: dict[str, object] = {"instId": symbol, "bar": bar, "limit": "300"}
        if cursor is not None:
            params["after"] = str(cursor)
        response = session.get(OKX_HISTORY_ENDPOINT, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX history error for {symbol}: {payload.get('msg') or payload}")
        rows = payload.get("data") or []
        if not rows:
            break
        frame = parse_okx_candles(rows, symbol)
        if frame.empty:
            break
        pages.append(frame)
        oldest_ms = int(frame.index.min().timestamp() * 1000)
        if oldest_ms <= int(start.timestamp() * 1000):
            break
        if oldest_ms == last_oldest:
            break
        last_oldest = oldest_ms
        cursor = oldest_ms - 1
        if request_pause_seconds > 0:
            time.sleep(request_pause_seconds)
    if not pages:
        return pd.DataFrame()
    result = pd.concat(pages).sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    result = result.loc[(result.index >= start) & (result.index <= end)]
    if not result.empty:
        result.reset_index().to_csv(okx_intraday_path(config, symbol, bar=bar), index=False)
    return result


def load_okx_public_intraday(
    config: ResearchConfig,
    symbol: str,
    *,
    bar: str = "5m",
) -> pd.DataFrame:
    path = okx_intraday_path(config, symbol, bar=bar)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["timestamp_utc", "close"])
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
    )


def load_fred_yields(config: ResearchConfig) -> pd.DataFrame:
    if not config.fred_yields_csv.exists():
        return pd.DataFrame()
    frame = pd.read_csv(config.fred_yields_csv)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("DGS2", "DGS10", "DGS2_change_bp", "DGS10_change_bp"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date").sort_index()


def coverage_for_frame(source: str, dataset: str, frame: pd.DataFrame, notes: str = "") -> Coverage:
    if frame.empty:
        return Coverage(source, dataset, 0, None, None, notes)
    index = pd.DatetimeIndex(frame.index)
    return Coverage(source, dataset, len(frame), str(index.min()), str(index.max()), notes)


def write_inventory(config: ResearchConfig, coverages: Iterable[Coverage]) -> Path:
    config.ensure_directories()
    path = config.output_dir / "data_inventory.json"
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "datasets": [asdict(item) for item in coverages],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
