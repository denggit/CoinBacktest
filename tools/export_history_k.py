#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 3/1/26 8:06 PM
@File       : export_history_k.py
@Description:
"""
import argparse
import os
import sys
from zoneinfo import ZoneInfo

import pandas as pd

# 添加项目根目录到 Python 路径
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_feed.okx_loader import OKXDataLoader

DEFAULT_SYMBOL = "ETH-USDT-SWAP"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-07-30"
DEFAULT_TIMEZONE = "Asia/Shanghai"


def parse_minute_timestamp(value: str, *, end_of_day: bool = False) -> pd.Timestamp:
    """解析到分钟；纯日期的结束时间按当天最后一根 1m K 线处理。"""
    ts = pd.Timestamp(value)
    if ts.second != 0 or ts.microsecond != 0 or ts.nanosecond != 0:
        raise argparse.ArgumentTypeError("时间请精确到分钟，例如: 2026-05-18 09:30")
    if end_of_day and len(value.strip()) == 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return ts


def parse_start_timestamp(value: str) -> pd.Timestamp:
    return parse_minute_timestamp(value)


def parse_end_timestamp(value: str) -> pd.Timestamp:
    return parse_minute_timestamp(value, end_of_day=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出指定分钟区间的 OKX 历史 K 线")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"交易对，默认 {DEFAULT_SYMBOL}")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help=f"K线周期，默认 {DEFAULT_TIMEFRAME}")
    parser.add_argument(
        "--start",
        type=parse_start_timestamp,
        default=parse_start_timestamp(DEFAULT_START),
        help=f"开始时间，精确到分钟；纯日期按 00:00，默认 '{DEFAULT_START}'",
    )
    parser.add_argument(
        "--end",
        type=parse_end_timestamp,
        default=parse_end_timestamp(DEFAULT_END),
        help=f"结束时间，精确到分钟；纯日期按当天 23:59，默认 '{DEFAULT_END}'",
    )
    parser.add_argument("--output", help="输出 CSV 文件路径；默认写入 data/history_k 并带上分钟区间")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help=f"K线 timestamp 所属时区，默认 {DEFAULT_TIMEZONE}")
    return parser


def format_filename_time(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d_%H%M")


def _isoformat_series(values: pd.Series) -> list[str]:
    return [ts.isoformat() for ts in values]


def normalize_okx_ts_ms_to_epoch_sec(ts_ms):
    ts = float(ts_ms)
    if ts > 10_000_000_000:
        return ts / 1000.0
    return ts


def _timestamp_ms_from_local_timestamp(values: pd.Series, timezone_name: str) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.dt.tz is None:
        local_ts = parsed.dt.tz_localize(timezone_name)
    else:
        local_ts = parsed.dt.tz_convert(timezone_name)
    utc_ts = local_ts.dt.tz_convert("UTC")
    timestamp_ms = (utc_ts.astype("int64") // 1_000_000).astype("int64")
    return timestamp_ms, utc_ts


def prepare_export_dataframe(df: pd.DataFrame, timezone_name: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    ZoneInfo(timezone_name)
    if df.empty:
        return df.reset_index()

    out = df.copy().reset_index()
    if "timestamp" not in out.columns and "index" in out.columns:
        out.rename(columns={"index": "timestamp"}, inplace=True)
    if "timestamp" not in out.columns:
        raise ValueError("Kline DataFrame missing timestamp index/column")

    if "timestamp_ms" in out.columns:
        timestamp_ms = pd.to_numeric(out["timestamp_ms"], errors="raise").astype("int64")
        if (timestamp_ms < 10_000_000_000).any() and "timestamp_epoch_sec" in out.columns:
            epoch_sec = pd.to_numeric(out["timestamp_epoch_sec"], errors="raise")
            if (epoch_sec >= 1_500_000_000).all():
                timestamp_ms = (epoch_sec.astype("float64") * 1000).round().astype("int64")
        if (timestamp_ms < 10_000_000_000).any():
            timestamp_ms, utc_ts = _timestamp_ms_from_local_timestamp(out["timestamp"], timezone_name)
        else:
            utc_ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
        out["timestamp_ms"] = timestamp_ms
    elif "ts_ms" in out.columns:
        timestamp_ms = pd.to_numeric(out["ts_ms"], errors="raise").astype("int64")
        if (timestamp_ms < 10_000_000_000).any():
            timestamp_ms, utc_ts = _timestamp_ms_from_local_timestamp(out["timestamp"], timezone_name)
        else:
            utc_ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
        out["timestamp_ms"] = timestamp_ms
    else:
        timestamp_ms, utc_ts = _timestamp_ms_from_local_timestamp(out["timestamp"], timezone_name)
        out["timestamp_ms"] = timestamp_ms

    local_ts = utc_ts.dt.tz_convert(timezone_name)
    out["timestamp"] = local_ts.dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["timestamp_epoch_sec"] = [normalize_okx_ts_ms_to_epoch_sec(value) for value in out["timestamp_ms"]]
    out["timestamp_utc"] = _isoformat_series(utc_ts)
    out["timestamp_local"] = _isoformat_series(local_ts)
    out["timezone"] = timezone_name

    preferred = ["timestamp", "timestamp_epoch_sec", "timestamp_ms", "timestamp_utc", "timestamp_local", "timezone"]
    return out[preferred + [col for col in out.columns if col not in preferred]]


# 获取项目根目录下的 data/reports 目录
current_file = os.path.abspath(__file__)
# 向上推三层：report.py -> utils -> src -> 根目录 (Momentum1.66)
project_root = os.path.dirname(os.path.dirname(current_file))
# 使用项目根目录下的 data/reports 目录
data_dir = os.path.join(project_root, 'data', 'history_k')
os.makedirs(data_dir, exist_ok=True)

if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        ZoneInfo(args.timezone)
    except Exception as exc:
        raise SystemExit(f"无效 timezone: {args.timezone}") from exc
    if args.end < args.start:
        raise SystemExit("结束时间不能早于开始时间")

    symbol = args.symbol
    timeframe = args.timeframe
    start_time = args.start
    end_time = args.end

    data_loader = OKXDataLoader(symbol, timeframe)
    df = data_loader.fetch_data_by_date_range(start_time, end_time)
    export_df = prepare_export_dataframe(df, timezone_name=args.timezone)
    output_file = args.output or os.path.join(
        data_dir,
        f"{symbol}_{timeframe}_{format_filename_time(start_time)}_{format_filename_time(end_time)}.csv"
    )
    export_df.to_csv(output_file, index=False)
