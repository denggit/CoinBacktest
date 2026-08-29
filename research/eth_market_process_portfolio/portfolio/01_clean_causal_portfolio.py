#!/usr/bin/env python
"""Run the frozen ETH clean causal portfolio research and write audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_market_process_portfolio.portfolio.clean_causal import (  # noqa: E402
    BARS_PER_DAY,
    PortfolioConfig,
    extract_position_episodes,
    period_summary,
    resample_to_4h,
    scenario_configs,
    shock_survival_table,
    simulate_portfolio,
    summarize_equity,
    validate_ohlcv,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "clean_causal_v1" / "results"
NEIGHBOURHOODS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("faster_5_20_60", (5, 20, 60)),
    ("frozen_7_30_90", (7, 30, 90)),
    ("slower_10_40_120", (10, 40, 120)),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2022-01-01 00:00:00")
    parser.add_argument("--end-date", default="2026-08-15 23:59:59")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _load_local_minute_bars(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    loader = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m")
    local = loader.load_local_data()
    if local.empty:
        raise RuntimeError("local ETH-USDT-SWAP 1m data is unavailable; network downloads are forbidden in this study")
    local = local.sort_index(kind="stable")
    local = local[~local.index.duplicated(keep="last")]
    selected = local[(local.index >= start) & (local.index <= end)].copy()
    if selected.empty or selected.index.min() > start or selected.index.max() < end.floor("min"):
        raise RuntimeError(
            f"local 1m coverage does not span requested window: {selected.index.min()} -> {selected.index.max()}"
        )
    for column in ("open", "high", "low", "close", "volume"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    return selected.dropna(subset=["open", "high", "low", "close", "volume"])


def _buy_hold_curve(bars: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    core = bars[(bars.index >= pd.Timestamp(cfg.start)) & (bars.index <= pd.Timestamp(cfg.end))].copy()
    returns = core["open"].shift(-1) / core["open"] - 1.0
    returns = returns.iloc[:-1].copy()
    if returns.empty:
        return pd.DataFrame()
    costs = pd.Series(cfg.annual_carry_drag / (365 * BARS_PER_DAY), index=returns.index)
    costs.iloc[0] += cfg.one_way_cost
    net = returns - costs
    equity = (1.0 + net).cumprod()
    peak = equity.cummax()
    return pd.DataFrame(
        {
            "net_return": net,
            "equity": equity,
            "drawdown": equity / peak - 1.0,
        },
        index=returns.index,
    )


def _summary_row(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {"scenario": name, **summarize_equity(frame)}


def _benchmark_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    elapsed = max((frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
    final = float(frame["equity"].iloc[-1])
    monthly = (1.0 + frame["net_return"]).groupby(frame.index.to_period("M")).prod() - 1.0
    return {
        "scenario": "buy_hold_1x_costed",
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "bars": int(len(frame)),
        "total_return": final - 1.0,
        "final_equity": final,
        "cagr": final ** (1 / elapsed) - 1.0,
        "annual_volatility": float(frame["net_return"].std(ddof=0) * np.sqrt(365 * BARS_PER_DAY)),
        "sharpe_zero_rf": float(
            frame["net_return"].mean() / frame["net_return"].std(ddof=0) * np.sqrt(365 * BARS_PER_DAY)
        ),
        "calmar": (final ** (1 / elapsed) - 1.0) / abs(float(frame["drawdown"].min())),
        "max_drawdown": float(frame["drawdown"].min()),
        "positive_month_rate": float((monthly > 0).mean()),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
        "max_abs_exposure": 1.0,
        "liquidation_events": 0,
    }


def _top_day_removal(frame: pd.DataFrame, count: int = 10) -> dict[str, object]:
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    removed = daily.drop(daily.nlargest(min(count, len(daily))).index)
    return {
        "removed_best_days": count,
        "remaining_total_return": float((1.0 + removed).prod() - 1.0),
        "original_total_return": float((1.0 + daily).prod() - 1.0),
        "largest_removed_day": float(daily.nlargest(1).iloc[0]),
    }


def _build_equity_dataset(base: pd.DataFrame, benchmark: pd.DataFrame) -> list[dict[str, object]]:
    strategy = base.resample("1D").last()[["equity", "drawdown", "net_exposure"]].dropna()
    buy_hold = benchmark.resample("1D").last()[["equity", "drawdown"]].dropna()
    rows: list[dict[str, object]] = []
    for timestamp, row in strategy.iterrows():
        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "series": "Clean causal portfolio",
                "equity": float(row["equity"]),
                "drawdown": float(row["drawdown"]),
                "exposure": float(row["net_exposure"]),
            }
        )
    for timestamp, row in buy_hold.iterrows():
        rows.append(
            {
                "date": timestamp.strftime("%Y-%m-%d"),
                "series": "ETH buy & hold (1x, costed)",
                "equity": float(row["equity"]),
                "drawdown": float(row["drawdown"]),
                "exposure": 1.0,
            }
        )
    return rows


def _source() -> dict[str, object]:
    return {
        "id": "eth_ohlcv_sqlite",
        "label": "Local OKX ETH-USDT-SWAP 1m OHLCV",
        "path": "data/crypto_history.db",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "description": "Local closed 1m candles loaded through src.data_feed.OKXDataLoader and causally resampled to 4H.",
            "sql": (
                "SELECT timestamp, open, high, low, close, volume "
                "FROM ETH_USDT_SWAP_1m "
                "WHERE timestamp >= '2021-08-01 00:00:00' "
                "AND timestamp <= '2026-08-15 23:59:59' ORDER BY timestamp"
            ),
            "tables_used": ["ETH_USDT_SWAP_1m"],
            "filters": [
                "Research evaluation: 2022-01-01 through 2026-08-15 23:59:59 Asia/Shanghai local-time bars",
                "Warmup starts 2021-08-01",
                "Duplicate timestamps keep the last local record",
            ],
            "metric_definitions": [
                "Net return = target exposure × next-open price return − turnover cost − conservative carry drag",
                "Signals use completed 4H close data and execute no earlier than the following 4H open",
                "Opening and closing each cost 0.05% of changed sleeve notional; gross exposure and margin retain both hedge-mode sides",
            ],
        },
    }


def _artifact(
    *,
    cfg: PortfolioConfig,
    generated_at: str,
    base_summary: dict[str, object],
    benchmark_summary: dict[str, object],
    scenario_rows: list[dict[str, object]],
    yearly_rows: list[dict[str, object]],
    equity_rows: list[dict[str, object]],
    shock_rows: list[dict[str, object]],
    data_quality_rows: list[dict[str, object]],
    neighbourhood_rows: list[dict[str, object]],
    top_day: dict[str, object],
) -> dict[str, object]:
    source = _source()
    title = "ETH Clean Causal Portfolio V1 — 2022–2026"
    total_return = float(base_summary["total_return"])
    cagr = float(base_summary["cagr"])
    max_dd = float(base_summary["max_drawdown"])
    liq_events = int(base_summary["liquidation_events"])
    stress_positive = all(float(row.get("total_return", -1.0)) > 0 for row in scenario_rows if row["scenario"] != "no_drawdown_throttle")
    summary_body = (
        "## 技术摘要\n\n"
        f"- **固定、无训练的三周期趋势组合在样本内实现总收益 {total_return:.1%}、CAGR {cagr:.1%}、最大回撤 {max_dd:.1%}。** "
        "三个袖套独立持仓，允许长周期做多与短周期做空同时存在；所有信号均在 4H 收盘后形成，并从后续开盘执行。\n"
        f"- **回测清算事件为 {liq_events}，策略实际名义敞口上限为 {cfg.strategy_notional_cap:.2f}×。** "
        f"15× 是交易所配置上限，不是策略目标敞口；50% 瞬时逆向冲击仍保留正的假设维持保证金余量。\n"
        f"- **固定压力场景{'全部保持正收益' if stress_positive else '并非全部保持正收益'}。** "
        "成本、延迟和 carry 已直接计入资金曲线；结果不使用订单流、资金费率或 2026 年短窗口字段来补全历史。\n"
        "- **这不是‘未来绝不爆仓’的承诺。** 它证明的是本地历史路径和声明的冲击集合内未触发清算；交易所规则变化、跳空超过压力范围、系统故障和流动性枯竭仍是残余风险。"
    )
    source_id = source["id"]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "A causal, low-complexity ETH long/short cross-margin portfolio research report.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "total_return",
                "description": "Compounded account return after trading cost and conservative carry drag.",
                "dataset": "headline",
                "sourceId": source_id,
                "metrics": [
                    {"label": "Total return", "field": "total_return", "format": "percent"},
                    {"label": "CAGR", "field": "cagr", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "max_drawdown",
                "description": "Worst peak-to-trough account decline on 4H marked equity.",
                "dataset": "headline",
                "sourceId": source_id,
                "metrics": [{"label": "Max drawdown", "field": "max_drawdown", "format": "percent"}],
            },
            {
                "id": "exposure",
                "description": "Observed gross notional exposure across both hedge-mode sides relative to current equity.",
                "dataset": "headline",
                "sourceId": source_id,
                "metrics": [
                    {"label": "Max gross exposure", "field": "max_gross_exposure", "format": "number"},
                    {"label": "Exchange cap", "field": "exchange_leverage_cap", "format": "number"},
                ],
            },
            {
                "id": "liquidation",
                "description": "Historical liquidation-rule breaches under the stated maintenance-margin model.",
                "dataset": "headline",
                "sourceId": source_id,
                "metrics": [{"label": "Liquidation events", "field": "liquidation_events", "format": "number"}],
            },
        ],
        "charts": [
            {
                "id": "equity_curve",
                "title": "Account equity and costed ETH benchmark",
                "subtitle": "Daily snapshots, 2022-01-01 through 2026-08-15; both series start at 1.0",
                "type": "line",
                "dataset": "equity_daily",
                "sourceId": source_id,
                "encodings": {
                    "x": {"field": "date", "type": "temporal", "label": "Date"},
                    "y": {"field": "equity", "type": "quantitative", "label": "Equity multiple"},
                    "color": {"field": "series", "type": "nominal", "label": "Series"},
                },
                "yAxisTitle": "Equity multiple",
                "valueFormat": "number",
                "layout": "full",
            },
            {
                "id": "yearly_return",
                "title": "Calendar-year portfolio returns",
                "subtitle": "2026 is partial through August 15; values include trading and carry costs",
                "type": "bar",
                "dataset": "yearly",
                "sourceId": source_id,
                "encodings": {
                    "x": {"field": "period", "type": "ordinal", "label": "Calendar year"},
                    "y": {"field": "return", "type": "quantitative", "label": "Return", "format": "percent"},
                },
                "yAxisTitle": "Return",
                "valueFormat": "percent",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "scenario_table",
                "title": "Predeclared execution and cost stress",
                "dataset": "scenarios",
                "sourceId": source_id,
                "defaultSort": {"field": "scenario", "direction": "asc"},
                "columns": [
                    {"field": "scenario", "label": "Scenario", "type": "text"},
                    {"field": "total_return", "label": "Total return", "format": "percent"},
                    {"field": "cagr", "label": "CAGR", "format": "percent", "movement": True},
                    {"field": "max_drawdown", "label": "Max drawdown", "format": "percent", "movement": True},
                    {"field": "positive_month_rate", "label": "Positive months", "format": "percent"},
                    {"field": "max_gross_exposure", "label": "Max gross exposure", "format": "number"},
                    {"field": "long_short_overlap_rate", "label": "Both sides active", "format": "percent"},
                    {"field": "liquidation_events", "label": "Liquidations", "format": "number"},
                ],
            },
            {
                "id": "shock_table",
                "title": "Instantaneous adverse-move survival",
                "dataset": "shock",
                "sourceId": source_id,
                "defaultSort": {"field": "absolute_exposure", "direction": "asc"},
                "columns": [
                    {"field": "adverse_instantaneous_move", "label": "Adverse move", "format": "percent"},
                    {"field": "absolute_exposure", "label": "Exposure", "format": "number"},
                    {"field": "equity_ratio_after_shock", "label": "Equity left", "format": "percent"},
                    {"field": "maintenance_required", "label": "Maintenance", "format": "percent"},
                    {"field": "maintenance_headroom", "label": "Headroom", "format": "percent"},
                    {"field": "survives_assumed_liquidation_rule", "label": "Survives", "type": "text"},
                ],
            },
            {
                "id": "neighbourhood_table",
                "title": "Frozen momentum-horizon neighbourhood",
                "dataset": "neighbourhood",
                "sourceId": source_id,
                "defaultSort": {"field": "variant", "direction": "asc"},
                "columns": [
                    {"field": "variant", "label": "Variant", "type": "text"},
                    {"field": "total_return", "label": "Total return", "format": "percent"},
                    {"field": "cagr", "label": "CAGR", "format": "percent", "movement": True},
                    {"field": "max_drawdown", "label": "Max drawdown", "format": "percent", "movement": True},
                    {"field": "positive_month_rate", "label": "Positive months", "format": "percent"},
                    {"field": "liquidation_events", "label": "Liquidations", "format": "number"},
                ],
            },
            {
                "id": "data_quality_table",
                "title": "Input data quality gate",
                "dataset": "data_quality",
                "sourceId": source_id,
                "defaultSort": {"field": "grain", "direction": "asc"},
                "columns": [
                    {"field": "grain", "label": "Grain", "type": "text"},
                    {"field": "rows", "label": "Rows", "format": "number"},
                    {"field": "start", "label": "Start", "type": "text"},
                    {"field": "end", "label": "End", "type": "text"},
                    {"field": "duplicate_timestamps", "label": "Duplicates", "format": "number"},
                    {"field": "missing_timestamp_rate", "label": "Missing rate", "format": "percent"},
                    {"field": "invalid_ohlc_rows", "label": "Invalid OHLC", "format": "number"},
                    {"field": "ready", "label": "Ready", "type": "text"},
                ],
            },
        ],
        "sources": [source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {"id": "technical_summary", "type": "markdown", "sourceId": source_id, "body": summary_body},
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["total_return", "max_drawdown", "exposure", "liquidation"]},
            {
                "id": "equity_finding",
                "type": "markdown",
                "sourceId": source_id,
                "body": "## 低频趋势而非短期套利承担主要收益任务\n\n三条趋势袖套只使用价格历史，方向投票后按滞后实现波动率缩放。下图比较净资金曲线与同成本口径的 1× ETH 持有；重点不是追求最高终值，而是观察跨牛熊周期的连续性和回撤。",
            },
            {"id": "equity_chart", "type": "chart", "chartId": "equity_curve", "layout": "full"},
            {
                "id": "yearly_finding",
                "type": "markdown",
                "sourceId": source_id,
                "body": "## 跨年度结果检验方向对称性\n\n逐年表现用于识别结果是否集中于单一行情。2026 是截至 8 月 15 日的部分年度，不能与完整年度等量比较；任何单年亏损都应被视为策略现实组成，而不是继续调参的理由。",
            },
            {"id": "yearly_chart", "type": "chart", "chartId": "yearly_return", "layout": "full"},
            {
                "id": "risk_finding",
                "type": "markdown",
                "sourceId": source_id,
                "body": "## 15× 是账户设置上限，1.5× 总敞口才是模型风险上限\n\n策略使用全账户交叉保证金和 hedge mode，允许多空袖套同时存在。手续费和维持保证金按两边总名义敞口计算，收益按各袖套贡献相加，因此不会用净额掩盖杠杆。清算审计以 0.5% 维持保证金率近似；50% 瞬时逆向冲击是模型化压力，不代表真实市场的最大可能跳空。",
            },
            {"id": "shock", "type": "table", "tableId": "shock_table", "layout": "full"},
            {
                "id": "robustness_finding",
                "type": "markdown",
                "sourceId": source_id,
                "body": (
                    "## 压力与邻域结果决定是否值得纸面交易\n\n"
                    f"固定压力场景覆盖成本、执行延迟和持仓成本；动量邻域预先固定为 5/20/60、7/30/90、10/40/120 日。"
                    f"剔除最好的 {int(top_day['removed_best_days'])} 个交易日后，剩余复合收益为 {float(top_day['remaining_total_return']):.1%}。"
                    "邻域和剔除测试用于识别单点参数或极少数日期依赖，不从中挑选收益最高的配置。"
                ),
            },
            {"id": "scenario", "type": "table", "tableId": "scenario_table", "layout": "full"},
            {"id": "neighbourhood", "type": "table", "tableId": "neighbourhood_table", "layout": "full"},
            {
                "id": "scope_data",
                "type": "markdown",
                "body": "## 数据边界和指标口径\n\n主证据只使用本地 OKX `ETH-USDT-SWAP` 1m OHLCV，经左闭右开规则聚合为 4H。研究窗口为 2022-01-01 至 2026-08-15（UTC+8 本地时间），2021-08-01 起的数据仅作预热。资金曲线按每个 4H 开盘到下一开盘标记；每个袖套开仓 0.05%、平仓 0.05%，按两边名义变动分别扣费，并对多空总敞口额外扣除年化 5% 保守 carry。",
            },
            {"id": "data_quality", "type": "table", "tableId": "data_quality_table", "layout": "full"},
            {
                "id": "method",
                "type": "markdown",
                "body": "## 模型规范与无未来函数约束\n\n每个 4H 收盘分别计算 7、30、90 日对数收益方向，各周期拥有独立的多空袖套；30 日已实现波动率按等风险预算分配总名义敞口，每日只再平衡一次。信号在收盘后才可用，目标仓位至少推迟一个 4H 开盘。回撤达到 10%/15% 后，后续开盘风险速度降至 50%/25%。模型不训练、不搜索阈值、不读取未来资金费率或订单流，也不使用同根 K 线收盘成交。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## 限制、不确定性与实盘结论\n\n历史回测无法证明未来绝不爆仓，也不能覆盖交易所 ADL、标记价格偏离、API/网络中断、无法成交、费率制度变化或超过 50% 的瞬时逆向跳空。资金费率历史只覆盖短窗口，因此基础回测改用保守统一 carry，而没有把短窗口实际费率外推到 2022 年。建议状态为 `paper_trade_only`：先做至少 8–12 周逐笔影子执行，核对目标仓位、成交价、资金费率、标记价格和清算距离，未通过前不要用真实 15× 杠杆。",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## 推荐下一步\n\n1. 按 `paper_trade_only` 部署，分别发送 7/30/90 日袖套的 long/short 目标，不把它们提前净额化，也不直接复用回测成交价。\n2. 实盘风控硬编码：多空总敞口不超过 1.5×、交易所杠杆配置不超过 15×、维持保证金余量跌破内部阈值立即按比例降两边仓位。\n3. 每日保存信号生成时间、下单时间、成交偏差、funding 和标记价格；任何字段晚于决策时间即阻断。\n4. 8–12 周后只做执行偏差审计，不因短期盈亏重新拟合三个趋势周期。",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## 后续问题\n\n- OKX 当前账户等级的真实维持保证金阶梯和强平手续费是多少？\n- 影子执行的单边滑点是否持续低于 2 bps，还是应提高基础成本？\n- 极端行情下，目标仓位从 1.5× 降至 0.75× 的实际延迟是否仍能满足内部清算余量？",
            },
        ],
    }
    headline = {
        **base_summary,
        "exchange_leverage_cap": cfg.exchange_leverage_cap,
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [headline],
                "equity_daily": equity_rows,
                "yearly": yearly_rows,
                "scenarios": scenario_rows,
                "shock": shock_rows,
                "data_quality": data_quality_rows,
                "neighbourhood": neighbourhood_rows,
            },
        },
        "sources": [source],
        "package_info": {
            "root": "research/eth_market_process_portfolio/portfolio/clean_causal_v1/results",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json",
        },
    }


def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = PortfolioConfig(start=args.start_date, end=args.end_date)
    cfg.validate()
    warmup_start = pd.Timestamp(cfg.start) - pd.Timedelta(days=max(150, max(cfg.momentum_days) + 40))
    end = pd.Timestamp(cfg.end)
    print(f"[load] local 1m {warmup_start} -> {end}", flush=True)
    minute = _load_local_minute_bars(warmup_start, end)
    minute_core = minute[(minute.index >= pd.Timestamp(cfg.start)) & (minute.index <= end)]
    minute_quality = validate_ohlcv(minute_core, expected_frequency="1min")
    minute_quality["grain"] = "1m source"
    print(f"[data] minute_rows={len(minute):,} core_rows={len(minute_core):,}", flush=True)
    bars = resample_to_4h(minute)
    core_bars = bars[(bars.index >= pd.Timestamp(cfg.start)) & (bars.index <= end)]
    bar_quality = validate_ohlcv(core_bars, expected_frequency="4h")
    bar_quality["grain"] = "4H research"
    bar_quality["incomplete_source_bars"] = int((core_bars["source_minutes"] < 228).sum())
    if minute_quality["missing_timestamp_rate"] > 0.01:
        raise RuntimeError(f"1m missing timestamp rate exceeds 1%: {minute_quality['missing_timestamp_rate']:.4%}")
    print(
        f"[quality] minute_missing={minute_quality['missing_timestamp_rate']:.4%} "
        f"4h_missing={bar_quality['missing_timestamp_rate']:.4%}",
        flush=True,
    )

    scenario_frames: dict[str, pd.DataFrame] = {}
    scenario_rows: list[dict[str, object]] = []
    for name, scenario_cfg in scenario_configs(cfg):
        print(f"[scenario] {name}", flush=True)
        replay = simulate_portfolio(bars, scenario_cfg)
        replay = replay[(replay.index >= pd.Timestamp(cfg.start)) & (replay.index <= end)].copy()
        scenario_frames[name] = replay
        scenario_rows.append(_summary_row(name, replay))
    base = scenario_frames["base"]
    base_summary = summarize_equity(base)
    benchmark = _buy_hold_curve(bars, cfg)
    benchmark_summary = _benchmark_summary(benchmark)

    neighbourhood_rows: list[dict[str, object]] = []
    for name, horizons in NEIGHBOURHOODS:
        print(f"[neighbourhood] {name}", flush=True)
        replay = simulate_portfolio(bars, replace(cfg, momentum_days=horizons))
        replay = replay[(replay.index >= pd.Timestamp(cfg.start)) & (replay.index <= end)]
        neighbourhood_rows.append({"variant": name, **summarize_equity(replay)})

    yearly = period_summary(base, "Y")
    monthly = period_summary(base, "M")
    episodes = extract_position_episodes(base)
    top_day = _top_day_removal(base)
    shock = shock_survival_table([1.0, 1.25, cfg.strategy_notional_cap, cfg.exchange_leverage_cap], cfg)
    leverage_audit = pd.DataFrame(
        [
            {
                "exchange_leverage_cap": cfg.exchange_leverage_cap,
                "strategy_notional_cap": cfg.strategy_notional_cap,
                "observed_max_net_exposure": float(base["net_exposure"].abs().max()),
                "observed_max_gross_exposure": float(base["gross_exposure"].max()),
                "long_short_overlap_rate": float(base["long_short_overlap"].mean()),
                "max_initial_margin_fraction_at_15x": float(base["gross_exposure"].max() / cfg.exchange_leverage_cap),
                "maintenance_margin_rate_assumption": cfg.maintenance_margin_rate,
                "min_observed_maintenance_headroom": float(base["maintenance_headroom"].min()),
                "observed_liquidation_events": int(base["liquidated"].sum()),
                "decision": "paper_trade_only",
            }
        ]
    )

    summary = pd.DataFrame([{"scenario": "base", **base_summary}, benchmark_summary])
    pd.DataFrame(scenario_rows).to_csv(output / "scenario_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    yearly.to_csv(output / "yearly.csv", index=False)
    monthly.to_csv(output / "monthly.csv", index=False)
    episodes.to_csv(output / "position_episodes.csv", index=False)
    base.reset_index().to_csv(output / "equity_4h.csv", index=False)
    pd.DataFrame(neighbourhood_rows).to_csv(output / "parameter_neighbourhood.csv", index=False)
    shock.to_csv(output / "shock_survival.csv", index=False)
    leverage_audit.to_csv(output / "leverage_audit.csv", index=False)
    pd.DataFrame([minute_quality, bar_quality]).to_csv(output / "data_quality.csv", index=False)
    core_bars.reset_index().rename(columns={"index": "timestamp"}).to_csv(output / "ohlcv_4h.csv", index=False)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "warmup_start": str(warmup_start),
                "neighbourhoods": [{"name": name, "momentum_days": list(days)} for name, days in NEIGHBOURHOODS],
                "top_day_removal": top_day,
                "decision": "paper_trade_only",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    artifact = _artifact(
        cfg=cfg,
        generated_at=generated_at,
        base_summary=base_summary,
        benchmark_summary=benchmark_summary,
        scenario_rows=scenario_rows,
        yearly_rows=yearly.to_dict(orient="records"),
        equity_rows=_build_equity_dataset(base, benchmark),
        shock_rows=shock.to_dict(orient="records"),
        data_quality_rows=pd.DataFrame([minute_quality, bar_quality]).fillna(0).to_dict(orient="records"),
        neighbourhood_rows=neighbourhood_rows,
        top_day=top_day,
    )
    (output / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"base": base_summary, "buy_hold": benchmark_summary, "top_day": top_day}, ensure_ascii=False, indent=2))
    print(f"[artifact] {output / 'artifact.json'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
