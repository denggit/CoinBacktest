from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .data_sources import (
    Coverage,
    build_monitor_panel,
    coverage_for_frame,
    fetch_alpaca_bars,
    fetch_cnbc_yield_intraday,
    fetch_okx_bars,
    fetch_okx_public_intraday,
    fetch_yahoo_daily,
    fetch_yahoo_intraday,
    load_alpaca_bars,
    load_cnbc_yield_intraday,
    load_existing_okx_bars,
    load_fred_yields,
    load_monitor_observations,
    load_okx_public_intraday,
    load_research_okx_bars,
    load_yahoo_daily,
    load_yahoo_intraday,
    write_inventory,
)
from .event_study import (
    aggregate_daily_bars,
    align_intraday_events,
    apply_round_trip_costs,
    assign_chronological_fold,
    daily_forward_returns,
    summarize_chronological_folds,
    summarize_event_returns,
)
from .features import (
    build_daily_proxy_panel,
    build_intraday_features,
    select_daily_proxy_events,
    select_daily_static_background,
    select_intraday_events,
)
from .report import build_html_report
from .scheduled_events import (
    align_scheduled_asset_responses,
    build_scheduled_macro_signals,
    intraday_quality_profile,
    leave_one_event_out_summary,
    macro_event_calendar,
    summarize_scheduled_responses,
)


LOGGER = logging.getLogger("macro_edge_research")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative macro repricing cross-asset edge research")
    parser.add_argument("--download", action="store_true", help="Refresh free remote source caches before analysis")
    parser.add_argument("--inventory-only", action="store_true", help="Write source coverage and stop")
    parser.add_argument("--start", default="2019-01-01", help="Long-history start date")
    parser.add_argument("--end", default=None, help="End date; defaults to now UTC")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _combine_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame()
    result = pd.concat(valid).sort_index()
    return result.loc[~result.index.duplicated(keep="last")]


def _monitor_bounds(observations: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if observations.empty:
        return None, None
    start = pd.Timestamp(observations["timestamp_utc"].min()).tz_convert("UTC") - pd.Timedelta(hours=2)
    end = pd.Timestamp(observations["timestamp_utc"].max()).tz_convert("UTC") + pd.Timedelta(hours=2)
    return start, end


def download_sources(config: ResearchConfig, start: str, end: pd.Timestamp) -> list[str]:
    """Refresh all network-backed research inputs; continue after source failures."""

    messages: list[str] = []
    observations = load_monitor_observations(config.macro_db)
    monitor_start, monitor_end = _monitor_bounds(observations)
    # Alpaca's free SIP entitlement rejects a request whose end timestamp falls
    # inside the current delayed-data window, even when the requested symbol
    # has no bar there (for example on a weekend). Keep the request safely
    # behind that boundary instead of treating the credentials as invalid.
    alpaca_latest = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    if monitor_end is not None:
        monitor_end = min(monitor_end, alpaca_latest)
    for symbol in config.equity_symbols:
        if monitor_start is None or monitor_end is None:
            continue
        try:
            frame = fetch_alpaca_bars(config, symbol, "1Min", monitor_start, monitor_end)
            messages.append(f"Alpaca {symbol} 1Min: fetched {len(frame):,} rows")
        except Exception as exc:
            messages.append(f"WARNING Alpaca {symbol} 1Min: {exc}")

    for symbol in (*config.equity_symbols, "DX-Y.NYB", "ZQ=F", "GC=F"):
        try:
            frame = fetch_yahoo_daily(config, symbol, start, end)
            messages.append(f"Yahoo {symbol} daily: fetched {len(frame):,} rows")
        except Exception as exc:
            messages.append(f"WARNING Yahoo {symbol}: {exc}")

    intraday_symbols = (
        *config.equity_symbols,
        "DX-Y.NYB",
        "ZQU26.CBT",
        "ZQV26.CBT",
        "ZT=F",
        "ZN=F",
        "^TNX",
        "ETH-USD",
        "GC=F",
    )
    for symbol in intraday_symbols:
        try:
            frame = fetch_yahoo_intraday(
                config,
                symbol,
                range_value=config.intraday_history_range,
                interval=config.intraday_history_interval,
            )
            messages.append(
                f"Yahoo {symbol} {config.intraday_history_interval}: fetched {len(frame):,} rows"
            )
        except Exception as exc:
            messages.append(f"WARNING Yahoo {symbol} intraday: {exc}")

    for symbol in ("US2Y", "US10Y"):
        try:
            frame = fetch_cnbc_yield_intraday(config, symbol, time_range="5D")
            messages.append(f"CNBC {symbol} exact yield 5D: fetched {len(frame):,} rows")
        except Exception as exc:
            messages.append(f"WARNING CNBC {symbol} intraday: {exc}")

    intraday_start = end - pd.Timedelta(days=65)
    for symbol in ("ETH-USDT-SWAP", "XAU-USDT-SWAP"):
        try:
            frame = fetch_okx_public_intraday(
                config,
                symbol,
                start_utc=intraday_start,
                end_utc=end,
                bar=config.intraday_history_interval,
            )
            messages.append(
                f"OKX {symbol} {config.intraday_history_interval}: fetched {len(frame):,} rows"
            )
        except Exception as exc:
            messages.append(f"WARNING OKX {symbol} intraday history: {exc}")

    if monitor_start is not None and monitor_end is not None:
        for symbol in ("ETH-USDT-SWAP", "XAU-USDT-SWAP"):
            try:
                frame = fetch_okx_bars(config, symbol, "1m", monitor_start, monitor_end)
                messages.append(f"OKX {symbol} 1m: cached {len(frame):,} rows")
            except Exception as exc:
                messages.append(f"WARNING OKX {symbol}: {exc}")
    return messages


def collect_inventory(config: ResearchConfig) -> tuple[list[Coverage], pd.DataFrame]:
    observations = load_monitor_observations(config.macro_db)
    coverages: list[Coverage] = []
    if observations.empty:
        coverages.append(Coverage("local SQLite", "macro monitor observations", 0, None, None, "missing"))
    else:
        indexed = observations.set_index("timestamp_utc")
        coverages.append(
            coverage_for_frame(
                "local SQLite",
                "true FedWatch / US2Y / US10Y / DXY observations",
                indexed,
                "true live-monitor data; short local history",
            )
        )

    for symbol in config.equity_symbols:
        frame = load_alpaca_bars(config, symbol, "1Min")
        coverages.append(
            coverage_for_frame(
                "Alpaca SIP",
                f"{symbol} 1Min split-adjusted",
                frame,
                "listed ETF; true-event window",
            )
        )

    existing_eth_daily = load_existing_okx_bars(config, "ETH-USDT-SWAP", "1D")
    coverages.append(coverage_for_frame("OKX local cache", "ETH-USDT-SWAP 1D", existing_eth_daily))
    for symbol in ("ETH-USDT-SWAP", "XAU-USDT-SWAP"):
        frame = load_research_okx_bars(config, symbol, "1m")
        coverages.append(
            coverage_for_frame(
                "OKX public market data",
                f"{symbol} 1m research cache",
                frame,
                "perpetual swap; not the listed ETF" if symbol.startswith("XAU") else "perpetual swap",
            )
        )
    fred = load_fred_yields(config)
    coverages.append(coverage_for_frame("FRED", "DGS2 / DGS10 daily", fred, "daily background"))
    for symbol, note in (
        ("SOXX", "listed ETF; adjusted daily history"),
        ("SOXL", "listed 3x ETF; measured directly"),
        ("QQQ", "listed ETF; adjusted daily history"),
        ("DX-Y.NYB", "DXY daily proxy source"),
        ("ZQ=F", "front continuous futures; not FedWatch probability history"),
        ("GC=F", "gold futures proxy; not OKX XAU perpetual"),
    ):
        coverages.append(coverage_for_frame("Yahoo Finance", f"{symbol} daily", load_yahoo_daily(config, symbol), note))

    for symbol, note in (
        ("SOXX", "listed ETF; free 60-day intraday window"),
        ("SOXL", "listed 3x ETF; measured directly"),
        ("QQQ", "listed ETF; free 60-day intraday window"),
        ("DX-Y.NYB", "DXY index quote"),
        ("ZQU26.CBT", "Sep-2026 30-Day Fed Funds Futures; proxy, not FedWatch"),
        ("ZQV26.CBT", "Oct-2026 post-FOMC Fed Funds Futures; proxy, not FedWatch"),
        ("ZT=F", "2Y Treasury futures price; not 2Y yield"),
        ("ZN=F", "10Y Treasury futures price; not 10Y yield"),
        ("^TNX", "10Y yield quote; used only where CNBC exact history is unavailable"),
        ("ETH-USD", "spot-index proxy fallback; OKX swap preferred"),
        ("GC=F", "gold futures proxy fallback; OKX XAU swap preferred"),
    ):
        coverages.append(
            coverage_for_frame(
                "Yahoo Finance",
                f"{symbol} {config.intraday_history_interval}",
                load_yahoo_intraday(config, symbol, interval=config.intraday_history_interval),
                note,
            )
        )
    for symbol in ("US2Y", "US10Y"):
        coverages.append(
            coverage_for_frame(
                "CNBC / Tradeweb",
                f"{symbol} exact yield intraday",
                load_cnbc_yield_intraday(config, symbol, time_range="5D"),
                "exact yield; public chart provides only a short intraday window",
            )
        )
    for symbol in ("ETH-USDT-SWAP", "XAU-USDT-SWAP"):
        coverages.append(
            coverage_for_frame(
                "OKX public market data",
                f"{symbol} {config.intraday_history_interval}",
                load_okx_public_intraday(
                    config, symbol, bar=config.intraday_history_interval
                ),
                "perpetual swap; direct instrument history",
            )
        )
    return coverages, observations


def _save(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def run_analysis(config: ResearchConfig, coverages: list[Coverage], observations: pd.DataFrame) -> dict[str, Path]:
    panel = build_monitor_panel(observations)
    features = build_intraday_features(panel)
    intraday_events = select_intraday_events(
        features,
        config.thresholds,
        cooldown_minutes=config.event_cooldown_minutes,
    )

    equity_intraday = {
        symbol: load_alpaca_bars(config, symbol, "1Min") for symbol in config.equity_symbols
    }
    eth_research = load_research_okx_bars(config, "ETH-USDT-SWAP", "1m")
    monitor_start, monitor_end = _monitor_bounds(observations)
    eth_existing = pd.DataFrame()
    if monitor_start is not None and monitor_end is not None:
        eth_existing = load_existing_okx_bars(
            config,
            "ETH-USDT-SWAP",
            "1m",
            monitor_start,
            monitor_end,
        )
    intraday_assets = {
        **equity_intraday,
        "ETH-USDT-SWAP": _combine_frames(eth_existing, eth_research),
        "XAU-USDT-SWAP": load_research_okx_bars(config, "XAU-USDT-SWAP", "1m"),
    }
    aligned_intraday = align_intraday_events(
        intraday_events,
        intraday_assets,
        config.intraday_horizons_minutes,
    )
    if not aligned_intraday.empty:
        aligned_intraday["event_cluster"] = pd.to_datetime(
            aligned_intraday["timestamp_utc"], utc=True
        ).dt.strftime("%Y-%m-%d")
    intraday_summary = summarize_event_returns(
        aligned_intraday,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed,
        cluster_column="event_cluster",
    )

    fred = load_fred_yields(config)
    dxy = load_yahoo_daily(config, "DX-Y.NYB")
    zq = load_yahoo_daily(config, "ZQ=F")
    daily_panel = build_daily_proxy_panel(fred, dxy, zq)
    daily_events = select_daily_proxy_events(daily_panel)

    daily_assets: dict[str, pd.DataFrame] = {}
    for symbol in config.equity_symbols:
        daily_assets[symbol] = aggregate_daily_bars(
            load_yahoo_daily(config, symbol), equity_session=True
        )
    daily_assets["ETH-USDT-SWAP"] = aggregate_daily_bars(
        load_existing_okx_bars(config, "ETH-USDT-SWAP", "1D"), equity_session=False
    )
    gold = load_yahoo_daily(config, "GC=F")
    daily_assets["GC=F proxy"] = aggregate_daily_bars(gold, equity_session=False)
    aligned_daily = daily_forward_returns(daily_events, daily_assets, config.daily_horizons_sessions)
    cost_basis_points = {
        "SOXX": 5.0,
        "QQQ": 5.0,
        "SOXL": 10.0,
        "ETH-USDT-SWAP": 10.0,
        "GC=F proxy": 8.0,
    }
    aligned_daily = apply_round_trip_costs(aligned_daily, cost_basis_points)
    if not aligned_daily.empty:
        aligned_daily["event_cluster"] = pd.to_datetime(aligned_daily["signal_date"]).dt.to_period("W").astype(str)
        aligned_daily["fold"] = assign_chronological_fold(aligned_daily["signal_date"])
    daily_for_summary = aligned_daily.rename(columns={"horizon_sessions": "horizon_minutes"})
    daily_summary = summarize_event_returns(
        daily_for_summary,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 10_000,
        cluster_column="event_cluster",
    ).rename(columns={"horizon_minutes": "horizon_sessions"})
    daily_net_summary = summarize_event_returns(
        daily_for_summary,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 20_000,
        return_column="net_forward_return_pct",
        cluster_column="event_cluster",
    ).rename(columns={"horizon_minutes": "horizon_sessions"})
    daily_fold_summary = summarize_chronological_folds(
        daily_for_summary,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 30_000,
        return_column="net_forward_return_pct",
        cluster_column="event_cluster",
    ).rename(columns={"horizon_minutes": "horizon_sessions"})

    ablation_parts: list[pd.DataFrame] = []
    if not daily_for_summary.empty and "primary_driver" in daily_for_summary:
        for offset, (subset_name, subset) in enumerate(daily_for_summary.groupby("primary_driver", sort=True)):
            summary = summarize_event_returns(
                subset,
                bootstrap_samples=config.bootstrap_samples,
                random_seed=config.random_seed + 40_000 + offset * 1_000,
                return_column="net_forward_return_pct",
                cluster_column="event_cluster",
            )
            if not summary.empty:
                summary.insert(0, "signal_subset", subset_name)
                ablation_parts.append(summary)
    daily_ablation_summary = (
        pd.concat(ablation_parts, ignore_index=True).rename(columns={"horizon_minutes": "horizon_sessions"})
        if ablation_parts
        else pd.DataFrame()
    )

    static_events = select_daily_static_background(daily_panel)
    aligned_static = daily_forward_returns(static_events, daily_assets, config.daily_horizons_sessions)
    aligned_static = apply_round_trip_costs(aligned_static, cost_basis_points)
    if not aligned_static.empty:
        aligned_static["event_cluster"] = pd.to_datetime(aligned_static["signal_date"]).dt.to_period("W").astype(str)
        aligned_static["fold"] = assign_chronological_fold(aligned_static["signal_date"])
    static_for_summary = aligned_static.rename(columns={"horizon_sessions": "horizon_minutes"})
    static_summary = summarize_event_returns(
        static_for_summary,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 50_000,
        return_column="net_forward_return_pct",
        cluster_column="event_cluster",
    ).rename(columns={"horizon_minutes": "horizon_sessions"})

    # Second iteration: scheduled 2026 releases using a 60-day, five-minute
    # free-data panel. Futures remain explicitly labelled proxies. CNBC exact
    # yields are used only where their short public intraday history overlaps.
    macro_intraday = {
        "ZQ_POST_FOMC": load_yahoo_intraday(
            config, "ZQV26.CBT", interval=config.intraday_history_interval
        ),
        "ZT": load_yahoo_intraday(
            config, "ZT=F", interval=config.intraday_history_interval
        ),
        "US10Y_YAHOO": load_yahoo_intraday(
            config, "^TNX", interval=config.intraday_history_interval
        ),
        "DXY": load_yahoo_intraday(
            config, "DX-Y.NYB", interval=config.intraday_history_interval
        ),
        "US2Y_EXACT": load_cnbc_yield_intraday(config, "US2Y", time_range="5D"),
        "US10Y_EXACT": load_cnbc_yield_intraday(config, "US10Y", time_range="5D"),
    }
    required_macro = [
        macro_intraday[name]
        for name in ("ZQ_POST_FOMC", "ZT", "US10Y_YAHOO", "DXY")
        if not macro_intraday[name].empty
    ]
    if len(required_macro) == 4:
        scheduled_start = max(pd.DatetimeIndex(frame.index).min() for frame in required_macro)
        scheduled_end = min(pd.DatetimeIndex(frame.index).max() for frame in required_macro)
        scheduled_calendar = macro_event_calendar(scheduled_start, scheduled_end)
    else:
        scheduled_calendar = pd.DataFrame()
    scheduled_signals = build_scheduled_macro_signals(
        scheduled_calendar,
        macro_intraday,
        config.scheduled_signal_delays_minutes,
    ) if not scheduled_calendar.empty else pd.DataFrame()

    scheduled_assets = {
        symbol: load_yahoo_intraday(
            config, symbol, interval=config.intraday_history_interval
        )
        for symbol in config.equity_symbols
    }
    okx_eth = load_okx_public_intraday(
        config, "ETH-USDT-SWAP", bar=config.intraday_history_interval
    )
    okx_xau = load_okx_public_intraday(
        config, "XAU-USDT-SWAP", bar=config.intraday_history_interval
    )
    if not okx_eth.empty:
        scheduled_assets["ETH-USDT-SWAP"] = okx_eth
    else:
        scheduled_assets["ETH-USD proxy"] = load_yahoo_intraday(
            config, "ETH-USD", interval=config.intraday_history_interval
        )
    if not okx_xau.empty:
        scheduled_assets["XAU-USDT-SWAP"] = okx_xau
    else:
        scheduled_assets["GC=F proxy"] = load_yahoo_intraday(
            config, "GC=F", interval=config.intraday_history_interval
        )

    scheduled_responses = align_scheduled_asset_responses(
        scheduled_signals,
        scheduled_assets,
        execution_delays_minutes=config.execution_delays_minutes,
        horizons_minutes=config.scheduled_horizons_minutes,
    ) if not scheduled_signals.empty else pd.DataFrame()
    scheduled_summary = summarize_scheduled_responses(
        scheduled_responses,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 60_000,
    )
    scheduled_event_type_summary = summarize_scheduled_responses(
        scheduled_responses,
        return_column="net_return_5bp_pct",
        by_event_type=True,
        bootstrap_samples=config.bootstrap_samples,
        random_seed=config.random_seed + 70_000,
    )
    cost_parts: list[pd.DataFrame] = []
    for offset, (cost_bp, return_column) in enumerate(
        ((0, "forward_return_pct"), (5, "net_return_5bp_pct"), (10, "net_return_10bp_pct"))
    ):
        part = summarize_scheduled_responses(
            scheduled_responses,
            return_column=return_column,
            bootstrap_samples=config.bootstrap_samples,
            random_seed=config.random_seed + 80_000 + offset * 5_000,
        )
        if not part.empty:
            part.insert(0, "cost_stress_bp", cost_bp)
            cost_parts.append(part)
    scheduled_latency_cost_summary = (
        pd.concat(cost_parts, ignore_index=True) if cost_parts else pd.DataFrame()
    )
    scheduled_loo = leave_one_event_out_summary(scheduled_responses)

    sensitivity_parts: list[pd.DataFrame] = []
    for offset, multiplier in enumerate((0.75, 1.0, 1.5)):
        sensitivity_signals = build_scheduled_macro_signals(
            scheduled_calendar,
            macro_intraday,
            config.scheduled_signal_delays_minutes,
            threshold_multiplier=multiplier,
        ) if not scheduled_calendar.empty else pd.DataFrame()
        sensitivity_responses = align_scheduled_asset_responses(
            sensitivity_signals,
            scheduled_assets,
            execution_delays_minutes=(0,),
            horizons_minutes=(60,),
        ) if not sensitivity_signals.empty else pd.DataFrame()
        sensitivity_summary = summarize_scheduled_responses(
            sensitivity_responses,
            return_column="net_return_5bp_pct",
            bootstrap_samples=config.bootstrap_samples,
            random_seed=config.random_seed + 90_000 + offset * 5_000,
        )
        if not sensitivity_summary.empty:
            sensitivity_summary.insert(0, "threshold_multiplier", multiplier)
            sensitivity_parts.append(sensitivity_summary)
    scheduled_threshold_sensitivity = (
        pd.concat(sensitivity_parts, ignore_index=True)
        if sensitivity_parts
        else pd.DataFrame()
    )
    scheduled_data_quality = intraday_quality_profile(
        {
            **{f"macro:{name}": frame for name, frame in macro_intraday.items()},
            **{f"asset:{name}": frame for name, frame in scheduled_assets.items()},
        }
    )

    paths = {
        "monitor_panel": config.output_dir / "01_monitor_panel.csv",
        "intraday_events": config.output_dir / "02_intraday_events.csv",
        "intraday_aligned": config.output_dir / "03_intraday_asset_responses.csv",
        "intraday_summary": config.output_dir / "04_intraday_edge_summary.csv",
        "daily_proxy_panel": config.output_dir / "05_daily_proxy_panel.csv",
        "daily_proxy_events": config.output_dir / "06_daily_proxy_events.csv",
        "daily_proxy_aligned": config.output_dir / "07_daily_proxy_asset_responses.csv",
        "daily_proxy_summary": config.output_dir / "08_daily_proxy_edge_summary.csv",
        "daily_proxy_net_summary": config.output_dir / "09_daily_proxy_net_edge_summary.csv",
        "daily_proxy_fold_summary": config.output_dir / "10_daily_proxy_fold_summary.csv",
        "daily_proxy_ablation_summary": config.output_dir / "11_daily_proxy_ablation_summary.csv",
        "static_background_events": config.output_dir / "12_static_background_events.csv",
        "static_background_summary": config.output_dir / "13_static_background_summary.csv",
        "scheduled_event_calendar": config.output_dir / "14_scheduled_event_calendar.csv",
        "scheduled_proxy_signals": config.output_dir / "15_scheduled_proxy_signals.csv",
        "scheduled_asset_responses": config.output_dir / "16_scheduled_asset_responses.csv",
        "scheduled_edge_summary": config.output_dir / "17_scheduled_edge_summary.csv",
        "scheduled_event_type_summary": config.output_dir / "18_scheduled_event_type_summary.csv",
        "scheduled_latency_cost_summary": config.output_dir / "19_scheduled_latency_cost_summary.csv",
        "scheduled_leave_one_event_out": config.output_dir / "20_scheduled_leave_one_event_out.csv",
        "scheduled_threshold_sensitivity": config.output_dir / "21_scheduled_threshold_sensitivity.csv",
        "scheduled_data_quality": config.output_dir / "22_scheduled_data_quality.csv",
        "report": config.output_dir / "macro_repricing_edge_report.html",
        "run_meta": config.output_dir / "run_meta.json",
    }
    _save(features.reset_index(), paths["monitor_panel"])
    _save(intraday_events, paths["intraday_events"])
    _save(aligned_intraday, paths["intraday_aligned"])
    _save(intraday_summary, paths["intraday_summary"])
    _save(daily_panel.reset_index(), paths["daily_proxy_panel"])
    _save(daily_events.reset_index(), paths["daily_proxy_events"])
    _save(aligned_daily, paths["daily_proxy_aligned"])
    _save(daily_summary, paths["daily_proxy_summary"])
    _save(daily_net_summary, paths["daily_proxy_net_summary"])
    _save(daily_fold_summary, paths["daily_proxy_fold_summary"])
    _save(daily_ablation_summary, paths["daily_proxy_ablation_summary"])
    _save(static_events.reset_index(), paths["static_background_events"])
    _save(static_summary, paths["static_background_summary"])
    _save(scheduled_calendar, paths["scheduled_event_calendar"])
    _save(scheduled_signals, paths["scheduled_proxy_signals"])
    _save(scheduled_responses, paths["scheduled_asset_responses"])
    _save(scheduled_summary, paths["scheduled_edge_summary"])
    _save(scheduled_event_type_summary, paths["scheduled_event_type_summary"])
    _save(scheduled_latency_cost_summary, paths["scheduled_latency_cost_summary"])
    _save(scheduled_loo, paths["scheduled_leave_one_event_out"])
    _save(scheduled_threshold_sensitivity, paths["scheduled_threshold_sensitivity"])
    _save(scheduled_data_quality, paths["scheduled_data_quality"])

    generated_at = pd.Timestamp.now(tz="UTC")
    build_html_report(
        paths["report"],
        coverages=coverages,
        intraday_events=intraday_events,
        intraday_summary=intraday_summary,
        daily_proxy_events=daily_events,
        daily_proxy_summary=daily_summary,
        daily_proxy_net_summary=daily_net_summary,
        daily_proxy_fold_summary=daily_fold_summary,
        daily_proxy_ablation_summary=daily_ablation_summary,
        static_background_events=static_events,
        static_background_summary=static_summary,
        scheduled_signals=scheduled_signals,
        scheduled_summary=scheduled_summary,
        scheduled_event_type_summary=scheduled_event_type_summary,
        scheduled_latency_cost_summary=scheduled_latency_cost_summary,
        scheduled_leave_one_event_out=scheduled_loo,
        scheduled_threshold_sensitivity=scheduled_threshold_sensitivity,
        scheduled_data_quality=scheduled_data_quality,
        generated_at_utc=generated_at,
    )
    meta = {
        "generated_at_utc": generated_at.isoformat(),
        "config": {
            "event_cooldown_minutes": config.event_cooldown_minutes,
            "bootstrap_samples": config.bootstrap_samples,
            "intraday_horizons_minutes": config.intraday_horizons_minutes,
            "daily_horizons_sessions": config.daily_horizons_sessions,
            "thresholds": asdict(config.thresholds),
        },
        "counts": {
            "monitor_observations": len(observations),
            "monitor_panel_rows": len(panel),
            "intraday_events": len(intraday_events),
            "intraday_aligned_rows": len(aligned_intraday),
            "daily_proxy_events": len(daily_events),
            "daily_proxy_aligned_rows": len(aligned_daily),
            "static_background_events": len(static_events),
            "scheduled_release_anchors": len(scheduled_calendar),
            "scheduled_proxy_signals": len(scheduled_signals),
            "scheduled_proxy_non_stable_signals": int(
                scheduled_signals["regime"].ne("stable").sum()
            ) if not scheduled_signals.empty else 0,
            "scheduled_asset_response_rows": len(scheduled_responses),
        },
        "limitations": [
            "Local true FedWatch history begins in 2026-08-28 and is not decision-grade for edge estimation.",
            "ZQ=F is a front continuous futures proxy and can contain contract-roll effects; it is not historical FedWatch probability.",
            "Daily FRED observations are end-of-day background and are not intraday alert timestamps.",
            "CNBC exact US2Y/US10Y intraday history is public but limited to roughly eight recent sessions.",
            "The 60-day scheduled-event panel uses ZT and explicit monthly ZQ futures as diagnostics; futures price is not Treasury yield and ZQ is not FedWatch probability.",
            "The scheduled-event sample is small and event types are heterogeneous; estimates remain exploratory.",
        ],
    }
    paths["run_meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return paths


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = ResearchConfig()
    config.ensure_directories()
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    LOGGER.info("[research] root=%s", config.research_data_dir.parent)
    if args.download:
        for message in download_sources(config, args.start, end):
            if message.startswith("WARNING"):
                LOGGER.warning(message)
            else:
                LOGGER.info(message)

    coverages, observations = collect_inventory(config)
    inventory_path = write_inventory(config, coverages)
    LOGGER.info("[research] inventory=%s", inventory_path)
    if args.inventory_only:
        for item in coverages:
            LOGGER.info("[coverage] %s %s rows=%s %s -> %s", item.source, item.dataset, item.rows, item.start_utc, item.end_utc)
        return 0

    paths = run_analysis(config, coverages, observations)
    LOGGER.info("[research] report=%s", paths["report"])
    LOGGER.info("[research] results=%s", config.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
