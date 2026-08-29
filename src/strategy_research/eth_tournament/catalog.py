from __future__ import annotations

from .contracts import SourceClass, StrategySpec


def strategy_catalog() -> list[StrategySpec]:
    return [
        StrategySpec(
            "s01_donchian_ensemble_long", "S01", "Donchian Ensemble Trend", "long-only daily ensemble",
            SourceClass.SOURCE_FAITHFUL,
            "Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907",
            "Nine Donchian speeds (5/10/20/30/60/90/150/250/360d), midpoint trailing exits, 90d volatility sizing to 25% annualized target, 2x cap; single-ETH implementation of the paper's trend model.",
            ("trade_1m",), "weight", {"long_short": False},
        ),
        StrategySpec(
            "s01_donchian_ensemble_ls", "S01", "Donchian Ensemble Trend", "symmetric long-short perpetual variant",
            SourceClass.SOURCE_VARIANT,
            "Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907",
            "Symmetric long/short implementation of the Donchian ensemble for perpetual futures; risk sizing follows the paper family.",
            ("trade_1m",), "weight", {"long_short": True},
        ),
        StrategySpec(
            "s02_ma20_50_voltrend", "S02", "Vol-Scaled Moving-Average Trend", "20/50 daily SMA crossover",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "A Decade of Evidence of Trend Following Investing in Cryptocurrencies",
            "https://arxiv.org/abs/2009.12155",
            "Canonical 20/50-day moving-average trend signal, scaled to 20% annualized 30d volatility with 2x cap. The cited work/code supports MA trend-following in crypto; this exact 20/50 risk overlay is frozen engineering, not claimed as the paper's optimum.",
            ("trade_1m",), "weight", {"fast": 20, "slow": 50, "vol_window": 30, "vol_target": 0.20},
        ),
        StrategySpec(
            "s03_bb_rsi_mr_1h", "S03", "Bollinger + RSI Mean Reversion", "1h symmetric mean reversion",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "Bollinger Bands under Varying Market Regimes + canonical RSI filter",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962",
            "1h BB(20,2) fade only when RSI(14) is <30/>70; exit at middle band or RSI normalization; 2 ATR catastrophe stop. Frozen before evaluation.",
            ("trade_1m",), "event", {"timeframe": "1h", "mode": "mean_reversion"},
        ),
        StrategySpec(
            "s03_bb_breakout_4h", "S03", "Bollinger Regime Strategy", "4h band breakout",
            SourceClass.SOURCE_VARIANT,
            "Bollinger Bands under Varying Market Regimes",
            "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962",
            "4h close breakout above/below BB(20,2), exit on middle-band failure; symmetric perpetual adaptation with 3 ATR catastrophe stop.",
            ("trade_1m",), "event", {"timeframe": "4h", "mode": "breakout"},
        ),
        StrategySpec(
            "s04_turtle_system2", "S04", "Original Turtle Breakout", "System 2",
            SourceClass.SOURCE_FAITHFUL,
            "Original Turtle Trading Rules - System 2",
            "https://www.theturtletrader.com/turtle-trading-rules/",
            "55-day breakout, ATR/N(20), 2N stop, add every 0.5N up to four units, exit on opposite 20-day extreme. Single-market risk is capped by tournament leverage/risk limits.",
            ("trade_1m",), "turtle", {"entry_days": 55, "exit_days": 20, "max_units": 4},
        ),
        StrategySpec(
            "s05_footprint_absorption", "S05", "Footprint Absorption Reversal", "r0020 step1",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "Practitioner footprint absorption / order-flow framework",
            "https://atas.net/blog/absorption-of-demand-and-supply-in-the-footprint-chart/",
            "Completed r0020 range bar: extreme one-sided footprint delta in outer third but close reclaims opposite part of bar. Past-only rolling thresholds; 1.25 range stop, 2R target, 180m max hold.",
            ("trade_1m", "range_r0020", "footprint_r0020_step1"), "event", {},
        ),
        StrategySpec(
            "s06_cvd_exhaustion", "S06", "CVD Exhaustion Fade", "15m divergence + reclaim",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "Practitioner CVD divergence/absorption framework",
            "https://www.backquant.com/learn/cvd",
            "15m price makes a 12-bar extreme while CVD fails to confirm and close reclaims the prior extreme. 1.5 ATR stop, 2.5 ATR target, 360m max hold.",
            ("trade_1m",), "event", {},
        ),
        StrategySpec(
            "s07_flow_confirmed_breakout", "S07", "Flow-Confirmed Breakout", "15m 24h channel + flow z-score",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "Order-flow imbalance literature + breakout trading",
            "https://www.sciencedirect.com/science/article/pii/S1386418126000029",
            "15m close breaks prior 24h channel while signed notional imbalance z-score confirms direction. 2 ATR stop, 3 ATR target, 720m max hold.",
            ("trade_1m",), "event", {},
        ),
        StrategySpec(
            "s08_quarter_hour_oi_4h", "S08", "Quarter-Hour Order Imbalance", "4h hold",
            SourceClass.SOURCE_INSPIRED_ENGINEERING,
            "The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures",
            "https://arxiv.org/abs/2607.09426",
            "First 10 seconds of 00/15/30/45-minute marks; past-only 30d z-score of signed notional imbalance; |z|>=1.5 enters in sign direction; fixed 4h hold with 2.5 ATR catastrophe stop.",
            ("trade_1m", "trade_5s"), "event", {"hold_minutes": 240},
        ),
        StrategySpec(
            "s08_quarter_hour_oi_8h", "S08", "Quarter-Hour Order Imbalance", "8h hold",
            SourceClass.SOURCE_VARIANT,
            "The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures",
            "https://arxiv.org/abs/2607.09426",
            "Same frozen first-10s imbalance signal with 8h hold; 4-12h forecast horizon is source-supported, trading rule is an engineering conversion.",
            ("trade_1m", "trade_5s"), "event", {"hold_minutes": 480},
        ),
        StrategySpec(
            "s08_quarter_hour_oi_12h", "S08", "Quarter-Hour Order Imbalance", "12h hold",
            SourceClass.SOURCE_VARIANT,
            "The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures",
            "https://arxiv.org/abs/2607.09426",
            "Same frozen first-10s imbalance signal with 12h hold; no future/full-sample thresholding.",
            ("trade_1m", "trade_5s"), "event", {"hold_minutes": 720},
        ),
    ]
