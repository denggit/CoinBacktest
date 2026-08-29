# Source Catalog — Frozen Before ETH Results

The source class is part of the research contract:

- `SOURCE_FAITHFUL`: source provides a sufficiently complete mechanical rule set and V1 follows it, subject only to explicit single-ETH risk/execution constraints.
- `SOURCE_VARIANT`: source explicitly supports the family/variant, but V1 adapts execution to ETH perpetual or conservative closed-data timing.
- `SOURCE_INSPIRED_ENGINEERING`: source establishes a market mechanism/effect but not a complete executable trading system. V1's conversion to entry/exit/risk rules is frozen here **before** seeing ETH tournament results.

No failed result may be repaired by silently changing these rules.

## S01 — Donchian Ensemble Trend

Source: Carlo Zarattini, Alberto Pagani, Andrea Barbon, *Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins* (Swiss Finance Institute Research Paper No. 25-80).

Source URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5209907

Frozen implementation:
- daily bars
- Donchian speeds: 5, 10, 20, 30, 60, 90, 150, 250, 360 days
- entry on close beyond prior channel extreme
- midpoint channel trailing exit per sleeve
- equal sleeve ensemble
- 90-day annualized volatility scaling
- 25% volatility target, max 2x exposure
- long-only base and symmetric long-short perpetual variant

The paper is multi-coin/rotational; this tournament deliberately tests whether the core time-series trend system works on ETH alone.

## S02 — Vol-Scaled Moving-Average Trend

Source: Evans Rozario, Samuel Holt, James West, Shaun Ng, *A Decade of Evidence of Trend Following Investing in Cryptocurrencies*.

Paper: https://arxiv.org/abs/2009.12155
Code: https://github.com/Globe-Research/bittrends

The paper/code support moving-average trend-following as a crypto research family. V1 freezes a canonical 20/50-day SMA crossover with a 30-day 20%-annualized volatility target and 2x cap. These exact parameters are an engineering specification, not claimed to be the paper's optimized choice.

## S03 — Bollinger Regime Strategies

Source: Efe Arda, *Bollinger Bands under Varying Market Regimes: A Comparative Study of Breakout and Mean-Reversion Strategies in BTC/USDT*.

Source URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5775962

V1 freezes two variants:
- 1h mean reversion: BB(20,2) lower/upper fade plus RSI(14) <30/>70; exit on middle-band/RSI normalization; 2 ATR catastrophe stop.
- 4h breakout: close outside BB(20,2), exit on middle-band failure; symmetric perpetual implementation; 3 ATR catastrophe stop.

The catastrophe stops and RSI combination are engineering risk controls; they are not presented as author-original parameters.

## S04 — Turtle System 2

Public complete-rule summary: https://www.theturtletrader.com/turtle-trading-rules/

Frozen implementation:
- 55-day breakout, both directions
- N = 20-day ATR
- initial stop = 2N
- add every +0.5N in favorable direction
- max 4 units
- opposite 20-day extreme exit
- intraday threshold monitoring on the 1m path using only prior completed daily context

The original portfolio-level correlated-market unit caps are not applicable to a single ETH instrument. Tournament-wide 1% first-unit risk and 2x max notional cap are explicit engineering safety constraints.

## S05 — Footprint Absorption Reversal

Practitioner source on absorption mechanics: https://atas.net/blog/absorption-of-demand-and-supply-in-the-footprint-chart/

The source describes aggressive sells/buys being absorbed by passive liquidity and emphasizes heavy directional activity at an extreme that fails to continue price.

V1 frozen engineering conversion:
- r0020 range bar + step1 footprint
- outer-third delta ratio compared with its own past-only rolling 10th/90th percentile
- long: extreme negative lower-third delta, total negative delta, but close position >= 65% of the range
- short: mirror
- stop = 1.25 range widths
- target = 2R
- max hold = 180m

## S06 — CVD Exhaustion Fade

Practitioner CVD reference: https://www.backquant.com/learn/cvd

V1 frozen engineering conversion:
- 15m CVD from aggressive buy minus sell notional
- 12-bar price extreme with non-confirming CVD extreme
- price must reclaim the prior extreme on the closed bar
- stop = 1.5 ATR(14)
- target = 2.5 ATR(14)
- max hold = 360m

## S07 — Flow-Confirmed Breakout

Evidence source: Alexia Anastasopoulos, Nikola Gradojevic, Fred Liu, Alex Maynard, Ilias Tsiakas, *Order flow and cryptocurrency returns*, Journal of Financial Markets 79 (2026), Article 101047.

Source URL: https://www.sciencedirect.com/science/article/pii/S1386418126000029

The paper establishes predictive information in crypto order flow; it does not prescribe this ETH intraday setup.

V1 frozen engineering conversion:
- 15m close breaks the prior 24h high/low
- same closed bar's normalized signed notional flow must have past-only 7d z-score >= +1 or <= -1
- stop = 2 ATR(14)
- target = 3 ATR(14)
- max hold = 720m

## S08 — Quarter-Hour Opening Order Imbalance

Source: Chan Kim, Peter Reinhard Hansen, *The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures*.

Source URL: https://arxiv.org/abs/2607.09426

The paper documents that order imbalance at quarter-hour openings predicts returns over roughly 4–12 hours. It is evidence of a predictable mechanism, not a published ready-to-run trade strategy.

V1 frozen engineering conversion:
- only the first 10 seconds of 00/15/30/45-minute marks
- signed notional imbalance = (buy - sell)/(buy + sell)
- past-only 30-day rolling z-score
- trade only when |z| >= 1.5, in imbalance direction
- 2.5 ATR(15m) catastrophe stop
- source-supported hold variants: 4h, 8h, 12h
- no full-sample quantile or future threshold

## Interpretation rule

A profitable source paper is not evidence that the same rule will make money on ETH perpetual after 0.11% round-trip cost. The entire purpose of this tournament is to reject that assumption quickly and consistently.
