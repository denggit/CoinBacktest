# Cumulative Stage Results

## V1 implementation stage

Status: **READY_FOR_LOCAL_TOURNAMENT_RUN**

Completed:
- abandoned R01 abstract opportunity-model path as the active strategy-discovery route
- built a source-backed external strategy tournament with 8 families / 12 frozen specs
- implemented unified real-cost causal replay and stress reporting
- implemented user-priority survivor ranking and non-optimized equal-sleeve portfolio construction
- kept 2026 sealed
- added synthetic causality/rule/fee/ambiguity tests

No profitability conclusion exists yet. The next evidence is the user's local full-data run.

Delivery validation:
- Tournament + data_feed regression: **28/28 passed**
- Python compile / CLI smoke: **PASS**
- Full-repo collection: **BLOCKED_BY_5_PRE_EXISTING_MISSING_LIQUIDITY_PANIC_MODULES**; unrelated to this patch


## V1 full-data result

Status: **COMPLETED / STRATEGY-LEVEL LESSON FROZEN**

Top base results from the user local run:
- Turtle System 2: +42.19%, PF 2.17, MDD 21.40%, 26 episodes, max flat 62.46d.
- MA20/50 vol trend: +27.60%, PF 1.52, MDD 24.13%, 21 episodes.
- Donchian ensemble long: +26.66%, PF 3.63, MDD 12.54%, 16 episodes, max flat 50.00d.
- BB breakout 4H: +7.53%, PF 1.09, MDD 16.58%, 251 trades.
- Donchian long/short: +3.19%, PF 1.10, MDD 19.17%, 44 episodes.

Major failures included footprint absorption (-99.23%), CVD exhaustion (-99.997%), flow-confirmed breakout (-68.48%), and quarter-hour imbalance variants (~-92% to -99.6%).

Frozen lesson: simple trend/channel systems were materially more credible than the complex microstructure event systems, but the winners were too sparse or had too much drawdown/top-trade concentration for the desired copy-trading portfolio.

## R02 implementation stage

Status: **READY_FOR_LOCAL_CONTINUOUS_PORTFOLIO_RUN**

Completed:
- changed research object from discrete trades to continuous target net ETH exposure
- internal sleeves can disagree, but exchange execution is a single net long/short/flat exposure
- froze four equal-weight simple trend families: channel, MA, TSMOM, faster 4H trend
- added causal 90D volatility targeting, 1.5x cap, deadband, optional DD governor, optional rebalance step cap
- added actual-turnover transaction-cost accounting and minute mark-to-market MDD
- redefined inactivity as continuous near-zero exposure time rather than legacy trade gaps
- added 2x/3x cost, +1m/+2m delay and top-positive-day dependency stress
- kept 2026 hard sealed

Delivery validation:
- R02 + V1 tournament + data_feed targeted regression: **39/39 passed**
- Python compile / CLI smoke: **PASS**
- Full-repo collection: **BLOCKED_BY_5_PRE_EXISTING_MISSING_LIQUIDITY_PANIC_MODULES**; no R02 failure reached collection.

## R02 full-data result

Status: **REJECTED AS LIVE PORTFOLIO / CONTINUOUS-EXPOSURE LESSON RETAINED**

User local result for `CP01_CORE_VOL`:
- total return +12.75%, CAGR 4.08%, MDD 15.62%, PF 1.49
- 2x cost +11.03%, 3x cost +9.33%
- 2023 -6.07%, 2024 +8.18%, 2025 +10.84%
- max flat 17.83d, max low exposure 158.33d, max consecutive losing days 15
- average absolute exposure only ~0.198x; maximum ~0.549x
- removing top 5 positive days reduced the strategy below zero

`CP02_DD_GOV` / `CP03_DD_GOV_SMOOTH` both fell to roughly -2.09% while only reducing MDD to ~13.92%. The frozen drawdown governor is therefore rejected; it self-locked the strategy at reduced risk and sacrificed too much recovery/trend participation.

Frozen lesson: continuous target exposure is a useful accounting/execution object, but R02's signal-first averaging diluted stronger standalone trend systems. Do not tune R02. First replicate mature source strategies faithfully, then combine surviving **complete sleeves**, not averaged opinions.

## R03 implementation stage — Source-Locked Trend Replication

Status: **READY_FOR_LOCAL_FULL-DATA_RUN**

Completed:
- stopped adding invented portfolio governors before source baselines are validated
- locked four externally specified baselines: Zarattini long-only, Zarattini long-short appendix, MOP 12M TSMOM, Original Turtle System-2 core
- explicitly separated `SOURCE CORE`, `REQUIRED ETH ADAPTATION`, and non-replicable historical details
- omitted MA from this stage rather than treating an ex-post/walk-forward-selected crypto MA pair as an immutable source rule
- corrected Turtle V1 sizing fidelity: published Turtle Unit sizing makes 1N approximately 1% account equity, so a 2N hard stop is approximately 2% initial risk per first Unit
- preserved 2026 hard seal, 0.11% round-trip baseline cost, 2x/3x cost and +1m/+2m delay stress

Known disclosed adaptations:
- Zarattini 20% volatility rebalance threshold is frozen as 0.20 absolute portfolio-weight points because the paper wording does not further disambiguate units
- MOP uses raw ETH-perpetual 12M return as the tradable single-asset proxy for contract excess return and a disclosed 60D EWMA volatility replication convention
- Turtle mechanical entry/exit/N/Unit/pyramiding rules are replicated, but current live equity replaces the historical Turtle notional-account process because the published annual account reset was discretionary

Delivery validation:
- R03 dedicated tests: **12/12 passed**
- V1 + R02 + data_feed regression: **39/39 passed**
- combined targeted validation: **51/51 passed**
- Python compile / CLI smoke: **PASS**

## R04 — Turtle Path Atlas

Status: code delivered; awaiting local full-data run.

Purpose: stop changing entries and learn how the profitable but high-drawdown R03 Turtle actually evolves after entry. R04 reuses the exact source-locked R03 Turtle engine and reconstructs each completed episode on 1m bars. It reports MFE/MAE in N, time to pyramid stages, giveback, and early-path checkpoints. 2023-2024 are discovery; 2025 is validation; 2026 remains sealed. No R04 path statistic is permitted as a live feature until converted into a causal rule and separately validated.


## R04.1 runner contract hotfix

Status: **FIXED / LOCAL RERUN REQUIRED**

The first user local R04 launch failed before path reconstruction because `runner.py` referenced `baseline.equity`, but the source-locked engine's `BacktestResult` contract names the minute equity curve `minute_equity`. This was an R04 integration/test-coverage defect, not a data or strategy failure. The runner now passes `baseline.minute_equity`, and a runner-level regression test enforces that exact contract. No Turtle rule or path statistic changed.
