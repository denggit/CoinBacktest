# R02 — Continuous Risk-Managed ETH Portfolio

## Goal

Stop treating ETH as a sequence of isolated entry/TP/SL trades. Maintain one causal **target net ETH exposure** continuously and trade only the difference between current and target exposure.

Exchange semantics are frozen as:

```text
internal virtual sleeves may disagree
        ↓
portfolio aggregator nets them
        ↓
one ETH target exposure in [-1.5x, +1.5x]
        ↓
no simultaneous long + short exchange legs
```

## Why R02 exists

V1 external-strategy tournament showed the simplest trend systems were the only credible winners:

- Turtle System 2: +42.19%, PF 2.17, MDD 21.40%
- MA20/50 volatility trend: +27.60%, PF 1.52, MDD 24.13%
- Donchian ensemble long: +26.66%, PF 3.63, MDD 12.54%
- BB breakout 4H: +7.53%
- Donchian long/short: +3.19%

The footprint/CVD/quarter-hour event systems lost heavily. R02 therefore tests whether simple trend information becomes more useful when it controls exposure continuously instead of producing isolated trades.

## Frozen signal layer

No parameter search is performed. Four signal families receive equal portfolio weight.

### 1. Channel family

- Donchian states at 5/10/20/30/60/90/150/250/360 daily lookbacks.
- Turtle-style 55-day entry / 20-day opposite-channel exit state.
- Family score = mean of Donchian ensemble and Turtle state.

### 2. Moving-average family

- Daily SMA20 / SMA50 direction.
- Daily SMA50 / SMA200 direction.
- Family score = mean.

### 3. Time-series momentum family

Own-return sign over:

- 21 days
- 63 days
- 126 days
- 252 days

Family score = mean.

### 4. Faster 4H trend family

- Supertrend(ATR10, multiplier 3)
- Keltner breakout(EMA20, 2 ATR20)
- ADX14 >= 25 with EMA50 direction

Family score = mean.

Final raw directional score is the equal mean of the four family scores and therefore remains in [-1, +1].

## Frozen risk layer

- 90-day realized volatility.
- 25% annualized volatility target.
- absolute net exposure cap: 1.5x.
- rebalance deadband: 0.10x.

Three pre-registered component-ablation specs:

1. `CP01_CORE_VOL`: volatility target + deadband.
2. `CP02_DD_GOV`: CP01 + drawdown governor.
3. `CP03_DD_GOV_SMOOTH`: CP02 + max 0.50x change per rebalance.

Drawdown governor uses equity known before the rebalance:

```text
DD < 5%      -> 1.00x target
5% <= DD<10% -> 0.75x
10%<= DD<15% -> 0.50x
DD >=15%     -> 0.25x
```

These values are frozen before the full-data R02 result is observed. They are not tuned against 2023-2025 losses.

## Causal execution

All daily and 4H bars are available only after the bar closes.

A target event at `available_time` is executed at the **next observable 1m open**. Cost is based on actual turnover:

```text
+0.80x -> +0.35x : turnover 0.45x
+0.35x -> -0.25x : turnover 0.60x
+1.00x -> -1.00x : turnover 2.00x = one complete round trip
```

Default round-trip cost is 0.11%.

## Evaluation

- Warmup: 2022-01-01.
- Research: 2023-01-01 through 2025-12-31.
- 2026 remains sealed.
- Base / 2x cost / 3x cost.
- +1m / +2m execution delay.
- top 1/5/10 positive daily return dependency.
- annual/monthly results.

Continuous-position operating metrics replace trade-count obsession:

1. maximum consecutive time abs(net exposure) <= 0.05x
2. maximum consecutive time abs(net exposure) <= 0.25x
3. maximum consecutive losing mark-to-market days
4. minute mark-to-market MDD
5. CAGR
6. total return
7. turnover and average absolute exposure

## Stop rule

If all three frozen R02 variants fail, do not parameter-mine MA/Donchian/Supertrend settings. Change the portfolio/risk construction or introduce a new externally specified simple strategy family.
