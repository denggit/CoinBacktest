# R17 — Trend Pullback Reclaim / Re-acceleration Path Atlas

Date: 2026-08-16

## Overall assessment: ready to share as a rejection

R17 tested the independently precommitted continuation sequence:

```text
aligned causal 1D + 4H HH/HL or LH/LL state
→ confirmed 30m counter-trend pivot
→ 15m pivot-range reclaim close
→ later 5m reclaim-bar break close
→ next observable 1m open
```

It did not reuse q70, the R13/R14 rule families, completed-trend sweep admission, or the earlier Higher-Low resting-limit entry. The event contract, 0.25× 30m ATR stop buffer, 1.50% maximum stop distance, 12-hour setup expiry, 72-hour diagnostic horizon, and 1R/2R/3R plus 4H structural target were frozen in `R17_PRECOMMITMENT.md` before outcomes were computed.

## Source, splits, and sample

- Market: OKX `ETH-USDT-SWAP` bare 1m K loaded only through `src.data_feed`.
- Requested data: 2022-01-01 through 2026-08-15 23:59:59.
- Discovery entries: 2023-01-01 through 2024-12-31.
- Validation entries: 2025-01-01 through 2025-06-30.
- July 2025 embargoed.
- Holdout begins 2025-08-01 and remains sealed; 796 aligned holdout pullbacks were counted, with zero holdout outcome rows.

The visible funnel contained 1,668 aligned pullbacks, 977 15m reclaims, 640 later 5m re-accelerations, and 422 executable setups after stop/runway checks. Discovery had 227 Long and 144 Short entries; validation had 35 Long and 16 Short entries.

## Primary result

| Direction / target | Discovery 2x PF | Validation 2x PF | Discovery top-5 removed | Validation top-5 removed |
| --- | ---: | ---: | ---: | ---: |
| Long / 4H structural | 0.60 | 0.61 | 0.49 | 0.16 |
| Long / 1R | 0.48 | 0.39 | 0.43 | 0.19 |
| Long / 2R | 0.57 | 0.46 | 0.49 | 0.14 |
| Long / 3R | 0.58 | 0.64 | 0.48 | 0.16 |
| Short / 4H structural | 0.66 | 0.25 | 0.47 | 0.00 |
| Short / 1R | 0.43 | 1.09 | 0.36 | 0.45 |
| Short / 2R | 0.65 | 0.97 | 0.52 | 0.25 |
| Short / 3R | 0.71 | 0.47 | 0.55 | 0.00 |

Every Long target is negative in 2023, 2024, and validation. Short discovery is negative for every target. The isolated validation Short 1R PF of 1.09 has only 16 trades, only 33.3% positive months, becomes unprofitable at 3× cost, and falls to PF 0.45 after removing five winners. It is sampling noise, not a sleeve.

Median stop distance is roughly 0.71%–1.04%, so R17 is not failing because costs exceed an artificially tiny stop as in R15. The confirmed entry itself has no continuation edge: target rates are too low and losses remain broad across years.

## Validation and calculation spot-checks

- 1,688 path rows = 422 unique setups × four targets; no duplicate setup/target rows.
- Independent brute-force raw-bar replay through `src.data_feed` found zero first-passage ordering discrepancies across all 1,688 paths.
- Recomputed gross-return and 2×-cost formulas differ from saved values by less than `3.2e-16`.
- Independently recomputed grouped PF differs from the scorecard by less than `2.7e-15`; trade counts match exactly.
- Thirteen causal checks have zero violations.
- Same-bar target/stop is stop-first, and seven focused R17 tests cover timing, future mutation, cost, direction separation, and audit behavior.
- Full MSS2 regression after R17: 132 passed, 14 deselected; the only warning is the existing unwritable pytest cache.
- Import-boundary scan still reports 164 historical unexpected violations and zero under `research/ict/mss2` or R17.

One non-headline Short MFE diagnostic initially used reciprocal price movement and slightly overstated favorable excursion. It was corrected to linear return before the final report rerun. Entry, exit, PF, cost, and first-passage results were unaffected.

## Frozen conclusions

1. The exact aligned 1D/4H structural-state → 30m pivot → 15m reclaim → 5m re-acceleration market-entry sequence has no economic edge.
2. Long and Short both fail discovery; validation cannot rescue a branch with negative discovery.
3. The 16-trade Short 1R validation cell is rejected by sample size, cost stress, month breadth, and top-winner dependence.
4. Do not tune pivot order, ATR buffer, stop ceiling, setup expiry, or reclaim thresholds to rescue R17.
5. Do not convert the 72-hour diagnostic horizon into a claimed final exit.
6. No strategy is promoted; portfolio metrics remain premature.
7. Existing MSS2 holdout remains sealed, and any future live approval still requires genuinely new forward data because other repository projects inspected overlapping 2025–2026 history.

## Next independent hypothesis audit

Trend-pullback continuation is not archived as a universal market concept, but this clean price-only structural sequence is closed. A first proposal—failed-auction range re-entry—was rejected before R18 coding because the repository has already tested it in `eth_market_process_portfolio/integration/R02`. Its base/loose/strict definitions all failed: the base family produced 452 trades at PF 0.34, 2×-fee PF 0.12, negative returns in every year, and negative return after removing the top ten. Repeating it in MSS2 would be renamed duplication.

The next genuinely independent boundary is therefore positioning unwind rather than another price-only confirmation stack: causal Binance 5m OI state aligned through `src.data_feed` to OKX ETH price, testing whether price/OI expansion followed by OI release and price stabilization creates asymmetric continuation or reversal paths. It must be independent of sweep admission, keep Long/Short separate, begin with a small mechanism/path atlas, and use no oracle turning-point label as an entry feature. This is a proposed R18 boundary, not a precommitted or promoted strategy.

## Primary evidence

- `data/reports/research/ict/mss2/r17_trend_pullback_reacceleration_atlas/04_setup_funnel.csv`
- `07_direction_target_scorecard.csv`
- `08_direction_target_years.csv`
- `09_causal_audit.csv`
- `manual_review/`
