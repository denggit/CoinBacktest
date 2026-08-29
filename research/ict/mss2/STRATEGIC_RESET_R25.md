# Strategic Reset after R23–R25

Date: 2026-08-17

## 1. What economic edge is actually proved?

None. R23's frozen panic-wick Long prior fails validation and top-ten removal.
R24's scheduled funding-window unwind has no gross edge. R25's r0020 run-
exhaustion reversal is a near-fair gross process whose 2x PF is only 0.38–0.42.
No sleeve is eligible for portfolio construction or holdout opening.

## 2. What is descriptive rather than tradable?

- Panic wicks can produce a concentrated discovery right tail, but the edge
  does not persist into 2025H1.
- Large moves occur around canonical funding times, but the clock without
  funding sign or positioning state is not predictive.
- Same-direction r0020 runs and opposite bars create many rapid paths, but the
  event does not distinguish terminal exhaustion from continued churn.
- Range-Bar formation frequency and duration change materially across market
  regimes; activity remains a volatility descriptor, not direction.

## 3. Strongest current candidate

There is no valid candidate. R23 is the strongest-looking visible prior because
its discovery 2x PF is 1.67, but validation PF is 0.96 and discovery becomes
negative after removing the top ten winners. R24 and R25 fail at gross-edge
formation before robustness is considered.

## 4. Candidate KPI reality

| Study | Visible 2x PF | Frequency / coverage | Stability | Status |
| --- | --- | --- | --- | --- |
| R23 panic-wick Long | 1.67 discovery / 0.96 validation | 5.0 / 18.5 trades per month | top-ten discovery sum -4.52%; 2025 loses | rejected |
| R24 funding-window R1 | Long 0.70/0.72; Short 0.70/0.37 | 4–5 trades per month | every visible year loses | rejected |
| R25 r0020 run reversal | Long 0.41/0.42; Short 0.39/0.38 | 190–371 trades per month per side | every month, quarter, and year loses | rejected |

No holdout KPI exists because no study earned an opening.

## 5. Directions permanently stopped

- Panic-wick session/flow/volatility/target/delay/exit rescue.
- Scheduled funding-clock z/ATR/target/hold/session rescue without actual
  funding-state history.
- Range-Bar activity as directional continuation.
- r0020 same-direction run exhaustion, alternate scales, run lengths, duration
  acceleration, flow, footprint, target, stop, or confirmation filters.
- ML or ranking filters on any of these gross-null families.
- Previously archived completed-trend reversal/acceptance, simple OI
  release/rebuild, daily channel, BTC catch-up, and fixed impulse families.

## 6. Filter-rescue warning

The current cycle stopped all three rules at their frozen gates. The risk now
is conceptual rather than implemented: R25 exposes many tempting duration,
delta, run-length, and session cuts. Using them to rescue a gross PF near one
would be outcome-conditioned feature mining and is prohibited.

## 7. Largest gap to the final portfolio

The primary gap is still **edge strength**. R25 supplies far more frequency and
coverage than required, yet every marginal trade loses after costs. R23 shows
that a discovery headline without validation and winner resilience is not an
edge. Frequency, flat periods, Long/Short balance, and position management are
secondary until one mechanism has positive stressed-cost expectancy in both
visible splits.

## 8. Gap classification

- Edge strength: critical.
- Cost sensitivity: critical, especially for high-frequency small-risk paths.
- Winner dependence: critical for panic-wick and historical trend priors.
- Data/regime state: important; funding/basis and some microstructure histories
  are too short or temporally heterogeneous.
- Frequency/coverage: not the bottleneck demonstrated by R25.
- Exit/position management: not the current bottleneck because entry-level path
  ordering is already near fair.

## 9. Next phase

Do not deepen R23–R25. Before R26, audit whether a full pre-embargo OKX ETH spot
series exists through `src.data_feed` with exact timestamp overlap to the swap.
If it does, a genuinely distinct candidate is spot-led perpetual dislocation:
completed spot movement or spot/swap divergence, then next-minute swap
convergence with a frozen structural invalidation. This is an observable
cross-venue price-discovery mechanism, not another filter on a failed ETH-only
event. It must be abandoned before coding if source coverage is incomplete or
if repository history shows the same mechanism was already tested.

If spot/perpetual history is unavailable, do not substitute short 2026 funding,
basis, books, or mark data. Return to source inventory and mechanism discovery
rather than another price-feature threshold.

## 10. Are we moving toward stable profit?

Falsification quality improved: R25 was frozen, source-audited, exact-replayed,
and stopped despite abundant sample. Economically, however, the cycle produced
no progress toward a profitable sleeve. Continuing becomes research-for-
research's-sake unless the next study introduces independently observable state
with a clear mechanism and complete pre-holdout history.

## Post-reset source gate for R26

The proposed spot-led perpetual-dislocation study fails before coding. Read-only
`src.data_feed.OKXDataLoader.load_local_data()` checks return zero rows for
`ETH-USDT`, `ETH-USDC`, `ETH-USD-SWAP`, and `BTC-USDT`. No spot proxy, remote
download, or short later-period basis substitute is allowed.

Other OKX-local microstructure sources also fail the visible-split gate:

- `OKXDerivativesLoader.coverage()` reports funding and mark-price history only
  for June 2026 and no liquidation rows. Its only pre-embargo contract-OI
  series is one daily row from 2024-01-01 through 2025-06-30 (547 visible rows),
  leaving the 2023 discovery year absent.
- A filename-only `OKXBooksLoader` inventory, without opening any archive,
  finds zero days before 2025-07-01. The 400-level files begin in January 2026
  and the 5000-level files begin in November 2025.

The complete remaining independent lane is Binance USD-M 5-minute futures
metrics through `src.data_feed`. Visible 2023--2025H1 top-trader position and
global-account ratios are effectively complete, causally published one minute
after their source timestamp, and were explicitly excluded from R18/R19. A
repository search finds them aligned or profiled only as post-sweep context;
no standalone top-trader-versus-global leadership-cross strategy exists.

Therefore the only justified R26 candidate is a separately precommitted
positioning-leadership mechanism: a fresh cross in top-trader position share
relative to global-account share, followed by same-direction completed-price
confirmation and next-minute OKX execution. It is not an OI-transition rescue:
base OI magnitude/change, sweep state, taker flow, funding, and learned filters
remain excluded. If this unfiltered mechanism fails the visible split, the
ratio branch stops.
