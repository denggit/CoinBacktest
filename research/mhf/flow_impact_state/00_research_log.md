# OKX Flow–Impact State Strategy — Research Log

- Edge ID: `ETH_MHF_FLOW_IMPACT_STATE`
- Goal: build a long-history, OKX-only MHF strategy family capable of supplying a future main copy-trading sleeve.
- Hard frequency target for a finished strategy: 40–90 trades/month, 1–3 trades/day, active-date ratio >=65%, longest no-trade gap <=3–5 days.
- Round 01 status: `research_pending`

## Scope boundary

The family studies one recurring process:

```text
aggressive buy/sell pressure forms
-> price and liquidity respond
-> pressure either advances effectively or becomes exhausted
-> continuation or reversal branch
```

Round 01 uses long-history rich OKX trade bars only. It does not use a 4H hard gate, static wall entry, TP/SL optimisation, Liquidity primitives, liquidation estimates, funding, OI, portfolio composition or machine-learning selection.

## Round 01 — Pressure event atlas

### Research question

After aggressive signed notional becomes historically abnormal, how do causal price-response states change continuation, reversal, first-touch and pressure-duration outcomes?

### Event definition

- Rolling pressure windows: 1, 3 and 5 closed bars.
- Pressure magnitude: log absolute rolling signed notional, normalized against a historical baseline ending before the complete current pressure window.
- Event onset: pressure-z crosses the default 1.5 threshold, or signed pressure direction flips while pressure remains above it. The threshold remains configurable and is not selected from forward returns.
- No price-direction requirement is used to create the event.
- A short per-window cooldown suppresses threshold chatter.

### Causal timing

- Local `OKXTradeBarLoader`, cache-only.
- Signal bar is fully closed.
- Entry/path origin is the immediate next bar open.
- Missing calendar buckets are regularized only for time alignment; any event whose pressure, entry or full path depends on a synthetic row is excluded.
- Post-flow, pressure duration, MFE/MAE and first-touch fields are outcomes only.

### Required output

- Event frequency and active-date coverage.
- Symmetric continuation and reversal fixed-horizon returns.
- Fee-only and conservative normal-cost results.
- MFE/MAE and conservative same-bar dual-touch classification.
- Price-response state comparison.
- Pressure-strength comparison.
- Annual/monthly stability.
- Pressure-state duration and post-event flow persistence.
- Causal/data audit and deterministic event sample.

### Decision boundary

Round 01 cannot promote a strategy. It may only:

- reject the broad family;
- retain one causal mechanism difference for Round 02;
- or show that the long-history foundation is strong enough to justify later 5s/Liquidity incremental validation.

## Round 01 result — 2026-07-25

- Valid pressure-window events at `min_pressure_z=1.5`: 141,903; unique pressure processes: 66,423.
- Frequency calibration selected `min_pressure_z=2.0` for R02 because it preserves roughly 16 unique pressure processes/day without using outcomes.
- No response-state cell achieved >=1,000 events, positive normal-cost expectancy, PF >1 and at least three positive years.
- The only normal-cost-positive pressure-strength cell had only 180 events and was rejected by the user's sample-size rule.
- Broad descriptive pattern: sell pressure showed a small short-horizon rebound tendency, but the gross effect was only a few basis points and did not clear costs.
- Decision: `research_continue` only for strict conditional discovery; no TP/SL backtest.

## Round 02 — Conditional edge discovery

### Frozen question

Can relative participation, large-flow consistency, pressure persistence, or impact efficiency isolate a broad positive-expectancy subset while retaining at least 1,000 events and 40–90 events/month?

### Anti-overfit design

- Event threshold fixed at `pressure_z >= 2.0` from R01 frequency calibration.
- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-09-30.
- Untouched holdout: 2025-10-01 through 2026-06-30.
- Discovery-only quantile thresholds.
- Single-feature cumulative tails first.
- Pairwise search only among discovery-frozen single conditions.
- Maximum two causal features.
- Benjamini-Hochberg correction on discovery monthly tests.
- No Books, TP/SL, ML classifier, maker-fill assumption, session hard gate or 4H hard gate.

### Stop rule

If no condition survives >=1,000 events, normal costs, validation, holdout and frequency gates, stop adding 1m environment filters. The next and only justified branch is raw-trade/5s impact-efficiency decay; Books may later be tested only as an incremental overlay on its available history.

## Round 02 result — 2026-07-25

- Valid window events: 40,400; unique pressure processes: 20,661.
- Single-variable conditions scanned: 3,420.
- No single condition was normal-cost positive in discovery, so pair search correctly remained empty.
- Highest broad gross uplift came from high activity/relative pressure, but gross expectancy stayed around 3–8 bps versus 15 bps conservative round-trip cost.
- No >=1,000-event condition survived discovery, validation and holdout.
- Decision: stop adding ordinary 1m environment filters.

## Round 03 — Accumulated pressure + causal Price Action

### Frozen question

Does multi-bar accumulated taker pressure become tradable only after price confirms one of two structural processes?

```text
accumulated pressure -> old swing sweep -> reclaim -> exhaustion reversal
accumulated pressure -> old swing break -> retest holds -> continuation
```

### Design

- Accumulation windows: 5, 10 and 20 closed 1m bars.
- Accumulated pressure is rolling net taker notional, normalized only against prior history.
- Marginal impact compares early-half versus late-half price movement per directional million USDT.
- Price Action uses causally confirmed swing highs/lows; a pivot becomes usable only after `right + 1` later bars.
- Entry occurs at the next open after a closed reclaim/resume bar.
- Stop comes from the attack/retest invalidation structure plus a small buffer.
- Target is the nearest already-known structure; continuation may use a measured-move fallback if no known target exists.
- No fixed TP/SL grid and no optimized strategy time exit.
- A 240-bar safety timeout is explicit and a candidate fails if timeout share exceeds 10%.
- Candidate sample and split rules remain hard: >=1,000 conflict-resolved trades, discovery/validation/holdout each net positive.

### Stop rule

If R03 produces no broad, cross-split, normal-cost-positive PA process, stop studying 1m entry timing for this family. Move only to 5s/raw-trade dynamic accumulation and impact-decay confirmation; do not create R04 by adding more 1m filters.
