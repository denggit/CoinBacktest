# R05 Research Notes — Entry Timing × Structural Stop × Runner

## Why R05 exists

R04 confirmed that the strongest Long liquidity-exhaustion cohorts can sustain a 5% diagnostic target across years, but it also exposed a structural mismatch: the median R04 episode-extreme stop for `N>=4 + 4H` is about 1.52%. That may be reasonable as a *final thesis invalidation* for a multi-day major reversal, but it is obviously mismatched to a 0.3%-0.75% rebound objective.

The user also clarified two execution habits that R05 must respect:

1. 1m/2m can be useful for *entry* because waiting for a 5m reclaim after a deep sweep can be late.
2. Trailing-stop structure should not be driven by noisy 1m swings. A new stop should be promoted on higher-quality 2m/5m/15m structure, especially a causally confirmed ITL/LTL, or after an unusually strong bullish displacement / FVG bar where the low of that completed bar may become a useful risk anchor.

R05 therefore separates **entry timeframe**, **initial invalidation**, and **runner trailing** instead of giving every horizon the same stop.

## Frozen entry comparison

For each hierarchy rule, the first qualifying liquidity stage is selected once per episode, independently of execution timeframe. The exact same `stage_id` is then compared across 1m / 2m / 5m `episode_reclaim` entries. This avoids comparing different liquidation checkpoints and calling the result a timeframe effect.

Frozen descriptive quality rules:

- `N>=3 + (4H or LT)`;
- `N>=4 + (4H or LT)`;
- `N>=3/4 + 4H`;
- `N>=3/4 + LT`;
- `N>=2 + IT+ + structural-key` as a broader short/medium-horizon diagnostic universe.

No rule is tuned in R05.

## Existing-report timing sanity check (not new holdout evidence)

Using the already-observed R02/R03.3 report only:

- `N>=4 + 4H`: 1m reclaim median sweep->entry ~38.5m vs 5m ~47m;
- the same episode-extreme stop still leaves median risk ~1.48% on 1m vs ~1.52% on 5m;
- `N>=3 + (4H or LT)`: 1m median sweep->entry ~27.5m vs 5m ~36m, with episode-stop median risk ~1.13% vs ~1.23%.

Therefore earlier entry helps, but **entry timeframe alone cannot solve the wide-stop problem** if every version still anchors to the full episode extreme.

## Structural initial-stop atlas

R05 compares entry-time-only structural candidates:

- `episode_extreme`: existing R02 final thesis stop;
- `qualifying_stage_extreme`: the sweep extreme of the causal stage that triggered the rule (never the future final episode stage);
- `reclaim_leg_extreme`: low from the qualifying sweep bar through the bars completed before entry;
- `signal_bar_extreme`: low of the completed reclaim signal bar;
- latest causally known `2m / 5m / 15m ITL-or-LTL` formed since episode start, when available below entry.

No percentage stop grid is introduced.

For +0.5/+0.75/+1/+2/+3/+5% diagnostic targets, R05 reports target-before-stop, 2x-cost PF/expectancy, and **MAE before the target**. The MAE table is specifically intended to answer how tight a short-rebound stop could be without using post-target drawdowns.

Same-bar target/stop is pessimistically stop-first.

## Causal 2m/5m/15m swing hierarchy for trailing

R05 reuses the recursive ICT hierarchy semantics:

- ST = confirmed base swing;
- IT = swing-on-swing relation over ST;
- LT = swing-on-swing relation over IT.

ITL/LTL labels are only usable after the right-side confirming swing itself is known. Eventual IT/LT class is never backfilled.

**1m trailing is forbidden in code.**

## Runner trailing atlas

Runner simulations begin with the original episode-extreme thesis stop so that trailing rules can be isolated. Stops only ratchet upward.

Frozen structural variants:

- 2m ITL/LTL;
- 5m ITL/LTL;
- 15m ITL/LTL;
- 5m LTL only;
- 15m LTL only.

A structure confirmed at time `T` is first active on the 1m bar starting at/after `T`. It cannot retrospectively stop the trade inside the bar that created/confirmed it.

## Strong bullish displacement / FVG anchors

The user explicitly described a case where ETH has been rising in ordinary 0.1%-0.2% small bars and then suddenly prints an exceptionally strong small-timeframe bullish displacement, e.g. ~1% in one bar. The low of that completed displacement bar may be a better stop anchor.

R05 does **not** hard-code `1% = strong`. It builds causal rolling 7-day bullish-body percentiles on 2m/5m/15m bars and keeps q90/q95/q99 as research bins. It also records absolute body-size buckets (<0.3%, 0.3-0.5%, 0.5-0.75%, 0.75-1%, >=1%) and whether that shock bar itself formed a bullish FVG.

Runner variants include:

- 2m / 5m / 15m q95 shock low;
- 5m q99 shock low;
- q95/q99 shock **with bullish FVG** variants;
- 15m ITL/LTL OR 5m q95 shock;
- 5m/15m ITL/LTL OR 5m q95 shock.

These are research variants, not promoted thresholds. The purpose is to learn whether the strongest displacement bars are good *trailing anchors*, and whether “stronger” is actually better.

## No fixed TP promoted

R04's 5% result remains an important right-tail benchmark, not the R05 final TP. Runner research exits on structural trailing stops; 14 days is censoring only. Reports retain whether 3%/5% was reached before the structural trail stopped the position and how much MFE the trail captured.

The future strategy may eventually combine short partial realization + structural runner, but R05 still avoids optimizing partial percentages.

## R04 performance warning fix

The user's real R04 run emitted `DataFrame is highly fragmented` at the yearly scoreboard `.loc` step. Root cause was repeated slicing of the full ~329-column future-label frame. The cumulative R05 patch fixes R04 by projecting only the scoreboard's required outcome columns, materializing one compact contiguous frame once, and then performing rule/year slicing on that compact table.

The real R04 report was replayed through the corrected scoreboard in the delivery environment: 11 rule rows + 44 rule-year rows, zero `PerformanceWarning`, ~0.28s.

## R05.1 — mutually exclusive opportunity-range reports

User feedback identified an important reporting ambiguity: a trade that eventually reaches +5% also appears in the nested +0.5% / +1% success tables. That is correct for transition / upgrade-probability research, but it contaminates any attempt to describe what a genuinely short-lived rebound looks like.

R05.1 therefore keeps the original nested target atlas **and adds a separate mutually-exclusive future-outcome atlas**. Using the frozen episode-extreme thesis stop and conservative same-bar stop-first semantics, each resolved opportunity is assigned exactly one future MFE bucket before thesis invalidation / complete 14-day censor:

- `<0.3%`;
- `0.3%-1.0%` short rebound;
- `1.0%-3.0%` medium reversal;
- `3.0%-5.0%` swing;
- `>=5.0%` major long tail.

A stop-touch bar is excluded from attainable MFE so a same-bar high cannot promote a stopped trade into a higher bucket. Opportunities at the right edge without a full 14-day window and without thesis-stop resolution remain `right_edge_incomplete` and are excluded from resolved bucket summaries.

These buckets are **future labels only** and are never merged back into causal entry features. They exist only to make short/medium/swing/major path diagnostics independent. The original nested tables remain because they answer a different question: conditional upgrade probabilities such as `P(3% | already reached 1%)`.

R05.1 writes a top-level bucket atlas and independent directories under `opportunity_buckets/`, each with its own overview, yearly breakdown, initial-stop/target summary, MAE-before-target summary, structural-trailing summary, and trade rows. This prevents >=5% winners from inflating short-rebound statistics while preserving the upgrade-ladder analysis.

## R05.2 — Initial-stop atlas algorithmic speedup — 2026-08-15

A real full-history run appeared to stall immediately after `[r05] structural initial-stop atlas`, before the downstream `[r05-initial-stops]` progress reporter appeared. Profiling identified the hot path in `attach_initial_structural_stops`: for every opportunity and each 2m/5m/15m hierarchy, the old `_latest_known_low` implementation copied the full low-pivot DataFrame, rebuilt IT/LT candidate frames, concatenated them, filtered by causal availability, and sorted again.

R05.2 replaces that repeated DataFrame work with a compact `_KnownLowLookup` built once per timeframe. Candidate IT/LT low rows are materialized into NumPy arrays sorted by `(pivot_time, class_available_time)`; each opportunity query uses `searchsorted` bounds plus a short backward scan. This preserves the exact causal semantics, including IT -> LT upgrades only after the LT availability time and the episode-local `min_pivot_time` lower bound.

Synthetic equivalence benchmark (3,000 hierarchy rows, 500 queries): old DataFrame path ~2.84s vs indexed path ~0.0026s, ~1,100x faster for the lookup hot spot with identical selected price/class results. Full-run speedup will differ because other R05 stages remain.

The initial-stop target atlas was also de-duplicated algorithmically: target first-touch and MAE-to-target are independent of stop variant, so they are now computed once per opportunity/target and reused across structural stop candidates. With 7 stops x 7 targets, first-touch threshold queries fall from up to 56 per opportunity to 14 (7 stop + 7 target), while repeated MAE range queries are similarly cached.

A new `[r05-stop-anchors]` progress reporter now covers the structural-stop attachment stage itself, so long runs no longer appear silent before `[r05-initial-stops]` begins. Stop/risk columns are materialized in one DataFrame concat rather than iterative insertion to avoid fragmentation.

Validation after R05.2: 52 targeted ICT-MSS2 tests passed, including a new causal equivalence test for IT-to-LT upgrade timing and episode-local pivot bounds. No strategy, threshold, entry, stop, target, or causality semantics were changed; this is an engineering/performance-only update.
