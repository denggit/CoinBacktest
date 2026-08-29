# R07 Research Notes — ICT Family Expansion

## Why R07 exists

The real R06 run preserved roughly 9–10 Long-reversal opportunities per month and retained a positive long-run edge, but 2x-cost equity quality remained inadequate for a standalone deployment: a multi-year drawdown, weak positive-month rate, and heavy dependence on a few right-tail winners. The response is **not** another chain of hard entry filters. R07 asks whether other causal ICT-style mechanisms can add independent opportunities and fill regimes where the SSL-exhaustion Long family is inactive.

## Frozen cautions

- `BSL sweep -> immediate Short` is not an ICT reversal model and is not tested as a trading rule.
- Existing confirmed BSL reversal variants from R02/R03.3 are re-audited across near and far targets before new mechanics are judged.
- A small FVG corridor must use a resting limit. Market entry at a completed signal close is intentionally unsupported.
- Small corridor economics must be studied with a local structural invalidation candidate as well as the wide original thesis stop.
- FVG by itself is not treated as edge. Direction must already be causally established for the corridor family; continuation must first show close-through acceptance beyond consumed liquidity.
- No NY Open / equity-index timing prior is used.

## Families

### A. Proper BSL/SSL reversal confirmation audit

Actual R02/R03.3 entries only: episode reclaim, structural MSS market, structural MSS+FVG limit, and post-sweep-ST MSS. Reports compare nearest liquidity, liquidity pools, 1H/4H/1D targets, and fixed-R controls. This prevents the false conclusion that a reversal has no edge merely because a distant 4H objective is a poor exit.

### B. Liquidity expansion continuation

A key-liquidity stage is followed by a close **through** the consumed boundary rather than rejection back inside. A directional FVG must then form. Entry is a resting limit at proximal or CE, with episode-structural and local FVG invalidation stops kept as separate variants. Both bullish and bearish continuation directions are studied.

### C. Confirmed reversal FVG corridor

After an episode reclaim has already established direction, R07 waits for the first same-direction FVG and places proximal/CE limits. It compares:

- original reclaim structural stop;
- local FVG-invalidation stop.

The small target is the nearest opposite-direction FVG that is already active, still unrebalanced, and still **ahead of current price when the resting order is placed**. If that frozen objective is delivered before the limit fills, the order is stale and removed. Structural-liquidity / fixed-R targets are reported separately.

## Complementarity

R07 monthly activity is compared with the R06 base family. Nested quality rules and execution sensitivities are deduplicated to one earliest family/episode opportunity before frequency and same-hour overlap are calculated. These are opportunity-complementarity diagnostics only; R07 does not combine unvalidated PnL into a fake portfolio curve.

## What would justify continuation

A new family is worth deeper capital research only if it has:

1. enough independent episodes to matter;
2. 2x-cost positive expectation with cross-year support;
3. economics that survive the correct maker/taker cost model for limit entries;
4. no causal-audit violations;
5. meaningfully different monthly/hourly activity from R06, especially in its long underwater regimes.

## R07.1 operational update — 2026-08-16

- New/future MSS2 default end date is `2026-08-15 23:59:59`.
- R07 now refuses mixed-window execution when R02/R06 persisted reports are older than the requested end date, and it verifies that R03.3 hierarchy contains every R02 stage id.
- Added `tools/prebuild_okx_ohlcv.py`, a thin wrapper around `src.data_feed.OKXDataLoader`, so naked 1m K can be cached before a full chain run without putting market-data logic into research code.
- Current R07 dependency chain requires naked 1m OHLCV and 1m Trade Bars only. Range footprints/books/OI are not default prebuild dependencies.
- Added `manual_review/` outputs with the most recent resolved executable examples so manual K-line validation can be done file-first rather than by pasting dates into chat.
