# Model Card — ETH Q70 Reclaim MF Long V1

## Purpose

Detect medium-horizon ETH-USDT-SWAP Long opportunities from causal multi-timeframe market-process features, enter immediately after a q70 signal, cut confirmed failures and retain occasional long winners through non-time structural exit logic.

## Data and decision cadence

- Base source: OKX `ETH-USDT-SWAP` public 1-minute Trade Bars through `src.data_feed`.
- Causal context: 1m, 5m, 15m, 30m, 1H, 4H and 1D features.
- Primary decision cadence: every completed 15 minutes.
- Execution path: next observable 1-minute open.
- 1-minute path also enforces exchange-side hard-stop execution.

## Model recipe

- Model family: frozen LightGBM opening-score model from the Long-context research pipeline.
- Fit period: `2023-01-01` through `2025-09-30 06:00`.
- Calibration period: `2025-10-01` through `2025-12-31`.
- Fit rows: `95,022`.
- Calibration rows: `8,832`.
- Frozen q70 score threshold: `-9.284925194833672e-05`.
- Feature schema SHA-256: `51b2003aea4d9c30baa9f74af3e7760d14b86096973154e27894866744a7183c`.
- Base model parameters preserved in the sealed report manifest: 420 estimators, learning rate 0.035, 31 leaves, minimum child samples 300, random seed 20260801.

The score was an opening selector only. It was never validated as a holding-renewal, add-on or risk-sizing signal.

## Frozen execution contract

```text
score >= frozen q70 threshold
→ enter at next observable 1m open
→ equal account risk across all q70 score tiers
→ real 2% exchange-side hard stop
→ after 1.5% adverse excursion, use completed 15m soft-failure confirmation
→ otherwise hold until deterministic failed_reclaim exit
→ no fixed take profit
→ no fixed-time final exit
→ no add-on, averaging down, split entry, Turtle or pyramid
```

## Intended role

A conditional medium-horizon Long sleeve, not an all-weather portfolio and not a full-cycle trend model. The historical median holding time was around fourteen hours, with roughly twenty account trades per month in 2024–2025.

## Final limitations

- Absolute q70 calibration drifted from approximately 30% exceedance in calibration to 58% in 2026 H1 and about 70% in Q2/July.
- High scores did not preserve stable cross-year ranking.
- Simple Bull/Bear gates did not explain or repair the sealed failure.
- July returns depended on the non-time exit overlay and concentrated winners while broad six-hour opening expectancy remained weak.
