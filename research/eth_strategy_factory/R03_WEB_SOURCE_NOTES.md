# R03 web-source notes

Sources verified before implementation:

- Zarattini, Pagani, Barbon, *Catching Crypto Trends; A Tactical Approach for Bitcoin and Altcoins* (2025), SSRN 5209907 / author page. The paper specifies nine Donchian horizons (5/10/20/30/60/90/150/250/360 days), 90-day volatility, 25% volatility target, 2x cap and equal-weight ensemble. Its cost section specifies that the 20% rebalance threshold applies only to volatility-driven sizing changes, while signal entries/exits update immediately. The appendix specifies the symmetric long-short extension.
- Moskowitz, Ooi, Pedersen, *Time Series Momentum*, Journal of Financial Economics 104(2), 2012. Canonical TSMOM focuses on sign of the past 12-month return, one-month holding/rebalance, and 40% divided by ex-ante volatility sizing. The original paper is a diversified 58-market futures/forwards study; single ETH is a material adaptation.
- Curtis Faith / OriginalTurtles.org, *The Original Turtle Trading Rules* (2003). System 2 uses intraday breaks of the preceding 55-day high/low, N-based volatility units, 2N stops, 0.5N pyramiding from actual fills, a four-Unit single-market maximum, and a 20-day opposite-breakout exit. The rules also describe a historical notional-account management process whose annual adjustments were discretionary; that part cannot be exactly mechanically reproduced and is explicitly adapted rather than guessed.

Important fidelity boundaries:

- `source rule` means the public source supports the mechanical core.
- `required adaptation` means ETH perpetual / 24x7 calendar / project causal execution forces a change.
- No source result is treated as evidence that ETH-USDT-SWAP must reproduce the published return.
- No result from 2023-2025 is allowed to change R03 parameters. 2026 remains sealed.
