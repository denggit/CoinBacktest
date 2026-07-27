# Flow–Impact R02 Web Research Notes

External research was used to constrain the hypotheses before writing R02, not to claim that an equity/Binance result automatically transfers to OKX ETH.

## Primary sources and implementation consequences

### The Price Impact of Order Book Events — Cont, Kukanov, Stoikov — arXiv:1011.6402

Order-flow imbalance explains short-horizon price changes more robustly than trade volume alone, and impact varies inversely with depth.

R02 consequence: relative volume is tested, but it cannot be treated as the main mechanism. Directional participation and impact efficiency remain separate features. Books are reserved for a later incremental depth-state test.

### A Million Metaorder Analysis of Market Impact on the Bitcoin — Donier, Bonart — arXiv:1412.4503

Bitcoin market impact is nonlinear and impact from uninformed flow can decay substantially.

R02 consequence: no linear absolute-volume threshold. The scan uses historical ratios/z-scores and tests both high-impact continuation and low-impact/exhaustion tails.

### To Make, or to Take, That Is the Question — Albers et al. — arXiv:2502.18625

Taker strategies must overcome taker fees; maker strategies face fill uncertainty and adverse selection.

R02 consequence: qualification is based on conservative taker-like normal costs. R02 does not assume a maker fill or use lower maker fees to rescue weak signals.

### When Does Order Flow Matter? State-Dependent L2 Liquidity-State Transitions in Crypto Futures — Jeon — arXiv:2607.09230

The incremental value of order flow is state- and asset-dependent; in the reported setup it is an overlay on an L2 baseline rather than a universal replacement.

R02 consequence: ETH is evaluated independently, and Books/L2 will only be added after a long-history trade-flow mechanism survives. The Books period will be used for incremental comparison, not to backfill unavailable history.

### The Quarter-Hour Effect — Kim, Hansen — arXiv:2607.09426

Crypto volume and order-flow bursts cluster at standardized clock boundaries and may carry different information by clock phase.

R02 consequence: clock phase is emitted as a diagnostic table only. It is explicitly forbidden from selecting or freezing candidates in R02, preventing another high-dimensional environment search.
