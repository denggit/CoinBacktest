# Flow-Impact R03 — External research rationale

R03 was designed after checking primary market-microstructure research rather than assuming that one-bar volume or order-flow pressure is directly tradable.

## Findings used

1. Cont, Kukanov and Stoikov show that short-horizon price changes are more robustly related to order-flow imbalance than to trade volume alone, and that impact depends inversely on market depth.
   - https://arxiv.org/abs/1011.6402

2. Empirical work on order-flow decomposition indicates that the temporal clustering and composition of trades matter for price impact, supporting accumulated/persistent flow rather than isolated prints.
   - https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4659712_code4840453.pdf?abstractid=4363082

3. Research on metaorders describes persistent same-sign flow, nonlinear impact and later impact decay. This supports testing early-versus-late marginal impact inside an accumulated pressure window.
   - https://arxiv.org/pdf/2506.07711

4. Recent crypto-futures evidence argues for state-first modelling: order flow adds value conditionally on liquidity state rather than replacing it. R03 therefore tests long-history trades + PA first; Books remain a later incremental layer over their shorter history.
   - https://arxiv.org/abs/2607.09230

## Translation into R03

- Do not use total volume as the event by itself.
- Accumulate signed taker notional over 5/10/20 closed bars.
- Compare late marginal impact with early marginal impact.
- Require a causal Price Action sequence before entry.
- Use structure-derived stop and target rather than a fixed bps grid.
- Reject profitable small cells; require >=1,000 conflict-resolved trades and untouched holdout survival.
