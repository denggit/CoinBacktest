# R03.4.2.13 Delivery

Status: **READY / LOCAL FULL-DATA RUN PENDING**

## Research objective

Determine whether q70-q80, q80-q90 and q90+ deserve different initial risk, or whether equal one-R is the more robust deployment rule.

## Formal candidates

- T1: 0.75R / 0.90R / 1.00R
- T2: 0.75R / 1.00R / 1.00R
- T3: 0.60R / 0.80R / 1.00R

## Diagnostics

- E075 equal 0.75R
- E100 equal 1.00R anchor
- E125 equal 1.25R aggressive scaling diagnostic

No trade is removed. All exits remain exactly those produced by C2.

## Validation

```text
R03.4.2.13专项：6 passed
R03.4.2.12 + 2.13：15 passed
R03.4.2.7～2.13相关回归：65 passed
AI Research + Data Feed：204 passed
```

The complete repository still has five pre-existing collection errors from missing liquidity/analyze-tool modules. Import-boundary audit still reports 155 pre-existing violations outside this stage; this patch adds none.
