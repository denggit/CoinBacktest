# ETH ICT-Macro / Price-Action / OKX Microstructure Research

## 当前结论

研究状态：`NO_LIVE_MODEL_APPROVED`。

截至 2026-08-17，本目录已用 **OKX ETH-USDT-SWAP** 数据完成 2022-01-01 至 2026-08-15 的新一轮严格因果研究。执行主假设为信号完成后的下一根 **1m open**，**2m open** 只作为延迟压力；单边手续费 0.05%，开仓和平仓均收费。所有新模型总 gross cap 不高于 0.75，远低于交易所允许的 15 倍。

没有一个候选同时通过正收益与 `CAGR >= 最大回撤`。因此不能把任何结果称为未来可稳定盈利、绝不爆仓或已批准实盘。

目前唯一没有被新战术腿改善的基线仍是低风险 daily PA/BOS core：

| 指标 | 结果 |
| --- | ---: |
| 最大连续空仓自然日 | 0 |
| 最大连续亏损自然日 | 9 |
| 最大回撤 | 12.08% |
| CAGR | 3.81% |
| 总收益 | 18.84% |
| Calmar | 0.315 |
| 最大 gross exposure | 0.30 |

它不满足 Calmar >= 1，只能作为研究基线，不能实盘带单。

这仍是一个活跃研究判定，不是把 frozen daily PA baseline 批准为实盘。

## V14-V22 新增独立模型族

下列机制都先固定定义、后运行一次；看到结果后没有搜索邻近阈值：

| 模型族 | 1m 代表结果 | 严格结论 |
| --- | --- | --- |
| 论文原尺度 10s pinned flow | 940,268 个极端窗、183,094 个 pin、146,313 个 release；最终接近归零 | 29,360 笔交易累计约 880.7% 账户费用，属于充分经济否证，不是小样本 |
| 7/28/91D 多尺度 BOS | CAGR 3.44%，MDD 19.98%，Calmar 0.17 | 空仓日为 0、对冲 bar 56.9%，但资金曲线仍不稳定 |
| 日线 liquidity reclaim | CAGR -1.20%，MDD 8.27% | 加入 BOS 后 Calmar 降至 0.11 |
| canonical NR7 breakout | CAGR -1.73%，MDD 12.65% | 加入 BOS 后 MDD 扩大到 25.89% |
| 8/32、16/64、32/128 EMA sleeves | CAGR 3.42%，MDD 18.80%，Calmar 0.18 | 双倍费用仍为正，但未解决长回撤 |
| PA core / EMA 机制等权 | CAGR 4.31%，MDD 16.96%，Calmar 0.25 | 分散化提高收益但未压低 MDD |
| 因果 inverse-vol allocator | CAGR 3.99%，MDD 18.07%，Calmar 0.22 | 不拟合收益的风险权重也不能制造稳定 edge |
| 4H displacement + OKX flow | 独立 CAGR -0.35%；core 组合 Calmar 0.30 | 顺势 displacement 无费用后边际 |
| 严格 daily PA Ridge | CAGR -9.81%，MDD 50.56% | 全区间 OOS、月度 refit、标签 purge 后仍失败；不调经济 gate |

所有执行都在所需 bar 完整结束后的第一根 1m open；2m 版本只是固定增加
一分钟延迟。10s trade-bar 数据截至 2026-07-03，所以任何 trade-derived
模型都必须披露末端约六周没有对应 10s 流数据，尽管 OKX 1m K-line 回放持续
至 2026-08-15。

## 新模型族的否证结果

| 模型族 | 代表性独立结果 | 结论 |
| --- | --- | --- |
| 4H PA + OKX flow 月度 walk-forward Logistic | CAGR -1.98%，MDD 25.05% | 方向 AUC 约 0.529，但费用后无边际 |
| 12H 低换手 Logistic + Ridge | CAGR -1.06%，MDD 16.44% | 低换手仍不能覆盖费用 |
| 12H 强正则浅层非线性模型 | CAGR -2.92%，MDD 21.21% | 非线性交互没有转化为收益 |
| quarter-hour 首 10 秒 OKX OI | Ridge CAGR -3.21%；极端 OI CAGR -4.65% | 原始效应约 1.15 bps，远低于 10 bps 往返费 |
| 因果 Portfolio 状态分配器 | 最好 CAGR 6.44%，MDD 17.62%，Calmar 0.365 | 提高收益同时放大回撤 |
| 7D 长持仓 PA/flow Ridge | CAGR -9.38%，MDD 43.87% | 周度状态不稳定 |
| 永久 long + 独立 crash-short | CAGR 1.11%，MDD 30.05% | 日线 hedge 确认和释放过慢 |
| 15m sweep/reclaim + flow absorption | CAGR -11.24%，MDD 43.91% | 1,653 个事件、止损占比过高、费用重 |

完整 41 个候选的用户优先级排序与批准门槛在：

- `ict_pa_v12/results/01_all_candidate_validation.csv`
- `ict_pa_v12/results/02_validation_verdict.json`

## 最重要的审计发现

早期 V2 曾出现高收益，但自动未来扰动测试抓出了两处 Pandas 标签对齐泄漏：4H 数据与 `index + 4h`、日线数据与 `index + 1 day` 在 DataFrame 构造时被按标签重新对齐，等价于读到未来收盘。两处都已改为 `.to_numpy()` 后按位置映射；修复后高收益消失。

新模型继续使用同一原则：

- 完成的 1H 特征在小时结束后才可用，缺少任一 15m 子 bar 的小时直接丢弃。
- 完成的日线特征按位置映射到 D+1，完整日要求 1,440 根 1m 和 96 根 15m。
- quarter-hour OI 只观察边界后的首个完整 10 秒，最早在下一根 1m open 成交。
- 月度 walk-forward 在测试月前 purge 所有仍未结束的标签。
- 手续费按每次仓位变化收费；反向翻仓同时收平旧仓和开新仓费用。
- gross exposure 指标记录的是 0.75 cap 之后的实际仓位，而不是请求仓位。

自动测试：

```text
16 passed
```

pytest 唯一警告是本机 `.pytest_cache` 权限，不影响计算和断言。

## 数据范围

- 价格：OKX ETH-USDT-SWAP 1m K-line，2022-01-01 至 2026-08-15 共 2,430,720 分钟，区间内无缺分钟。
- 成交流：OKX 官方 trades 聚合的 15m 与 10s bar。
- 15m 成交流存在 74 个缺失 bar；特征层禁止把不完整小时或不完整日当成完整观察。
- quarter-hour 首 10 秒记录 157,731 条，建模样本 157,440 条。
- 禁止进入最终模型的数据：Binance、现货、现货/永续套利、资金费套利。

`quarter_hour_effect_source.pdf` 只用于冻结一个可否证的公开研究假设；本地回测没有读取论文使用的 Binance 行情，最终结论完全由 OKX 数据决定。

## 复现

```powershell
python research/eth_ict_price_action_portfolio/17_okx_walkforward_model.py
python research/eth_ict_price_action_portfolio/18_okx_low_turnover_model.py
python research/eth_ict_price_action_portfolio/19_okx_nonlinear_walkforward.py
python research/eth_ict_price_action_portfolio/20_okx_quarter_hour_model.py
python research/eth_ict_price_action_portfolio/21_causal_state_allocator.py
python research/eth_ict_price_action_portfolio/22_okx_weekly_regime_model.py
python research/eth_ict_price_action_portfolio/23_structural_long_crash_hedge.py
python research/eth_ict_price_action_portfolio/24_15m_liquidity_sweep_model.py
python research/eth_ict_price_action_portfolio/27_pinned_10s_release_model.py
python research/eth_ict_price_action_portfolio/28_multiscale_bos_portfolio.py
python research/eth_ict_price_action_portfolio/29_daily_liquidity_reclaim.py
python research/eth_ict_price_action_portfolio/30_nr7_breakout_model.py
python research/eth_ict_price_action_portfolio/31_multispeed_ema_trend.py
python research/eth_ict_price_action_portfolio/32_equal_mechanism_portfolio.py
python research/eth_ict_price_action_portfolio/33_causal_inverse_vol_portfolio.py
python research/eth_ict_price_action_portfolio/34_4h_displacement_continuation.py
python research/eth_ict_price_action_portfolio/35_daily_pa_ridge_model.py
python research/eth_ict_price_action_portfolio/25_consolidated_validation.py
python -m pytest tests/research/eth_ict_price_action_portfolio -q
```

## 实盘边界

历史无清算不等于未来绝不爆仓。跳空、穿仓、交易所风控变更、系统故障、网络中断、流动性枯竭和模型失效都无法由历史回测排除。即使未来发现通过本地门槛的候选，也必须先进行至少 8-12 周 sealed paper-forward，期间不得根据结果改参数；通过后才有资格讨论极小资金灰度，而不是直接使用 15 倍杠杆。
