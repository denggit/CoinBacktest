# ETH AI Trading Research

## R00：初始化研究框架

```bat
python research\eth_ai_trading\00_initialize_framework.py
```

## R01：Trades-only短线诊断基线

```bat
python research\eth_ai_trading\01_trades_only_supervised_baseline.py
```

R01已归档为短线方向信息和固定时间退出诊断，不代表整个AI Bot。

## R02：三-Sleeve框架

```bat
python research\eth_ai_trading\02_upgrade_three_sleeve_framework.py
```

## R03：中线Swing多周期监督学习

```bat
python research\eth_ai_trading\03_swing_multiframe_supervised_baseline.py
```

R03使用日线/4H/1H判断方向，30m/15m/5m/1m优化入场，目标是未来3%–5%且MAE受控的行情，退出采用结构、状态和利润保护，不以固定持仓时间为主要规则。

## 开发边界

- 稳定、可复用算法放入正确的 `src/` 功能域。
- 编号研究脚本不得 import 其他研究脚本。
- CoinBacktest 不放 AetherEdge 下单代码。
- 所有市场数据必须通过 `src.data_feed`。

## R03.1：3%–5% Swing 开仓 MVP

```bat
python research\eth_ai_trading\03_1_swing_entry_mvp.py
```

R03.1复用R03特征缓存，将标签修正为目标先于风险线的真实路径结果；多空独立验证，不设最低持仓时间，也不再用15分钟噪声退出。

## R03.2：长上下文 3%–5% Swing 机会模型

```bat
python research\eth_ai_trading\03_2_swing_long_context.py
```

R03.2 保持 R03.1 的标签、退出、成本和模型参数不变，只把日线扩展到 365 天、4H 扩展到约 120 天、1H 扩展到约 30 天，并加入趋势年龄、长期结构、推动/回调和波动生命周期。缓存与 R03/R03.1 隔离。

## R03.3：未来市场过程启动概率

```bat
python research\eth_ai_trading\03_3_future_process_forecast.py
```

R03.3不再把长期方向与当前低MAE入场压进同一个标签。它先切分上涨扩张、下跌扩张和高波动双向震荡的独立启动事件，再预测未来6/12/24小时是否启动。启动后的信号只计入tail-car，不算预测成功。模型对比长期、多周期以及多周期+5s Trade Flow，2026H1保持封存。

## R03.3.1：独立预警与剩余交易空间审核

```bat
python research\eth_ai_trading\03_3_1_process_alert_value_audit.py
```

R03.3.1不增加特征或改事件标签，只把连续高分合并为独立预警段，并只审核第一次预警。允许状态刚启动后的短暂早期确认，但必须仍有足够方向目标或双向区间；同时统计独立预警成功率与独立事件覆盖率，判断R03.3到底是可交易早期预警还是晚确认。

## R03.3.2：连续未来机会强度预测

```bat
python research\eth_ai_trading\03_3_2_future_opportunity_intensity.py
```

R03.3.2不再预测某个人工稀疏事件是否发生，而是连续预测未来6/12小时的完整区间、最大单向空间、双向空间和相对当前4H ATR的异常倍数。当下市场状态定义为决策时刻已经可见的多周期因果状态向量，不使用未来结果反推。报告以跨年Rank IC、十分位单调性和Top Decile实际机会提升判断是否值得作为AI Bot环境评分。



## R03.3.3：多周期市场状态连续性与转换

```bat
python research\eth_ai_trading\03_3_3_market_state_continuity.py
```

R03.3.3不是开仓模型。它把2020—2021普通1m K线与2022以后1m Trade Bar统一为OHLCV状态历史，分别表达战略、战术、入场和活跃度状态，并研究状态年龄、翻转、持续概率和转换风险。Universal分支只使用所有年份共有的OHLCV特征，Trade增强分支只在真实Trade特征存在的年份做增量对照。状态输出用于后续方向、入场、持仓和风险模型，不直接触发订单。

## R03.3.3.1：市场状态连续性小修正与审计

```bat
python research\eth_ai_trading\03_3_3_1_market_state_continuity_audit.py
```

R03.3.3.1 修复战略状态固定阈值不可达的问题，使用严格滞后一日的365天因果分位阈值；持续标签改为整个窗口内不可发生任何状态切换。完整模型还必须和状态年龄、迟滞边界距离及当前状态组成的机械基准比较，并将连续低持续概率信号合并为独立转换预警段审核。市场状态仍只作为方向、入场、持仓和风险模型的辅助上下文。


## R03.4：市场状态上下文对开仓价值模型的增量消融

```bat
python research\eth_ai_trading\03_4_state_context_ablation.py
```

R03.4保持未来6小时多空开仓价值目标、训练窗口、LightGBM参数、校准阈值和真实成本完全一致，只改变是否加入战略、战术、入场、活跃度及活跃持续概率。2024与2025分别纯OOS，2026继续封存。6小时收盘收益只用于同口径诊断，不是最终实盘退出方案。

## R03.4.2.4：q70跨年开仓池审核

```bat
python research\eth_ai_trading\03_4_2_4_q70_cross_year_audit.py
```

R03.4.2.4先把q70与q90的开仓Edge独立审核清楚。它不依赖恢复、续持或长持模型，因此2024不会再因为持仓标签样本不足而被整年跳过。固定6小时只作为冻结诊断基准；最终退出仍将通过后续路径研究改为结构化、无机械持仓上限的管理。

## R03.4.2.5：q70极高置信坏单退出Overlay

```bat
python research\eth_ai_trading\03_4_2_5_q70_failure_overlay.py
```

R03.4.2.5保留q70-q80、q80-q90和q90+全部分数层。T+60只做风险预警；T+180只有在失败概率持续极高、价格跌破结构且恢复不足时才允许提前退出。更高分事件需要更强的失败证据。3%宽安全底线仅用于黑天鹅风险诊断，并按触发后的下一分钟开盘执行。固定6小时仍只作为冻结收益基准，持仓中评分升级仅做未来分层加仓研究，不在本阶段执行加仓。

## R03.4.2.6：增量持仓价值与非时间退出信号

```bat
python research\eth_ai_trading\03_4_2_6_incremental_hold_value.py
```

R03.4.2.6不再把机器学习概率直接映射成止损。它在T+180m、T+360m、T+720m、T+24h和T+48h比较“现在平仓”与“继续到后续决策节点”的增量收益和新增回撤，训练价格路径结构模型预测增量持仓价值。检查点只是重新决策，不是强制退出；120小时只是标签与删失窗口。q70完整保留并继续拆分q70-q80、q80-q90、q90+，开仓分数只做消融。

## 跨窗口交接

接手本研究前必须依次阅读：

1. `RESEARCH_HANDOFF.md`
2. `COMPLETED_WORK.md`
3. `OPEN_ITEMS_AND_ROADMAP.md`
4. `DECISION_LOG.md`
5. 当前阶段 `R03_4_2_10_DELIVERY.md`

## R03.4.2.7：q70因果结构状态机与非时间退出审核

```bat
python research\eth_ai_trading\03_4_2_7_non_time_structural_exit.py
```

R03.4.2.7停止按年份挑选持仓机器学习冠军，改用一套同时运行于2024和2025的因果结构状态机。候选策略没有固定或最大持仓时间；只在确认的结构跌破、失败收回、低高点/低低点延续、利润回吐伴随结构转弱或宽灾难底线触发时退出。OOS结束和数据缺口只记录为右删失盯市，不属于策略退出。固定6小时仅作为冻结收益基准。

## R03.4.2.8A：持仓中新q70信号图谱与严格Tranche资格门

```bat
python research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas.py
```

R03.4.2.8A不增加仓位，先把单仓`failed_reclaim`持仓期间出现的新q70信号拆成健康趋势、回撤修复、危险摊低和不明确。实测结果为`FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY`：严格健康/修复子集在2024只有47个、2025只有28个，且收益集中与延迟压力未同时通过。这个结果否定的是“只选极少完美加仓信号”，不是否定完整q70开仓Edge，也不是最终账户级双槽位结论。

## R03.4.2.8B：双风险槽位账户级覆盖研究

```bat
python research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account.py
```

R03.4.2.8B不重新筛成极少信号，直接比较P0单槽、P1等风险双槽、P2 0.65R+0.35R和P3带危险摊低保护的0.65R+0.35R。每个虚拟Tranche独立使用冻结的`failed_reclaim`和3%灾难保护；固定6小时只做诊断。报告使用分钟级账户盯市、真实双边成本、最大两个Tranche和1R总预设槽位上限，重点判断能否恢复q70覆盖率并同时改善2024/2025账户收益。

## R03.4.2.9：结构保护止损与动态风险释放

```bat
python research\eth_ai_trading\03_4_2_9_dynamic_risk_release.py
```

R03.4.2.9不再静态把首仓削弱为0.65R或0.5R。每个独立q70首仓仍从1R开始，3%只作为最外层灾难保护；研究最新确认结构低点和落后一层确认低点形成的、只升不降的真实硬保护位。保护位只能在15m结构bar完整关闭后生成，并于下一根1m open生效。只有该真实止损已经减少原仓剩余风险时，第二Tranche才可使用已释放风险，最多两个Tranche，账户实时剩余风险不超过1R。`failed_reclaim`继续负责正常非时间结构退出，固定6小时仍只用于诊断。


## R03.4.2.10：结构驱动部分减仓与q70风险迁移

```bat
python research\eth_ai_trading\03_4_2_10_risk_migration.py
```

R03.4.2.9实测为`FAIL_NO_ROBUST_STRUCTURE_PROTECTION`：最新结构低点几乎取代全部`failed_reclaim`，落后一层结构在2025有效但2024明显损失收益，因此15m Pivot硬止损路线停止。R03.4.2.10保留3%灾难保护和`failed_reclaim`软结构退出，比较两类真实账户动作：已证明趋势第一次进入软BROKEN且旧仓不亏时减仓25%/50%；后续q70出现时，先真实减少旧仓或使用已部分减仓释放的风险，再把最多0.35R/0.50R迁移到新Tranche。每个风险周期从完整1R首仓开始，最多两个Tranche，总初始亏损风险预算不超过1R，不使用固定时间退出。
