# ETH AI Trading：三-Sleeve 分阶段研究与实盘落地计划（V4）

## 1. 总目标

把 CoinBacktest 已有的 Trades、OHLCV、OI、Books、Range、Footprint、市场结构和历史研究，收束为一套统一的 ETH AI Trading Bot：

```text
CoinBacktest：研究、训练、回测、压测、模型导出
                         ↓
AetherEdge：实时数据、推理、统一决策、风控、执行
```

不重建第三套完整交易系统。所有市场数据只通过 `src.data_feed`；研究脚本不能直接读取 Raw ZIP、SQLite 或交易所接口。

## 2. 三个独立 Sleeve

| Sleeve | 主要持仓 | 目标波幅 | 方向上下文 | 入场上下文 | 主要退出 |
|---|---:|---:|---|---|---|
| Short-horizon | 5–60分钟 | 0.3%–0.8% | 1m/5m/15m | 1s/1m/5m | 止损、MFE保护、订单流/模型失效、时间上限 |
| Intraday trend | 1–12小时 | 1%–2.5% | 4H/1H/30m | 30m/15m/5m/1m | 结构/状态失效、追踪保护；时间仅安全上限 |
| Swing | 目标触发前，最长5天 | 3%–5% | 1D/4H/1H | 30m/15m/5m/1m | 目标/风险/利润保护；持仓时长是结果，不是约束 |

三个 Sleeve 共享数据和标准输出，但必须保持：

1. 标签独立。
2. 模型独立。
3. 退出逻辑独立。
4. 验收门槛独立。
5. 最终统一成一个 ETH 目标净仓位，不能各自直接下单。

## 3. 固定研究原则

1. 默认研究期：2023-01-01 至 2026-06-30；2022用于warmup。
2. 不随机拆分时序数据；训练、校准、验证之间必须留出覆盖最长标签周期的 embargo。
3. 高周期特征按 `bar_available_time = bar_start_time + timeframe` 对齐，禁止左标签高周期bar提前 `ffill`。
4. 市价完整手续费默认0.11%，另外计入单边0.01%滑点；必须做2x/3x成本压力。
5. 长任务分块读取、缓存、断点续跑并显示进度；禁止一次性把多年1s或Raw Trades放入内存。
6. 数据平台已被大量研究使用。新方向只做轻量 Loader smoke check；结果异常时才做专项数据诊断。
7. 2026H1已在R01整体结果中被观察，因此从R03开始只能称为“锁定样本外”，不是项目级从未看过的纯封存集。禁止用它选择模型、目标、阈值或退出参数。
8. 模型指标必须落到完整交易结果；AUC、IC或Accuracy不能替代扣费后收益、PF、回撤、MFE/MAE和跨期稳定性。

## 4. 阶段总览

| 阶段 | 核心目标 |
|---|---|
| R00 | 固定架构、因果规则和研究门槛 |
| R01 | 归档Trades-only短线诊断基线 |
| R02 | 固定Short / Intraday / Swing三-Sleeve合同 |
| R03 | Swing初版方向/入场基线（已发现15m噪声退出使持仓失真） |
| R03.1 | 用精确目标先于风险标签验证3%–5%开仓MVP |
| R03.2 | 扩展到数月级高周期上下文与连续市场过程，保持标签和退出不变 |
| R04 | 研究1–12小时日内趋势Sleeve |
| R05 | 重做5–60分钟短线Sleeve，不再固定时间退出 |
| R06 | 三Sleeve统一成一个ETH目标仓位 |
| R07 | 每个Sleeve内部测试TCN等序列挑战者 |
| R08 | OI、Books、Range、Footprint等逐项增量消融 |
| R09 | 统一组合与风险管理 |
| R10 | 执行优化 |
| R11 | 受限强化学习退出层 |
| R12 | 模型包与CoinBacktest/AetherEdge重放一致性 |
| R13 | AetherEdge影子实盘 |
| R14 | 小资金真实验收 |

---

## R00 — Framework and governance

固定项目边界、研究窗口、费用、因果时序、产物和停止规则。CoinBacktest只负责研究；AetherEdge只负责实盘运行。

---

## R01 — Short-horizon trades-only diagnostic baseline

使用1s Trade Bar每5秒预测未来1/3/5/15分钟。该阶段证明Trade Flow存在弱方向信息，但当前固定持有退出无法稳定覆盖成本。R01归档为短线诊断，不再代表整个AI Bot，也不通过盲目换大模型救结果。

---

## R02 — Three-sleeve framework and contracts

### 目的

固定三种不同市场过程的研究边界，避免一个模型同时学习几分钟和几天：

```text
Short-horizon → Trade Flow微观过程
Intraday      → 当日趋势与波动扩张
Swing         → 日线/4H方向 + 低周期回踩入场
```

### 统一合同

每个Sleeve只输出：

- `ModelEvidence`
- `TradeCandidate`
- `SleeveContribution`
- `TargetPositionDecision`

研究层不能产生交易所订单。最终由统一决策层给出一个ETH目标净仓位。

### 运行

```bat
python research\eth_ai_trading\02_upgrade_three_sleeve_framework.py
```

报告：`data\reports\research\eth_ai_trading\02_three_sleeve_framework`

---

## R03 — Medium-horizon swing direction and entry baseline

### 研究假设

用户历史上能够盈利的交易过程是：

```text
日线 + 4H判断方向
→ 1H / 30m确认趋势结构
→ 15m / 5m / 1m寻找低风险入场
→ 持有十几个小时到数天
→ 捕捉3%–5%的单边行情
```

R03直接将这个过程模型化，而不是继续只预测固定几分钟后的收益。

### 数据

只读取：

```python
OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="1m")
```

从公共1m Trade Bar因果聚合1D、4H、1H、30m、15m、5m。每个高周期特征的索引移动到真实可用时间后，才对齐15m决策轴。

### 两层模型比较

1. `high_logistic`：仅1D/4H/1H，作为线性方向基准。
2. `high_lightgbm`：仅1D/4H/1H，测试非线性方向模型。
3. `full_lightgbm`：高周期 + 30m/15m/5m/1m，直接测试多周期入场。
4. `hierarchical_lightgbm`：高周期方向分数和全周期入场分数都通过才产生候选。

这不是大参数网格；模型规模、特征和三档信号分位数在运行前固定。

### 标签

不使用“固定持有72小时后的收盘收益”作为唯一标签。首版包含两个低MAE目标：

```text
move3_lowmae_72h：未来72h MFE≥3%，且最大逆向波动≤1.25%
move5_lowmae_120h：未来120h MFE≥5%，且最大逆向波动≤1.75%
```

多空分别训练。标签要求整个未来路径的MAE受控，优先寻找可以安全持有的单边行情。

### 入场

- 每15分钟产生一次候选。
- 只使用该时刻已关闭的所有周期数据。
- 基础执行使用决策后下一分钟Open；压力测试2分钟和5分钟延迟。
- 近期4H结构止损距离超过1.8%的候选直接拒绝，避免用大止损硬扛。

### 出场

不采用固定72h/120h退出。完整交易回测使用：

1. 近期4H Swing结构 + ATR buffer硬止损。
2. 1H跌破/突破EMA20且4H趋势斜率反转时退出。
3. 出现足够强的反方向模型候选时退出。
4. MFE达到1.5%后推进至覆盖成本的盈亏平衡保护。
5. MFE达到2.5%后使用15m/4H波动率动态追踪。
6. 120小时只作安全上限，不能当主要收益来源。

Trailing stop只在一根1m bar完成后更新，从下一根bar生效，避免同bar未来路径乐观。

### Walk-forward

- `WF_2024`：诊断跨期稳定性。
- `WF_2025`：选择模型架构、目标和信号分位数。
- `WF_2026`：只评估2025选出的唯一冠军；禁止用于选择。

### 2025通过门槛

- 至少24笔完整Swing交易。
- 单笔净期望>0，PF>1.20。
- 最大回撤优于-20%。
- 2x成本和5分钟延迟仍盈利。
- 去掉前5大盈利仍为正。
- 至少一半季度盈利。
- 相邻信号分位数仍有正收益，避免单点阈值碰巧。

### 2026锁定样本外门槛

- 至少12笔交易。
- 单笔净期望>0，PF>1.15。
- 2x成本、5分钟延迟、去掉前5大盈利后仍为正。
- 最大回撤优于-20%。

### 运行

```bat
python research\eth_ai_trading\03_swing_multiframe_supervised_baseline.py
```

报告：`data\reports\research\eth_ai_trading\03_swing_baseline`

详细说明：`docs/ETH_AI_TRADING_R03_SWING_BASELINE_RUNBOOK.md`。

---

## R03.1 — Exact-path 3%–5% swing entry MVP

R03报告发现大量交易在15分钟趋势失效规则下退出，无法验证原始3%–5%开仓假设。R03.1不规定最低持仓时间，改为：

1. 使用现有R03多周期特征缓存。
2. 逐1分钟路径生成“目标先于风险线”精确标签；同一分钟冲突按风险先触发。
3. 多空独立训练和验收，允许单边MVP。
4. 取消15分钟趋势失效和模型反转退出。
5. 比较固定风险目标与4H结构+利润保护两套简单出场。
6. 2024必须提供正向支持，2025选择冠军，2026只复核唯一冠军。

R03.1通过后直接进入模型合同、离线/实时特征一致性和AetherEdge影子推理；持仓管理模型留到开仓edge被证明以后。

运行：

```bat
python research\eth_ai_trading\03_1_swing_entry_mvp.py
```

详细说明：`docs/ETH_AI_TRADING_R03_1_SWING_ENTRY_MVP_RUNBOOK.md`。

---


## R03.2 — Long-context 3%–5% swing opportunity model

R03.1 证明 3%–5% 机会在单年可能存在，但 2024 与 2025 的同一套逻辑不稳定。R03.2 只改变市场表达，不改变标签、退出、成本、LightGBM 参数和验收门槛：

1. 日线上下文扩展到 90 / 180 / 365 日。
2. 4H 上下文扩展到约 15 / 30 / 60 / 120 日。
3. 1H 上下文扩展到 3 / 7 / 14 / 30 日。
4. 増加长期高低点位置、趋势年龄、结构抬升或降低、推动与回调、恢复速度、波动率生命周期和订单流持续性。
5. 增加日线、4H、1H 的方向一致性及大趋势中的战术回调关系。
6. 30m 至 1m 仍只负责入场位置，不与高周期平级投票。

R03.2 仍是 LightGBM 因果滚动特征模型，不是单根 K 线，也暂时不使用 TCN / Transformer。若长上下文仍不能让同一套配置跨 2024 与 2025 稳定，则停止通过继续调退出或扩大参数网格救模型。

运行：

```bat
python research\eth_ai_trading\03_2_swing_long_context.py
```

详细说明：`docs/ETH_AI_TRADING_R03_2_LONG_CONTEXT_RUNBOOK.md`。

---

## R04 — Intraday trend sleeve

研究1–12小时、1%–2.5%的趋势扩张。4H/1H/30m定义方向，15m/5m/1m入场；结构和状态反转退出。必须证明它比Swing增加交易频率且不只是重复同一方向暴露。

---

## R05 — Short-horizon sleeve redesign

回到1s缓存，预测5–60分钟达到0.3%–0.8%的概率、MFE、MAE、达到目标时间和坏交易风险。退出使用止损、MFE回吐、订单流/模型反转，时间只作上限。

---

## R06 — Unified multi-sleeve decision layer

三Sleeve统一输出一个ETH目标仓位。高周期Swing拥有方向优先权，Intraday决定当日参与程度，Short用于优化进出场和短期独立机会。禁止简单相加仓位。

---

## R07 — Sequence-model challengers

在每个合适Sleeve内部，用完全相同标签、切分和交易规则比较LightGBM与小型TCN。只有多随机种子、交易结果和CPU推理均提升才保留。

---

## R08 — Incremental data ablation

按Sleeve逐项加入OI、Books、Range、Footprint、流动性墙和历史研究证据。每次只加入一种数据；没有锁定样本外增量价值就删除。

---

## R09 — Portfolio and risk management

固定风险预算、最大ETH净敞口、相关性折扣、回撤降仓、模型漂移和Kill switch。组合提升不能来自重复edge或隐藏杠杆。

---

## R10 — Execution optimisation

以保守市价单为基线，验证等待、拆单和Maker/Taker。必须计入未成交、排队和滑点，不使用乐观成交假设。

---

## R11 — Constrained reinforcement-learning exit overlay

RL只允许选择继续持有、减仓或退出，不能突破硬止损和风险上限，也不能负责凭空发现方向edge。

---

## R12 — Model package and replay parity

导出模型、特征Schema、训练区间、阈值、费用假设和校验和。CoinBacktest与AetherEdge必须对同一市场重放产生相同bar、特征、预测和决策。

---

## R13 — AetherEdge shadow deployment

实时运行但不下单。记录数据缺口、推理延迟、预测漂移和离线重放差异；故障时默认不新增仓位。

---

## R14 — Small-capital live acceptance

小资金、严格风险上限实盘。比较真实手续费、滑点、延迟、PF、回撤和模型漂移；样本足够后再决定推广或拒绝。

---

## R03.3 — Future market-process start forecast

R03.2证明继续堆普通长期滚动统计不能稳定解决3%–5%开仓问题。R03.3先把方向机会与具体入场拆开，并避免重复旧市场状态地图：

1. 自动切分上涨扩张、下跌扩张、高波动双向震荡的独立启动事件；
2. 预测未来6/12/24小时内是否出现新的启动点，而不是识别已经发生的状态；
3. 正样本严格位于启动前，启动后信号进入ongoing/tail-car审核；
4. 复用R03.2长期多周期特征，新增公共5s Trade Bar微观流、吸收、冲击和压力持续性；
5. 使用事件均衡权重，避免一个持续行情因多个重叠预测时点支配训练；
6. 只使用2023–2025进行训练和两次纯OOS，2026H1继续封存；
7. 通过后只进入R03.4低MAE入场模型，不直接形成交易策略。

运行：

```bat
python research\eth_ai_trading\03_3_future_process_forecast.py
```

详细说明：`docs/ETH_AI_TRADING_R03_3_FUTURE_PROCESS_FORECAST_RUNBOOK.md`。

---

## R03.3.1 — Actionable first-alert audit

R03.3的点位级precision无法直接代表真实预警价值。R03.3.1保持模型、标签和2026封存不变，只做严格后验审核：

1. 连续高分按时间合并成独立预警段，每段只看第一次信号；
2. 允许启动后2小时内且过程完成不超过25%的早期确认；
3. 单边预警后至少还剩2.5%目标空间，高波动震荡至少还剩3%双向区间；
4. 输出独立预警可交易命中率、事件级覆盖率、首次预警领先时间、过程进度和剩余机会；
5. 若第一次预警普遍已经晚或剩余空间不足，停止状态预测路线；若剩余空间足但误报多，再处理标签与概率稳定性。

运行：

```bat
python research\eth_ai_trading\03_3_1_process_alert_value_audit.py
```

详细说明：`docs/ETH_AI_TRADING_R03_3_1_ALERT_VALUE_AUDIT_RUNBOOK.md`。

---

## R03.3.2 — Continuous future opportunity intensity

R03.3.1确认状态预警通常不算太晚，但离散事件阈值导致误报率高且跨年分数漂移。R03.3.2改为连续预测未来机会强度：

1. 当下市场状态是完全因果的多周期状态向量，不使用未来状态反推；
2. 预测未来6/12小时完整区间、最大单向空间、双向空间和ATR标准化区间；
3. 比较长期、多周期和多周期+5s微观模型；
4. 以Rank IC、预测十分位单调性和Top Decile实际提升做跨年验收；
5. 2026H1继续封存；
6. 若连续强度排序仍失败，停止继续堆普通价格和Trade Bar特征。

运行：

```bat
python research\eth_ai_trading\03_3_2_future_opportunity_intensity.py
```

详细说明：`docs/ETH_AI_TRADING_R03_3_2_FUTURE_INTENSITY_RUNBOOK.md`。



---

## R03.3.3 — Multi-timescale market-state continuity

R03.3.2已经证明多周期因果状态能够稳定排序未来6小时机会强度，但每15分钟独立打分仍可能产生状态抖动。R03.3.3不直接交易，而是构建后续模型共享的状态上下文：

1. 2020—2021普通1m OHLCV通过`OKXDataLoader`读取，2022以后通过现有`OKXTradeBarLoader`读取；
2. Universal模型只使用全历史共有的OHLCV派生特征，Trade增强模型单独验证订单流增量；
3. 分别表达战略、战术、入场和活跃度状态，允许大周期与小周期同时处于不同过程；
4. 每层使用连续分数、因果迟滞状态、状态年龄、翻转率和跨层对齐；
5. 监督预测未来状态是否持续及转换风险，不把状态模型直接用作开仓器；
6. 输出不同状态组合对应的未来6小时实际机会厚度，保持研究围绕赚钱目标；
7. 增加训练年份归因矩阵，查明2025表现更好来自更多训练数据还是市场本身更连续；
8. 2026H1继续封存。

运行：

```bat
python research\eth_ai_trading\03_3_3_market_state_continuity.py
```

详细说明：`docs/ETH_AI_TRADING_R03_3_3_MARKET_STATE_CONTINUITY_RUNBOOK.md`。

---

## R03.3.3.1 — Market-state continuity calibration and audit

R03.3.3 已证明部分状态连续性可预测，但战略状态固定阈值不可达，且高 AUC 可能主要来自状态年龄和迟滞边界。R03.3.3.1 做最小修正与严格审计：

1. 战略层使用前一日及更早365天分布的因果分位阈值；
2. 持续标签要求整个预测窗口内没有任何切换；
3. 增加状态年龄、边界距离、年龄+边界+当前状态机械基准；
4. 完整模型只有在2024和2025都超过最佳机械基准时，才认定存在额外市场过程信息；
5. 连续低持续概率信号合并成独立转换预警，统计预警频率、成功率、领先时间和事件覆盖；
6. 所有状态仍然仅作为后续方向、入场、持仓和风险模型的上下文；
7. 2026H1继续封存。

运行：

```bat
python research\eth_ai_trading\03_3_3_1_market_state_continuity_audit.py
```

详细说明：`docs/ETH_AI_TRADING_R03_3_3_1_MARKET_STATE_CONTINUITY_AUDIT_RUNBOOK.md`。

