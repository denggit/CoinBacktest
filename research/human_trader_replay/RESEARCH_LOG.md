# Human Trader Replay Research Log

## R01 - Manual Setup Execution Audit

### 目的

把 Replay Lab 中真实的人工决策与本地 OKX 1m 未来路径连接起来，优先区分：

- Direction failure
- Early entry / premature confirmation
- Stop-first-then-target
- Manual Bias conflict
- Same-side re-entry evidence

### 明确不做

- 不重新定义 ICT/MSS/FVG。
- 不用未来数据产生入场信号。
- 不搜索最优 SL 百分比。
- 不因为当前样本少就微调规则。

### 当前数据兼容

- 只把 active branch 的 `TRADE_CLOSED` 视为完整生命周期交易。
- rewind 后归档的 `is_active=0` 事件不进入主统计。
- 旧版只有 LONG/SHORT、没有 `TRADE_CLOSED` 的记录单独输出到 `legacy_unclosed_trade_candidates.csv`，不擅自猜输赢。

### 后续

继续积累 Blind Replay Episode，用完全相同的 R01 脚本重跑；样本扩大前不冻结新的过滤规则。
